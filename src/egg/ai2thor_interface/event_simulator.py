# pyright: reportExplicitAny=none, reportAny=none
from datetime import datetime
from scipy.spatial.distance import cdist
import numpy as np
from typing import Any
from ai2thor.controller import Controller
from ai2thor.server import Event
import logging

from egg.utils.data import Ai2ThorObjectMetadata
from egg.utils.timestamp import datetime_to_ns
from egg.ai2thor_interface.actions import ActionType

from egg.graph.egg import EGG
from egg.graph.node import AgentNode, ObjectNode
from egg.utils.geometry import Position, Rotation
from egg.utils.logger import getLogger
import egg.ai2thor_interface.navigation as ai2thor_nav

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="ai2thor_interfaceh/event_simulator.log",
)


class EventSimulator:
    def __init__(
        self, egg: EGG, ai2thor_controller: Controller, crouch_height: float = 0.5
    ) -> None:
        self.egg: EGG = egg
        self.ai2thor_controller: Controller = ai2thor_controller
        self.crouch_height: float = crouch_height

    def get_sim_previous_event(self) -> Event:
        last_event = self.ai2thor_controller.last_event
        assert isinstance(last_event, Event), "Cannot retrieve previous event"
        return last_event

    def is_sim_event_successful(self) -> bool:
        return self.get_sim_previous_event().metadata["lastActionSuccess"]

    def get_sim_grid_size(self) -> float:
        return self.ai2thor_controller.initialization_parameters["gridSize"]

    def get_sim_objects(self) -> list[dict[str, Any]]:
        return self.get_sim_object_metadata()

    def get_sim_spawnable_positions_at_receptacle(
        self, receptacle_name: str
    ) -> list[Position]:
        spawnable_positions = self.ai2thor_controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=receptacle_name,
            anywhere=False,
        ).metadata["actionReturn"]
        return [Position.model_validate(p) for p in spawnable_positions]

    def get_sim_visible_objects(self) -> list[str]:
        last_event = self.get_sim_previous_event()
        instance_detections2D = last_event.instance_detections2D
        if instance_detections2D:
            return list(instance_detections2D.keys())
        else:
            return []

    def get_sim_agent_state(self) -> tuple[Position, Rotation, float, bool]:
        event = self.get_sim_previous_event()
        agent_metadata: dict[str, Any] = event.metadata["agent"]
        return (
            Position.model_validate(agent_metadata["position"]),
            Rotation.model_validate(agent_metadata["rotation"]),
            agent_metadata["cameraHorizon"],
            bool(agent_metadata["isStanding"]),
        )

    def get_sim_object_metadata(self, object_name: str) -> Ai2ThorObjectMetadata | None:
        event = self.get_sim_previous_event()
        all_objects_metadata: list[dict[str, Any]] = event.metadata["objects"]
        for metadata in all_objects_metadata:
            if metadata["objectId"] == object_name:
                obj_metadata = Ai2ThorObjectMetadata.model_validate(metadata)
                return obj_metadata
        logger.warning(f"Could not retrieve metadata of {object_name}")
        return None

    def get_object_state_from_sim_metadata(
        self, object_name: str
    ) -> ObjectNode.ObjectState | None:
        object_metadata = self.get_sim_object_metadata(object_name=object_name)
        assert object_metadata, f"Could not retrieve metadata of {object_name}"
        return ObjectNode.ObjectState.from_ai2thor(object_metadata)

    def get_object_capabilities_from_sim_metadata(
        self, object_name: str
    ) -> ObjectNode.ObjectCapabilities | None:
        object_metadata = self.get_sim_object_metadata(object_name=object_name)
        assert object_metadata, f"Could not retrieve metadata of {object_name}"
        return ObjectNode.ObjectCapabilities.from_ai2thor(object_metadata)

    def get_sim_object_being_held_by_agent(self) -> list[dict[str, str]]:
        return self.get_sim_previous_event().metadata["inventoryObjects"]

    def get_sim_reachable_positions(self) -> list[Position]:
        metadata = self.ai2thor_controller.step(action="GetReachablePositions").metadata
        reachable_positions: list[Position] = []
        for pos in metadata["actionReturn"]:
            reachable_positions.append(Position.model_validate(pos))
        return reachable_positions

    def get_sim_interactable_poses(
        self,
        object_name: str,
        horizons: list[float],
        is_standing: list[bool],
    ) -> list[tuple[Position, Rotation, bool, float]]:
        event = self.ai2thor_controller.step(
            action="GetInteractablePoses",
            objectId=object_name,
            horizons=horizons,
            standings=is_standing,
        )

        poses: list[str, Any] = event.metadata["actionReturn"]
        interactable_poses: list[tuple[Position, Rotation, bool, float]] = [
            (
                Position(x=float(p["x"]), y=float(p["y"]), z=float(p["z"])),
                Rotation(x=0, y=float(p["rotation"]), z=0),
                bool(p["standing"]),
                float(p["horizon"]),
            )
            for p in poses
        ]
        agent_position, _, _, _ = self.get_sim_agent_state()
        interactable_distances = [
            p[0].euclidean(agent_position) for p in interactable_poses
        ]
        sorted_interactable_poses: list[tuple[Position, Rotation, bool, float]] = []
        for j in np.argsort(interactable_distances):
            sorted_interactable_poses.append(interactable_poses[j])
        return sorted_interactable_poses

    def get_closest_sim_interactable_poses(
        self, object_name: str, is_standing: list[bool]
    ) -> tuple[Position, Rotation]:

        agent_position, _, agent_horizon, _ = self.get_sim_agent_state()
        interactable_poses = self.get_sim_interactable_poses(
            object_name=object_name, horizons=[agent_horizon], is_standing=is_standing
        )
        interactable_positions: list[Position] = [p[0] for p in interactable_poses]
        interactable_angles: list[float] = [p[1].y for p in interactable_poses]

        R = np.asarray(
            [p.as_numpy_2d().reshape(-1)[:2] for p in interactable_positions], float
        )
        robot_xz = agent_position.as_numpy_2d().reshape(-1)[:2]

        # Distances
        d_robot = cdist(R, robot_xz[None, :], metric="euclidean").ravel()

        # Primary: min d_target; Secondary: min d_robot
        best_idx = int(np.sort(d_robot)[0])
        desired_yaw_deg = interactable_angles[best_idx]

        return interactable_positions[best_idx], Rotation(x=0, y=desired_yaw_deg, z=0)

    def get_picked_up_egg_object_node(self, object_name: str) -> ObjectNode:
        _, object_node = self.egg.spatial.get_object_node_by_name(object_name)
        assert isinstance(
            object_node, ObjectNode
        ), f"{object_node} does not exist in EGG"
        assert (
            object_node.capabilities.is_pickupable
        ), f"Trying pick and place but {object_node} is not pickupable."
        return object_node

    def get_receptacle_egg_object_node(self, receptacle_name: str) -> ObjectNode:
        _, receptacle_object_node = self.egg.spatial.get_object_node_by_name(
            receptacle_name
        )
        assert isinstance(
            receptacle_object_node, ObjectNode
        ), f"{receptacle_object_node} does not exist in EGG"
        assert (
            receptacle_object_node.capabilities.is_receptacle
        ), f"{receptacle_name} is not a receptacle."
        return receptacle_object_node

    def move_to_position(
        self,
        nav_position: Position,
        nav_angle: Rotation,
        teleport: bool = False,
        standing: bool = True,
    ) -> bool:
        if teleport:
            _ = self.ai2thor_controller.step(
                action="Teleport",
                position=nav_position.model_dump(),
                rotation=nav_angle.model_dump(),
                standing=standing,
            )
            return self.is_sim_event_successful()
        else:
            agent_position, agent_rotation, _, _ = self.get_sim_agent_state()
            _, path_planning_cmds = ai2thor_nav.plan_path_and_command(
                start_pos=agent_position,
                start_yaw_deg=agent_rotation.y,
                reachable_positions=self.get_sim_reachable_positions(),
                goal_pos=nav_position,
                grid_size=self.get_sim_grid_size(),
            )
            for cmd in path_planning_cmds:
                _ = self.ai2thor_controller.step(action=cmd)
                if not self.is_sim_event_successful():
                    _ = self.ai2thor_controller.step(action="Done")
                    return False

            _ = self.ai2thor_controller.step(action="Done")
            return True

    def execute_pick(
        self,
        pick_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        _, _, agent_horizon, _ = self.get_sim_agent_state()

        success_pick: bool = False

        # TODO: Analyze interactive obj height and select standing / crouching
        for standing in [True, False]:
            if success_pick:
                break
            interactable_poses = self.get_sim_interactable_poses(
                object_name=pick_object_name,
                horizons=[agent_horizon],
                is_standing=[standing],
            )

            for pose in interactable_poses:
                pick_nav_position: Position = pose[0]
                pick_nav_angle: Rotation = pose[1]
                _ = self.move_to_position(
                    nav_position=pick_nav_position,
                    nav_angle=pick_nav_angle,
                    teleport=teleport,
                    standing=standing,
                )
                _ = self.ai2thor_controller.step(
                    action="PickupObject",
                    objectId=pick_object_name,
                    forceAction=False,
                    manualInteract=False,
                )
                if self.is_sim_event_successful():
                    logger.info(
                        f"Successfully picked up {pick_object_name} at {pick_nav_position}, {'standing' if standing else 'crouching'}"
                    )
                    success_pick = True
                    break
                else:
                    logger.warning(
                        f"Unable to pick up {pick_object_name} at {pick_nav_position}, {'standing' if standing else 'crouching'}"
                    )
        if success_pick:
            self.update_agent_state(
                timestamp=timestamp, holding=pick_object_name
            )
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(f"Unable to pick up {pick_object_name} after all attempts!")
        return success_pick

    def execute_close(
        self,
        openable_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        _, openable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=openable_object_name
        )
        assert isinstance(
            openable_object_node, ObjectNode
        ), f"{openable_object_node} does not exists"
        success_close: bool = False
        if openable_object_node.capabilities.is_openable:
            _, _, agent_horizon, _ = self.get_sim_agent_state()
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_close:
                    break
                interactable_poses = self.get_sim_interactable_poses(
                    object_name=openable_object_name,
                    horizons=[agent_horizon],
                    is_standing=[standing],
                )
                for pose in interactable_poses:
                    close_nav_position: Position = pose[0]
                    close_nav_angle: Rotation = pose[1]
                    _ = self.move_to_position(
                        nav_position=close_nav_position,
                        nav_angle=close_nav_angle,
                        teleport=teleport,
                        standing=standing,
                    )
                    _ = self.ai2thor_controller.step(
                        action="CloseObject",
                        objectId=openable_object_name,
                        forceAction=False,
                    )
                    if self.is_sim_event_successful():
                        logger.info(
                            f"Successfully closed {openable_object_name} at {close_nav_position}, {'standing' if standing else 'crouching'}"
                        )
                        success_close = True
                        break
                    else:
                        logger.warning(
                            f"Could not close {openable_object_name} from {close_nav_position}, {'standing' if standing else 'crouching'}"
                        )
        if success_close:
            self.update_agent_state(timestamp=timestamp)
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(
                f"Unable to close {openable_object_name} after all attempts!"
            )
        return success_close

    def execute_toggle_on(
        self,
        toggleable_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        _, toggleable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=toggleable_object_name
        )
        success_toggle_on: bool = False
        assert isinstance(
            toggleable_object_node, ObjectNode
        ), f"{toggleable_object_node} does not exists"
        if toggleable_object_node.capabilities.is_toggleable:
            _, _, agent_horizon, _ = self.get_sim_agent_state()
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_toggle_on:
                    break
                interactable_poses = self.get_sim_interactable_poses(
                    object_name=toggleable_object_name,
                    horizons=[agent_horizon],
                    is_standing=[standing],
                )
                for pose in interactable_poses:
                    toggle_nav_position: Position = pose[0]
                    toggle_nav_angle: Rotation = pose[1]
                    _ = self.move_to_position(
                        nav_position=toggle_nav_position,
                        nav_angle=toggle_nav_angle,
                        teleport=teleport,
                        standing=standing,
                    )
                    _ = self.ai2thor_controller.step(
                        action="ToggleObjectOn",
                        objectId=toggleable_object_name,
                        forceAction=False,
                    )
                    if self.is_sim_event_successful():
                        logger.info(
                            f"Successfully toggled on {toggleable_object_name} at {toggle_nav_position}, {'standing' if standing else 'crouching'}"
                        )
                        success_toggle_on = True
                        break
                    else:
                        logger.warning(
                            f"Could not toggle on {toggleable_object_name} from {toggle_nav_position}, {'standing' if standing else 'crouching'}"
                        )
        if success_toggle_on:
            self.update_agent_state(timestamp=timestamp)
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(
                f"Unable to toggle on {toggleable_object_name} after all attempts!"
            )
        return success_toggle_on

    def execute_toggle_off(
        self,
        toggleable_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        _, toggleable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=toggleable_object_name
        )
        assert isinstance(
            toggleable_object_node, ObjectNode
        ), f"{toggleable_object_node} does not exists"
        success_toggle_off: bool = False
        if toggleable_object_node.capabilities.is_toggleable:
            _, _, agent_horizon, _ = self.get_sim_agent_state()
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_toggle_off:
                    break
                interactable_poses = self.get_sim_interactable_poses(
                    object_name=toggleable_object_name,
                    horizons=[agent_horizon],
                    is_standing=[standing],
                )
                for pose in interactable_poses:
                    toggle_nav_position: Position = pose[0]
                    toggle_nav_angle: Rotation = pose[1]
                    _ = self.move_to_position(
                        nav_position=toggle_nav_position,
                        nav_angle=toggle_nav_angle,
                        teleport=teleport,
                        standing=standing,
                    )
                    _ = self.ai2thor_controller.step(
                        action="ToggleObjectOff",
                        objectId=toggleable_object_name,
                        forceAction=False,
                    )
                    if self.is_sim_event_successful():
                        logger.info(
                            f"Successfully toggled off {toggleable_object_name} at {toggle_nav_position}, {'standing' if standing else 'crouching'}"
                        )
                        success_toggle_off = True
                        break
                    else:
                        logger.warning(
                            f"Could not toggle off {toggleable_object_name} from {toggle_nav_position}, {'standing' if standing else 'crouching'}"
                        )
        if success_toggle_off:
            self.update_agent_state(timestamp=timestamp)
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(
                f"Unable to toggle off {toggleable_object_name} after all attempts!"
            )
        return success_toggle_off

    def execute_open(
        self,
        openable_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        _, openable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=openable_object_name
        )
        success_open: bool = False
        assert isinstance(
            openable_object_node, ObjectNode
        ), f"{openable_object_node} does not exists"
        if openable_object_node.capabilities.is_openable:
            _, _, agent_horizon, _ = self.get_sim_agent_state()
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_open:
                    break
                interactable_poses = self.get_sim_interactable_poses(
                    object_name=openable_object_name,
                    horizons=[agent_horizon],
                    is_standing=[standing],
                )
                for pose in interactable_poses:
                    open_nav_position: Position = pose[0]
                    open_nav_angle: Rotation = pose[1]
                    _ = self.move_to_position(
                        nav_position=open_nav_position,
                        nav_angle=open_nav_angle,
                        teleport=teleport,
                        standing=standing,
                    )
                    _ = self.ai2thor_controller.step(
                        action="OpenObject",
                        objectId=openable_object_name,
                        openness=1,
                        forceAction=False,
                    )
                    openable_object_state = self.get_object_state_from_sim_metadata(
                        object_name=openable_object_name
                    )
                    assert isinstance(openable_object_state, ObjectNode.ObjectState)
                    if (
                        openable_object_state.openness == 1.0
                        and openable_object_state.is_open
                    ):
                        logger.info(
                            f"Successfully opened {openable_object_name} from {open_nav_position}, {'standing' if standing else 'crouching'}"
                        )
                        success_open = True
                        break
                    else:
                        logger.warning(
                            f"Could not open {openable_object_name} from {open_nav_position}, {'standing' if standing else 'crouching'}"
                        )
        if success_open:
            self.update_agent_state(timestamp=timestamp)
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(
                f"Unable to open {openable_object_name} after all attempts!"
            )
        return success_open

    def execute_place(
        self,
        held_object: str,
        receptacle_object_name: str,
        timestamp: int | None = None,
        teleport: bool = False,
    ) -> bool:
        success_place: bool = False
        held_object_info = self.get_sim_object_being_held_by_agent()
        if not held_object_info:
            logger.warning(f"Robot not holding any object but trying to place")
            return False
        held_object_name = held_object_info[0]["objectId"]
        assert held_object_name == held_object

        _, _, agent_camera_horizon, _ = self.get_sim_agent_state()

        # TODO: Analyze interactive obj height and select standing / crouching
        for standing in [True, False]:
            if success_place:
                break
            interactable_poses = self.get_sim_interactable_poses(
                object_name=receptacle_object_name,
                horizons=[agent_camera_horizon],
                is_standing=[standing],
            )
            for pose in interactable_poses:
                place_nav_position: Position = pose[0]
                place_nav_angle: Rotation = pose[1]
                _ = self.move_to_position(
                    nav_position=place_nav_position,
                    nav_angle=place_nav_angle,
                    teleport=teleport,
                    standing=standing,
                )
                _ = self.ai2thor_controller.step(
                    action="PutObject",
                    objectId=receptacle_object_name,
                    forceAction=False,
                    placeStationary=True,
                )
                if self.is_sim_event_successful():
                    logger.info(
                        f"Succesfully placed {held_object_name} on {receptacle_object_name} at {place_nav_position}, {'standing' if standing else 'crouching'}."
                    )
                    success_place = True
                    break
                else:
                    logger.warning(
                        f"Unable to place {held_object_name} at {receptacle_object_name} at {place_nav_position}, {'standing' if standing else 'crouching'}, retrying with spawning"
                    )
                    success_place = self.execute_spawn_on_receptacle(
                        object_name=held_object_name,
                        receptacle_name=receptacle_object_name,
                    )
                    if success_place:
                        break
        if success_place:
            self.update_agent_state(timestamp=timestamp)
            self.update_visible_objects_states(timestamp=timestamp)
            _ = self.ai2thor_controller.step(action="Done")
        else:
            logger.warning(
                f"Unable to place {held_object_name} at {receptacle_object_name} after all attempts."
            )
        return success_place

    def execute_spawn(self, object_name: str, position: Position) -> bool:
        _ = self.ai2thor_controller.step(
            action="PlaceObjectAtPoint",
            objectId=object_name,
            position=position.model_dump(),
        )
        return self.is_sim_event_successful()

    def execute_spawn_on_receptacle(
        self,
        object_name: str,
        receptacle_name: str,
    ) -> bool:
        spawnable_positions = self.get_sim_spawnable_positions_at_receptacle(
            receptacle_name=receptacle_name
        )
        for position in spawnable_positions:
            spawn_success = self.execute_spawn(
                object_name=object_name, position=position
            )
            if spawn_success:
                return True
        logger.warning(f"Unable to spawn {object_name} at {receptacle_name}.")
        return False

    def update_agent_state(
        self, timestamp: int | None, holding: str | None = None, agent_id: int = 0
    ):
        (
            agent_position,
            agent_rotation,
            agent_camera_horizon,
            agent_is_standing,
        ) = self.get_sim_agent_state()
        agent_state = AgentNode.AgentState(
            position=agent_position,
            rotation=agent_rotation,
            camera_horizon=agent_camera_horizon,
            is_standing=agent_is_standing,
            holding=holding,
        )
        self.egg.spatial.update_agent_state(
            agent_node_id=agent_id, new_agent_state=agent_state, timestamp=timestamp
        )

    def update_object_state(self, object_name: str, timestamp: int | None):
        if timestamp is None:
            timestamp = datetime_to_ns(datetime.now())
        object_node_id, _ = self.egg.spatial.get_object_node_by_name(
            node_name=object_name
        )
        if object_node_id:
            object_state = self.get_object_state_from_sim_metadata(
                object_name=object_name
            )
            if object_state:
                self.egg.spatial.update_object_state(
                    object_node_id=object_node_id,
                    new_object_state=object_state,
                    timestamp=timestamp,
                )
        elif "room" not in object_name.lower():
            capabilities = self.get_object_capabilities_from_sim_metadata(
                object_name=object_name
            )
            assert capabilities
            state = self.get_object_state_from_sim_metadata(object_name=object_name)
            assert state
            new_node_id = self.egg.gen_id()
            object_metadata = self.get_sim_object_metadata(object_name=object_name)
            assert object_metadata
            object_node = ObjectNode(
                node_id=new_node_id,
                name=object_name,
                object_class=object_metadata.objectType,
                capabilities=capabilities,
                timestamped_states={timestamp: state},
            )
            logger.info(f"Adding new object node: {object_node}")
            self.egg.spatial.add_object_node(new_object_node=object_node)

    def update_visible_objects_states(self, timestamp: int | None):
        for object_name in self.get_sim_visible_objects():
            self.update_object_state(object_name=object_name, timestamp=timestamp)

    def move_to_object(self, object_name: str, teleport: bool) -> bool:
        interactable_pose = self.get_sim_interactable_poses(
            object_name=object_name,
            horizons=[30],
            is_standing=[True],
        )[0]

        nav_position: Position = interactable_pose[0]
        nav_angle: Rotation = interactable_pose[1]
        is_move_success = self.move_to_position(
            nav_position=nav_position,
            nav_angle=nav_angle,
            teleport=teleport,
        )
        agent_position, _, _, _ = self.get_sim_agent_state()
        dist_to_goal = agent_position.euclidean2d(nav_position)
        logger.info(f"Distance to nav goal: {dist_to_goal}")
        return is_move_success

    def perform_action_from_str(
        self,
        action_type: str,
        object_name: str,
        receptacle_name: str | None,
        teleport: bool = False,
    ) -> bool:
        logger.info(
            f"Performing action: {action_type}({object_name}, {receptacle_name})"
        )
        if action_type == ActionType.MOVE.lower():
            return self.move_to_object(object_name=object_name, teleport=teleport)
        elif action_type == ActionType.OPEN.lower():
            return self.execute_open(
                openable_object_name=object_name, teleport=teleport
            )
        elif action_type == ActionType.CLOSE.lower():
            return self.execute_close(
                openable_object_name=object_name, teleport=teleport
            )
        elif action_type == ActionType.TOGGLE_ON.lower():
            return self.execute_toggle_on(
                toggleable_object_name=object_name, teleport=teleport
            )
        elif action_type == ActionType.TOGGLE_OFF.lower():
            return self.execute_toggle_off(
                toggleable_object_name=object_name, teleport=teleport
            )
        elif action_type == ActionType.PICK.lower():
            return self.execute_pick(pick_object_name=object_name, teleport=teleport)
        elif action_type == ActionType.PLACE.lower():
            assert receptacle_name is not None
            return self.execute_place(
                held_object=object_name,
                receptacle_object_name=receptacle_name,
                teleport=teleport,
            )
        else:
            logger.error(f"Unknown action type: {action_type}")
            raise NotImplementedError
