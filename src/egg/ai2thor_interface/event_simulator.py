# pyright: reportExplicitAny=none, reportAny=none
from datetime import datetime
from pydantic import JsonValue
from scipy.spatial.distance import cdist
import numpy as np
import random
from typing import Any, Literal
from ai2thor.controller import Controller
from ai2thor.server import Event
import logging
from numpy import empty
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.geometry import JOIN_STYLE

from egg.utils.data import Ai2ThorObjectMetadata
from egg.utils.timestamp import datetime_to_ns

_JOIN_STYLE = {
    "mitre": JOIN_STYLE.mitre,
    "miter": JOIN_STYLE.mitre,
    "round": JOIN_STYLE.round,
    "bevel": JOIN_STYLE.bevel,
}

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

    def get_grid_size(self) -> float:
        return self.ai2thor_controller.initialization_parameters["gridSize"]

    def get_picked_up_oject(self, object_name: str) -> ObjectNode:
        _, object_node = self.egg.spatial.get_object_node_by_name(object_name)
        assert isinstance(
            object_node, ObjectNode
        ), f"{object_node} does not exist in EGG"
        assert (
            object_node.capabilities.is_pickupable
        ), f"Trying pick and place but {object_node} is not pickupable."
        return object_node

    def get_receptacle_object(self, receptacle_name: str) -> ObjectNode:
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

    def get_agent_state(self) -> tuple[Position, Rotation, float, bool]:
        event = self.ai2thor_controller.last_event
        assert isinstance(event, Event)
        agent_metadata: dict[str, float | bool] = event.metadata["agent"]
        return (
            Position.model_validate(agent_metadata["position"]),
            Rotation.model_validate(agent_metadata["rotation"]),
            agent_metadata["cameraHorizon"],
            agent_metadata["isStanding"],
        )

    def get_object_state(self, object_name: str) -> ObjectNode.ObjectState | None:
        event = self.ai2thor_controller.last_event
        assert isinstance(event, Event)
        all_objects_metadata: list[dict[str, Any]] = event.metadata["objects"]
        for object_metadata in all_objects_metadata:
            if object_metadata["objectId"] == object_name:
                object_state = ObjectNode.ObjectState.from_ai2thor(
                    object_metadata=Ai2ThorObjectMetadata.model_validate(
                        object_metadata
                    )
                )
                return object_state
        logger.warning(f"Could not retrieve metadata of {object_name}")
        return None

    def get_object_being_held_by_agent(self) -> list[dict[str, str]]:
        event = self.ai2thor_controller.last_event
        assert event, "Event is None"
        return event.metadata["inventoryObjects"]

    def get_reachable_positions(self) -> list[Position]:
        metadata = self.ai2thor_controller.step(action="GetReachablePositions").metadata
        reachable_positions: list[Position] = []
        for pos in metadata["actionReturn"]:
            reachable_positions.append(Position.model_validate(pos))
        return reachable_positions

    def get_interactable_poses(
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
        agent_position, _, _, _ = self.get_agent_state()
        interactable_distances = [
            p[0].euclidean(agent_position) for p in interactable_poses
        ]
        sorted_interactable_poses: list[tuple[Position, Rotation, bool, float]] = []
        for j in np.argsort(interactable_distances):
            sorted_interactable_poses.append(interactable_poses[j])
        return sorted_interactable_poses

    def get_closest_interactable_poses(
        self, object_name: str, is_standing: list[bool]
    ) -> tuple[Position, Rotation]:

        agent_position, _, agent_horizon, _ = self.get_agent_state()
        interactable_poses = self.get_interactable_poses(
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

    def get_free_spawn_coordinates_on_receptacle(
        self,
        receptacle_name: str,
        offset: float | None = None,
        remove_occupied_spawn_coordinates: bool = True,
    ) -> list[Position]:
        spawn_coordinates: list[dict[str, float]] = []
        free_spawn_coordinates: list[Position] = []

        _, receptacle_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=receptacle_name
        )
        if isinstance(receptacle_object_node, ObjectNode):
            assert (
                receptacle_object_node.capabilities.is_receptacle
            ), f"Cannot get spawn coordinate from {receptacle_name} because it's not a receptacle"
            metadata = self.ai2thor_controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=receptacle_name,
                anywhere=True,
            ).metadata
            for pos in metadata["actionReturn"]:
                spawn_coordinates.append(pos)

            if remove_occupied_spawn_coordinates:
                _, receptacle_latest_state = (
                    receptacle_object_node.get_previous_timestamp_and_states()
                )
                if isinstance(receptacle_latest_state, ObjectNode.ObjectState):
                    objects_on_receptacle = (
                        receptacle_latest_state.receptacle_object_ids
                    )
                    free_spawn_coordinates = self.remove_occupied_spawn_coordinates(
                        objects_on_receptacle=objects_on_receptacle,
                        spawn_coordinates=spawn_coordinates,
                        offset=offset,
                    )
            else:
                free_spawn_coordinates = [
                    Position.model_validate(pos) for pos in spawn_coordinates
                ]
        return free_spawn_coordinates

    def remove_occupied_spawn_coordinates(
        self,
        objects_on_receptacle: list[str] | None,
        spawn_coordinates: list[dict[str, float]],
        offset: float | None = None,
    ) -> list[Position]:
        free_spawn_coordinates: list[Position] = []
        occupied_spawn_coordinates: list[dict[str, float]] = []
        if objects_on_receptacle:
            for object in objects_on_receptacle:
                _, obj_node = self.egg.spatial.get_object_node_by_name(object)
                assert isinstance(obj_node, ObjectNode)
                if obj_node.object_class not in ["Floor", "Wall"]:
                    _, obj_state = obj_node.get_previous_timestamp_and_states()
                    assert isinstance(obj_state, ObjectNode.ObjectState)
                    obj_aabb = obj_state.bounding_box
                    for pos in spawn_coordinates:
                        pos_receptacle = Position.model_validate(pos)
                        if obj_aabb.contains_2d(
                            pos=pos_receptacle,
                            omitted_axis="y",
                            offset=(
                                obj_aabb.size.get_max_dim() / 2
                                if offset is None
                                else offset
                            ),
                        ):
                            occupied_spawn_coordinates.append(pos)
            for pos in spawn_coordinates:
                if pos not in occupied_spawn_coordinates:
                    free_spawn_coordinates.append(Position.model_validate(pos))
        return free_spawn_coordinates

    def remove_points_near_inferred_boundary_convex(
        self,
        points: list[Position],
        d_offset: float | None = None,
        join_style: Literal["mitre", "round", "bevel"] = "mitre",
        mitre_limit: float = 5.0,
        resolution: int = 16,
    ) -> list[Position]:

        kept: list[Position] = []

        if d_offset is None:
            return kept
        elif d_offset < 0:
            raise ValueError("d_offset must be non-negative or None")

        # Project to 2D
        pts2d = [(p.x, p.z) for p in points]
        # Infer convex hull
        hull = MultiPoint(pts2d).convex_hull

        # Degenerate hulls (no interior) -> nothing can be ≥ d_offset from boundary
        if not isinstance(hull, Polygon):
            logger.warning("Empty hull")
            return []

        # Create inward buffer
        if d_offset > 0:
            js_key = join_style.lower()
            if js_key in _JOIN_STYLE:
                shrunken = hull.buffer(
                    -d_offset,
                    join_style=_JOIN_STYLE[js_key],
                    mitre_limit=mitre_limit,
                    resolution=resolution,
                )
            else:
                raise ValueError(f"{join_style} is not a valid joint style")

            for pt3d, (xz_x, xz_z) in zip(points, pts2d):
                p = Point(xz_x, xz_z)
                inside = shrunken.covers(p)
                if inside:
                    kept.append(pt3d)

        return kept

    # def spawn_object_at_receptacle(
    #     self,
    #     object_name: str,
    #     receptacle_name: str,
    #     remove_occupied_spawn_coordinates: bool = True,
    #     remove_points_near_inferred_boundary_convex: bool = True,
    # ) -> tuple[Position, list[Position]]:
    #     object_node = self.get_picked_up_oject(object_name=object_name)
    #     _, object_state = object_node.get_previous_timestamp_and_states()
    #
    #     spawn_coord = Position(x=0, y=0, z=0)
    #     free_spawn_coordinates: list[Position] = []
    #
    #     if object_state:
    #         action_return = None
    #         while action_return is None:
    #             object_aabb = object_state.bounding_box
    #             object_max_dim = object_aabb.size.get_max_dim()
    #
    #             free_spawn_coordinates = self.get_free_spawn_coordinates_on_receptacle(
    #                 receptacle_name=receptacle_name,
    #                 offset=object_max_dim / 2,
    #                 remove_occupied_spawn_coordinates=remove_occupied_spawn_coordinates,
    #             )
    #
    #             if remove_points_near_inferred_boundary_convex:
    #                 free_spawn_coordinates = (
    #                     self.remove_points_near_inferred_boundary_convex(
    #                         points=free_spawn_coordinates,
    #                         d_offset=object_max_dim / 2,
    #                     )
    #                 )
    #
    #             if len(free_spawn_coordinates) == 0:
    #                 break
    #             spawn_coord = random.choice(free_spawn_coordinates)
    #             event = self.ai2thor_controller.step(
    #                 action="PlaceObjectAtPoint",
    #                 objectId=object_name,
    #                 position=spawn_coord.model_dump(),
    #             )
    #             action_return = event.metadata["actionReturn"]
    #             if action_return is None:
    #                 free_spawn_coordinates.remove(spawn_coord)
    #                 logger.warning(
    #                     f"Retrying placing {object_name} on {receptacle_name}, removing {spawn_coord}"
    #                 )
    #                 spawn_coord = Position(x=0, y=0, z=0)
    #                 if len(free_spawn_coordinates) == 0:
    #                     logger.warning(
    #                         f"Could not place {object_name} on {receptacle_name}"
    #                     )
    #             else:
    #                 logger.info(
    #                     f"Succesfully placed {object_name} on {receptacle_name} at {spawn_coord}."
    #                 )
    #     return spawn_coord, free_spawn_coordinates

    def move_to_position(
        self,
        nav_position: Position,
        nav_angle: Rotation,
        teleport: bool = False,
        standing: bool = True,
    ) -> Event | None:
        if teleport:
            event = self.ai2thor_controller.step(
                action="Teleport",
                position=nav_position.model_dump(),
                rotation=nav_angle.model_dump(),
                standing=standing,
            )
            return event
        else:
            agent_position, agent_rotation, _, _ = self.get_agent_state()
            _, path_planning_cmds = ai2thor_nav.plan_path_and_command(
                start_pos=agent_position,
                start_yaw_deg=agent_rotation.y,
                reachable_positions=self.get_reachable_positions(),
                goal_pos=nav_position,
                grid_size=self.get_grid_size(),
            )
            event = None
            for cmd in path_planning_cmds:
                event = self.ai2thor_controller.step(action=cmd)
            return event

    def pick(self, pick_object_name: str, teleport: bool = False):
        picked_up_object_node = self.get_picked_up_oject(object_name=pick_object_name)
        _, picked_up_object_prev_state = (
            picked_up_object_node.get_previous_timestamp_and_states()
        )
        assert isinstance(
            picked_up_object_prev_state, ObjectNode.ObjectState
        ), f"Could not get previous state of {pick_object_name}"

        _, _, agent_horizon, _ = self.get_agent_state()

        parent_receptacles = picked_up_object_prev_state.parent_receptacles
        opened_parents: list[str] = []
        if parent_receptacles:
            for parent in parent_receptacles:
                parent_node = self.get_receptacle_object(receptacle_name=parent)
                if parent_node.capabilities.is_openable:
                    self.try_open(openable_object_name=parent, teleport=teleport)
                    opened_parents.append(parent)
        opened_parents.reverse()

        success_pick: bool = False

        # TODO: Analyze interactive obj height and select standing / crouching
        for standing in [True, False]:
            if success_pick:
                break
            interactable_poses = self.get_interactable_poses(
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
                event = self.ai2thor_controller.step(
                    action="PickupObject",
                    objectId=pick_object_name,
                    forceAction=False,
                    manualInteract=False,
                )
                if event.metadata["lastActionSuccess"]:
                    logger.info(
                        f"Successfully picked up {pick_object_name} at {pick_nav_position}"
                    )
                    self.update_agent_state(timestamp=None, holding=pick_object_name)
                    self.update_visible_objects_states(timestamp=None)
                    success_pick = True
                    break
                else:
                    logger.warning(
                        f"Unable to pick up {pick_object_name} at {pick_nav_position}"
                    )
        for parent in opened_parents:
            self.try_close(openable_object_name=parent, teleport=teleport)

    def try_close(self, openable_object_name: str, teleport: bool = False):
        _, openable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=openable_object_name
        )
        assert isinstance(
            openable_object_node, ObjectNode
        ), f"{openable_object_node} does not exists"
        if openable_object_node.capabilities.is_openable:
            _, openable_object_prev_state = (
                openable_object_node.get_previous_timestamp_and_states()
            )
            assert isinstance(
                openable_object_prev_state, ObjectNode.ObjectState
            ), f"Could not get previous state of {openable_object_name}"
            _, _, agent_horizon, _ = self.get_agent_state()
            success_close: bool = False
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_close:
                    break
                interactable_poses = self.get_interactable_poses(
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
                    event = self.ai2thor_controller.step(
                        action="CloseObject",
                        objectId=openable_object_name,
                        forceAction=False,
                    )
                    if event.metadata["lastActionSuccess"]:
                        logger.info(f"Successfully closed {openable_object_name}")
                        self.update_agent_state(timestamp=None)
                        self.update_visible_objects_states(timestamp=None)
                        success_close = True
                        break
                    else:
                        logger.warning(
                            f"Could not close {openable_object_name} from {close_nav_position}"
                        )

    def try_open(self, openable_object_name: str, teleport: bool = False):
        _, openable_object_node = self.egg.spatial.get_object_node_by_name(
            node_name=openable_object_name
        )
        assert isinstance(
            openable_object_node, ObjectNode
        ), f"{openable_object_node} does not exists"
        if openable_object_node.capabilities.is_openable:
            _, openable_object_prev_state = (
                openable_object_node.get_previous_timestamp_and_states()
            )
            assert isinstance(
                openable_object_prev_state, ObjectNode.ObjectState
            ), f"Could not get previous state of {openable_object_name}"
            _, _, agent_horizon, _ = self.get_agent_state()
            success_open: bool = False
            # TODO: Analyze interactive obj height and select standing / crouching
            for standing in [True, False]:
                if success_open:
                    break
                interactable_poses = self.get_interactable_poses(
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
                    event = self.ai2thor_controller.step(
                        action="OpenObject",
                        objectId=openable_object_name,
                        openness=1,
                        forceAction=False,
                    )
                    openable_object_state = self.get_object_state(
                        object_name=openable_object_name
                    )
                    assert isinstance(openable_object_state, ObjectNode.ObjectState)
                    if (
                        openable_object_state.openness == 1.0
                        and openable_object_state.is_open
                    ):
                        logger.info(f"Successfully opened {openable_object_name}")
                        self.update_agent_state(timestamp=None)
                        self.update_visible_objects_states(timestamp=None)
                        success_open = True
                        break
                    else:
                        logger.warning(
                            f"Could not open {openable_object_name} from {open_nav_position}"
                        )

    def place(
        self,
        receptacle_object_name: str,
        teleport: bool = False,
    ):
        # TODO: Add sanity check that robot is holding sth
        receptacle_object_node = self.get_receptacle_object(
            receptacle_name=receptacle_object_name
        )
        _, receptacle_object_prev_state = (
            receptacle_object_node.get_previous_timestamp_and_states()
        )
        assert isinstance(
            receptacle_object_prev_state, ObjectNode.ObjectState
        ), f"Could not get previous state of {receptacle_object_name}"
        held_object_info = self.get_object_being_held_by_agent()
        assert held_object_info, f"Robot not holding any object but trying to place"
        held_object_name = held_object_info[0]["objectId"]

        self.try_open(openable_object_name=receptacle_object_name, teleport=teleport)
        _, _, agent_camera_horizon, _ = self.get_agent_state()
        success_place: bool = False

        # TODO: Analyze interactive obj height and select standing / crouching
        for standing in [True, False]:
            if success_place:
                break
            interactable_poses = self.get_interactable_poses(
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
                event = self.ai2thor_controller.step(
                    action="PutObject",
                    objectId=receptacle_object_name,
                    forceAction=False,
                    placeStationary=True,
                )
                if event.metadata["lastActionSuccess"]:
                    logger.info(
                        f"Succesfully placed {held_object_name} on {receptacle_object_name} at {place_nav_position}"
                    )
                    self.update_agent_state(timestamp=None)
                    self.update_visible_objects_states(timestamp=None)
                    success_place = True
                    break
                else:
                    logger.warning(
                        f"Unable to place {held_object_name} at {receptacle_object_name} at {place_nav_position}, retrying"
                    )
        self.try_close(openable_object_name=receptacle_object_name, teleport=teleport)

    #
    # TODO: Add kwargs to params
    # def try_action(self, action_name: str, object_name: str, teleport: bool = False):
    #     _, _, agent_camera_horizon, _ = self.get_agent_state()
    #     interactable_poses = self.get_interactable_poses(
    #         object_name=object_name, horizons=[agent_camera_horizon]
    #     )
    #     # TODO: Try this by analyzing the object to interact with (BBox/position)
    #     for standing in [True, False]:
    #         for pose in interactable_poses:
    #             nav_position: Position = pose[0]
    #             nav_angle: Rotation = pose[1]
    #             _ = self.move_to_position(
    #                 nav_position=nav_position,
    #                 nav_angle=nav_angle,
    #                 teleport=teleport,
    #                 standing=standing,
    #             )
    #             event = self.ai2thor_controller.step(
    #                 action=action_name,
    #                 objectId=object_name,
    #                 forceAction=False,
    #                 placeStationary=True,
    #             )
    #             if event.metadata["lastActionSuccess"]:
    #                 logger.info(
    #                     f"Succesfully perform {action_name} on {object_name} at {nav_position}, {'standing' if standing else 'crouching'}"
    #                 )
    #                 self.update_agent_state(timestamp=None)
    #                 self.update_visible_objects_states(timestamp=None)
    #                 break
    #             else:
    #                 logger.warning(
    #                     f"Unable to perform {action_name} on {object_name} at {nav_position}, {'standing' if standing else 'crouching'}. Retrying."
    #                 )

    def update_agent_state(
        self, timestamp: int | None, holding: str | None = None, agent_id: int = 0
    ):
        (
            agent_position,
            agent_rotation,
            agent_camera_horizon,
            agent_is_standing,
        ) = self.get_agent_state()
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
        object_node_id, _ = self.egg.spatial.get_object_node_by_name(
            node_name=object_name
        )
        if object_node_id:
            object_state = self.get_object_state(object_name=object_name)
            if object_state:
                self.egg.spatial.update_object_state(
                    object_node_id=object_node_id,
                    new_object_state=object_state,
                    timestamp=timestamp,
                )

    def update_visible_objects_states(self, timestamp: int | None):
        for object_name in self.get_visible_objects():
            self.update_object_state(object_name=object_name, timestamp=timestamp)

    def get_visible_objects(self) -> list[str]:
        last_event = self.ai2thor_controller.last_event
        visible_objects = list(last_event.instance_detections2D.keys())
        return visible_objects

    def pick_and_place(
        self,
        pick_object_name: str,
        place_receptacle_name: str,
        teleport: bool = False,
        close_receptacle_after_placing: bool = True,
    ):
        # TODO: Update EGG State after every action
        self.pick(
            pick_object_name=pick_object_name,
            teleport=teleport,
        )
        self.place(
            receptacle_object_name=place_receptacle_name,
            teleport=teleport,
        )
        _ = self.ai2thor_controller.step(action="Done")
