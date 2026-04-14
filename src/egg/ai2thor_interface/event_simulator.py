import random
from typing import ClassVar, Literal
from pydantic import BaseModel, ConfigDict, Field
from ai2thor.controller import Controller
from ai2thor.server import MultiAgentEvent
import logging
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.geometry import JOIN_STYLE

_JOIN_STYLE = {
    "mitre": JOIN_STYLE.mitre,
    "miter": JOIN_STYLE.mitre,
    "round": JOIN_STYLE.round,
    "bevel": JOIN_STYLE.bevel,
}

from egg.graph.egg import EGG
from egg.graph.event import EventComponents
from egg.graph.node import ObjectNode
from egg.graph.spatial import SpatialComponents
from egg.utils.geometry import Position, Rotation
from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="ai2thor_interfaceh/event_simulator.log",
)


class EventSimulator(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )
    ai2thor_controller: Controller
    egg: EGG = Field(default=EGG(spatial=SpatialComponents(), events=EventComponents()))

    def get_picked_up_oject(self, object_name: str) -> ObjectNode:
        object_node = self.egg.spatial.get_object_node_by_name(object_name)
        assert isinstance(
            object_node, ObjectNode
        ), f"{object_node} does not exist in EGG"
        assert (
            object_node.capabilities.is_pickupable
        ), f"Trying pick and place but {object_node} is not pickupable."
        return object_node

    def get_receptacle_object(self, receptacle_name: str) -> ObjectNode:
        receptacle_node = self.egg.spatial.get_object_node_by_name(receptacle_name)
        assert isinstance(
            receptacle_node, ObjectNode
        ), f"{receptacle_node} does not exist in EGG"
        assert (
            receptacle_node.capabilities.is_receptacle
        ), f"{receptacle_name} is not a receptacle."
        return receptacle_node

    def get_agent_position(self) -> tuple[Position, Rotation]:
        event = self.ai2thor_controller.last_event
        assert isinstance(event, MultiAgentEvent)
        agent_metadata: dict[str, float] = event.metadata["agent"]
        return (
            Position.model_validate(agent_metadata["position"]),
            Rotation.model_validate(agent_metadata["rotation"]),
        )

    def get_reachable_positions(self) -> list[Position]:
        metadata = self.ai2thor_controller.step(action="GetReachablePositions").metadata
        reachable_positions: list[Position] = []
        for pos in metadata["actionReturn"]:
            reachable_positions.append(Position.model_validate(pos))
        return reachable_positions

    def get_free_spawn_coordinates_on_receptacle(
        self, receptacle_name: str, offset: float | None = None
    ) -> list[Position]:
        spawn_coordinates: list[dict[str, float]] = []
        free_spawn_coordinates: list[Position] = []

        receptacle_node = self.egg.spatial.get_object_node_by_name(
            node_name=receptacle_name
        )
        if isinstance(receptacle_node, ObjectNode):
            assert (
                receptacle_node.capabilities.is_receptacle
            ), f"Cannot get spawn coordinate from {receptacle_name} because it's not a receptacle"
            metadata = self.ai2thor_controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=receptacle_name,
                anywhere=True,
            ).metadata
            for pos in metadata["actionReturn"]:
                spawn_coordinates.append(pos)

            _, receptacle_latest_state = (
                receptacle_node.get_previous_timestamp_and_states()
            )
            if isinstance(receptacle_latest_state, ObjectNode.ObjectState):
                objects_on_receptacle = receptacle_latest_state.receptacle_object_ids
                free_spawn_coordinates = self.remove_occupied_spawn_coordinates(
                    objects_on_receptacle=objects_on_receptacle,
                    spawn_coordinates=spawn_coordinates,
                    offset=offset,
                )

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
                obj_node = self.egg.spatial.get_object_node_by_name(object)
                assert isinstance(obj_node, ObjectNode)
                if obj_node.object_class not in [ "Floor" , "Wall" ]:
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

    def spawn_object_at_receptacle(
        self, object_name: str, receptacle_name: str
    ) -> tuple[Position, list[Position]]:
        object_node = self.get_picked_up_oject(object_name=object_name)
        _, object_state = object_node.get_previous_timestamp_and_states()

        spawn_coord = Position(x=0, y=0, z=0)
        free_spawn_coordinates: list[Position] = []

        if object_state:
            object_aabb = object_state.bounding_box
            object_max_dim = object_aabb.size.get_max_dim()
            free_spawn_coordinates = self.get_free_spawn_coordinates_on_receptacle(
                receptacle_name=receptacle_name,
                offset=object_max_dim / 2,
            )

            free_spawn_coordinates = self.remove_points_near_inferred_boundary_convex(
                points=free_spawn_coordinates,
                d_offset=object_max_dim / 2,
            )

            spawn_coord = random.choice(free_spawn_coordinates)
            self.ai2thor_controller.step(
                action="PlaceObjectAtPoint",
                objectId=object_name,
                position=spawn_coord.model_dump(),
            )
        return spawn_coord, free_spawn_coordinates
