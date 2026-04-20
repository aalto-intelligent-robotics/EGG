from datetime import datetime
from dis import Positions
import sys
import numpy as np
from numpy.typing import NDArray
import logging
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Annotated, ClassVar, Literal
from typing_extensions import Self

from egg.utils.data import (
    Ai2ThorAgentMetadata,
    Ai2ThorObjectMetadata,
    Ai2ThorRoomMetadata,
    Ai2ThorTemperature,
)
from egg.utils.geometry import Polygon, Position, Odometry, Rotation
from egg.utils.logger import getLogger
from egg.utils.geometry import AxisAlignedBoundingBox
from egg.utils.timestamp import (
    datetime_to_ns,
    print_timestamped_position,
    ns_to_datetime,
    print_timestamped_observation_odom,
)

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/node.log",
)


class GraphNode(BaseModel):
    """
    Represents a basic node in a graph with a unique identifier.

    :param node_id: Unique identifier for the node.
    :type node_id: int
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    node_id: Annotated[int, Field(ge=0)]


class SpatialNode(GraphNode):
    """
    Represents a spatial node, extending the basic graph node by adding a name.

    :param name: Name of the spatial node.
    :type name: str
    """

    name: str


class EventNode(GraphNode):
    """
    Represents an event node in the graph, capturing event-related metadata.

    :param event_description: Description of the event.
    :type event_description: str
    :param start: Start timestamp of the event.
    :type start: int
    :param end: End timestamp of the event.
    :type end: int
    :param timestamped_observation_odom: Odometry data associated with timestamps, in the format
        {timestamp: {"base_odom": [[x,y,z],[x,y,z,w]], "camera_odom": [[x,y,z],[x,y,z,w]]}}.
    :type timestamped_observation_odom: dict[int, dict[str, list]]
    :param involved_object_ids: list of object IDs involved in the event.
    :type involved_object_ids: list[int]
    :param location: Location where the event takes place.
    :type location: str
    """

    event_description: str = "Default description"
    start: int = Field(default=0, gt=0)
    end: int = Field(default=1, gt=0)
    timestamped_observation_odom: dict[int, Odometry] = Field(
        default_factory=dict,
        description=("""
            Format: {
                timestamp:
                {
                    'position': {'x': <x>, 'y': <y>, 'z': <z>}, 
                    'rotation': {'roll': <roll>, 'pitch': <pitch>, 'yaw': <yaw>}
                }
            }
            """),
    )
    location: str

    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than or equal to start")
        return self

    def is_in_time_range(
        self, min_timestamp: int = 0, max_timestamp: int = sys.maxsize
    ):
        """
        Checks if the event is within a specified time range.

        :param min_timestamp: Minimum timestamp for the range.
        :type min_timestamp: int
        :param max_timestamp: Maximum timestamp for the range.
        :type max_timestamp: int
        :returns: True if the event is within the range, False otherwise.
        :rtype: bool
        """
        if self.start >= min_timestamp and self.end <= max_timestamp:
            return True
        return False

    def is_in_location(self, location_list: list[str] | None = None):
        """
        Checks if the event location is in a given list of locations.

        :param location_list: List of locations to check against.
        :type location_list: Optional[List[str]]
        :returns: True if the event location is in the list, False otherwise.
        :rtype: bool
        """
        return location_list is None or self.location in location_list

    def pretty_str(self) -> str:
        """
        Generates a formatted string representation of the event node details.

        :returns: String representing the event node.
        :rtype: str
        """
        event_node_str = (
            "\n🕛 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Start: {str(ns_to_datetime(self.start))}\n"
            + f"End: {str(ns_to_datetime(self.end))}\n"
            + f"Node type: Event\n"
            + f"Description: {self.event_description}\n"
            + f"Location: {self.location}\n"
            + f"Timestamped Observation Positions: {print_timestamped_observation_odom(self.timestamped_observation_odom)}\n"
        )
        return event_node_str


class AgentNode(SpatialNode):
    class AgentState(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        )
        position: Position
        rotation: Rotation
        camera_horizon: float
        is_standing: bool
        holding: str | None = None

        @classmethod
        def from_ai2thor(cls, agent_metadata: Ai2ThorAgentMetadata) -> Self:
            return cls(
                position=agent_metadata.position,
                rotation=agent_metadata.rotation,
                camera_horizon=agent_metadata.cameraHorizon,
                is_standing=agent_metadata.isStanding,
            )

    timestamped_states: dict[int, AgentState] = Field(default_factory=dict)

    def state_str(self) -> str:
        state_str = "\n"
        for ts, ts_state in self.timestamped_states.items():
            state_str += (
                f"\n\t- {ns_to_datetime(ts)}:"
                + f"\n\t\t- Position: {ts_state.position}"
                + f"\n\t\t- Rotation: {ts_state.rotation}"
                + f"\n\t\t- Camera Horizon: {ts_state.camera_horizon}"
                + f"\n\t\t- Standing: {ts_state.is_standing}"
                + f"\n\t\t- Holding: {ts_state.holding}"
            )
        return state_str

    def pretty_str(self) -> str:
        """
        Generates a formatted, human-readable string for a RoomNode.

        :returns: String representing the room node.
        :rtype: str
        """

        room_str = (
            "\n🤖 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Name: {self.name}\n"
            + f"Node type: Agent\n"
            + f"Timestamped states: {self.state_str()}\n"
        )
        return room_str


class RoomNode(SpatialNode):
    """
    Represents a room node, extending SpatialNode by adding position data.

    :param position: Position of the room as a numpy array.
    :type position: np.ndarray
    """

    name: str = Field(frozen=True)
    floor_polygon: Polygon

    @classmethod
    def from_ai2thor(cls, node_id: int, room_metadata: Ai2ThorRoomMetadata) -> Self:
        return cls(
            node_id=node_id,
            name=room_metadata.roomType,
            floor_polygon=Polygon(corners=room_metadata.get_polygon_corners()),
        )

    def pretty_str(self) -> str:
        """
        Generates a formatted, human-readable string for a RoomNode.

        :returns: String representing the room node.
        :rtype: str
        """

        room_str = (
            "\n🏠 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Name: {self.name}\n"
            + f"Node type: Room\n"
            + f"Floor polygon: {self.floor_polygon}\n"
        )
        return room_str


class ObjectNode(SpatialNode):
    """
    Represents an object node in the graph, adding specific metadata relevant
    to object tracking and visualization.

    :param object_class: Classification of the object.
    :type object_class: str
    :param timestamped_position: Position data indexed by timestamps.
    :type timestamped_position: dict[int, np.ndarray]
    :param visual_embedding: [Optional] A visual embedding tensor.
    :param instance_views: List of visual observations of the object.
    :type instance_views: list[np.ndarray]
    :param caption: [Optional] A textual description of the object.
    :type caption: Optional[str]
    """

    class ObjectCapabilities(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        )

        is_pickupable: bool = Field(default=False, frozen=True)
        is_moveable: bool = Field(default=False, frozen=True)
        is_toggleable: bool = Field(default=False, frozen=True)
        is_breakable: bool = Field(default=False, frozen=True)
        can_fill_with_liquid: bool = Field(default=False, frozen=True)
        is_dirtyable: bool = Field(default=False, frozen=True)
        can_be_used_up: bool = Field(default=False, frozen=True)
        is_cookable: bool = Field(default=False, frozen=True)
        is_heat_source: bool = Field(default=False, frozen=True)
        is_cold_source: bool = Field(default=False, frozen=True)
        is_sliceable: bool = Field(default=False, frozen=True)
        is_openable: bool = Field(default=False, frozen=True)
        is_receptacle: bool = Field(default=False, frozen=True)

        @classmethod
        @classmethod
        def from_ai2thor(cls, object_metadata: Ai2ThorObjectMetadata) -> Self:
            return cls(
                is_pickupable=object_metadata.pickupable,
                is_moveable=object_metadata.moveable,
                is_toggleable=object_metadata.toggleable,
                is_breakable=object_metadata.breakable,
                can_fill_with_liquid=object_metadata.canFillWithLiquid,
                is_dirtyable=object_metadata.dirtyable,
                can_be_used_up=object_metadata.canBeUsedUp,
                is_cookable=object_metadata.cookable,
                is_heat_source=object_metadata.isHeatSource,
                is_cold_source=object_metadata.isColdSource,
                is_sliceable=object_metadata.sliceable,
                is_openable=object_metadata.openable,
                is_receptacle=object_metadata.receptacle,
            )

    class ObjectState(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        )
        is_visible: bool = False
        position: Position
        bounding_box: AxisAlignedBoundingBox
        is_picked_up: bool = False
        is_moving: bool = False
        is_toggled: bool = False
        is_broken: bool = False
        is_filled_with_liquid: bool = False
        is_dirty: bool = False
        is_used_up: bool = False
        is_cooked: bool = False
        is_sliced: bool = False
        is_open: bool = False
        temperature: Ai2ThorTemperature = "RoomTemp"
        parent_receptacles: list[str] | None = None
        receptacle_object_ids: list[str] | None = None
        instance_view: NDArray[np.uint8] | None = None
        openness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

        @classmethod
        def from_ai2thor(cls, object_metadata: Ai2ThorObjectMetadata) -> Self:
            bounding_box = object_metadata.get_bounding_box()
            bounding_box.round(ndigits=3)
            return cls(
                position=bounding_box.center,
                bounding_box=bounding_box,
                is_visible=object_metadata.visible,
                is_picked_up=object_metadata.isPickedUp,
                is_moving=object_metadata.isMoving,
                is_toggled=object_metadata.isToggled,
                is_broken=object_metadata.isBroken,
                is_filled_with_liquid=object_metadata.isFilledWithLiquid,
                is_dirty=object_metadata.isDirty,
                is_cooked=object_metadata.isCooked,
                is_open=object_metadata.isOpen,
                openness=object_metadata.openness,
                parent_receptacles=object_metadata.parentReceptacles,
                receptacle_object_ids=object_metadata.receptacleObjectIds,
                temperature=object_metadata.temperature,
            )

    object_class: str = Field(frozen=True)

    caption: str | None = None

    capabilities: ObjectCapabilities
    timestamped_states: dict[int, ObjectState] = Field(default_factory=dict)

    @classmethod
    def from_ai2thor(cls, node_id: int, object_metadata: Ai2ThorObjectMetadata) -> Self:
        # TODO: Use config file
        object_capabilities = cls.ObjectCapabilities.from_ai2thor(
            object_metadata=object_metadata
        )
        object_state = cls.ObjectState.from_ai2thor(object_metadata=object_metadata)
        return cls(
            node_id=node_id,
            name=object_metadata.objectId,
            object_class=object_metadata.objectType,
            capabilities=object_capabilities,
            timestamped_states={datetime_to_ns(datetime.now()): object_state},
        )

    def crop_timestamped_states(self, min_timestamp: int, max_timestamp: int):
        """
        Removes positions that fall outside a specified timestamp range.

        :param min_timestamp: Minimum timestamp for retention.
        :type min_timestamp: int
        :param max_timestamp: Maximum timestamp for retention.
        :type max_timestamp: int
        """
        for timestamp in self.timestamped_states.keys():
            if timestamp <= min_timestamp and timestamp >= max_timestamp:
                _ = self.timestamped_states.pop(timestamp)

    def capabilities_str(self) -> str:
        capabilities = {
            "pickupable": self.capabilities.is_pickupable,
            "moveable": self.capabilities.is_moveable,
            "toggleable": self.capabilities.is_toggleable,
            "breakable": self.capabilities.is_breakable,
            "can_fill_with_liquid": self.capabilities.can_fill_with_liquid,
            "dirtyable": self.capabilities.is_dirtyable,
            "can_be_used_up": self.capabilities.can_be_used_up,
            "cookable": self.capabilities.is_cookable,
            "is_heat_source": self.capabilities.is_heat_source,
            "is_cold_source": self.capabilities.is_cold_source,
            "sliceable": self.capabilities.is_sliceable,
            "openable": self.capabilities.is_openable,
            "is_receptacle": self.capabilities.is_receptacle,
        }
        caps_on = [k for k, v in capabilities.items() if v is True]
        return ", ".join(caps_on) if caps_on else "None"

    def state_str(self) -> str:
        state_str = ""
        for ts, ts_state in self.timestamped_states.items():
            state = {
                "is_picked_up": ts_state.is_picked_up,
                "is_moving": ts_state.is_moving,
                "is_toggled": ts_state.is_toggled,
                "is_broken": ts_state.is_broken,
                "is_filled_with_liquid": ts_state.is_filled_with_liquid,
                "is_dirty": ts_state.is_dirty,
                "is_used_up": ts_state.is_used_up,
                "is_cooked": ts_state.is_cooked,
                "is_sliced": ts_state.is_sliced,
                "is_open": ts_state.is_open,
            }
            state_on = [k for k, v in state.items() if v is True]
            state_str += (
                f"\n\t- {ns_to_datetime(ts)}:"
                + f"\n\t\t- Position: {ts_state.position}"
                + f"\n\t\t- {', '.join(state_on) if state_on else None}"
                + f"\n\t\t- Temperature: {ts_state.temperature}, Visible: {ts_state.is_visible}"
                + f"\n\t\t- Openness: {ts_state.openness}"
                + f"\n\t\t- Parent Receptacles: {', '.join(ts_state.parent_receptacles) if ts_state.parent_receptacles else None}"
                + f"\n\t\t- Contained Object IDs: {', '.join(map(str, ts_state.receptacle_object_ids)) if ts_state.receptacle_object_ids else None}\n"
            )
        return state_str

    def has_been_seen(self, timestamp: int) -> bool:
        """
        Checks if the object has been observed before a given timestamp.

        :param timestamp: The reference timestamp.
        :type timestamp: int
        :returns: True if the object has been seen, False otherwise.
        :rtype: bool
        """
        first_timestamp = next(iter(self.timestamped_states.keys()))
        return first_timestamp <= timestamp

    def get_previous_timestamp_and_states(self, ref_timestamp: int = sys.maxsize):
        """
        Gets the previous timestamp and position relative to a reference timestamp.
        If a referecne timestamp is not provided, default to the last time the node was registered

        :param ref_timestamp: The reference timestamp.
        :type ref_timestamp: int
        :returns: Tuple of the previous timestamp and position, or (None, None).
        :rtype: Tuple[Optional[int], Optional[np.ndarray]]
        """
        if self.has_been_seen(ref_timestamp):
            prev_timestamp = next(iter(self.timestamped_states.keys()))
            for timestamp in self.timestamped_states.keys():
                if timestamp >= ref_timestamp:
                    return (prev_timestamp, self.timestamped_states[prev_timestamp])
                else:
                    prev_timestamp = timestamp
            return (prev_timestamp, self.timestamped_states[prev_timestamp])
        else:
            return (None, None)

    def pretty_str(self) -> str:
        """
        Generates a formatted, human-readable string for an ObjectNode.

        :returns: String representing the object node.
        :rtype: str
        """
        # Caption
        caption = getattr(self, "caption", None) or "—"

        obj_str = (
            "\n📦 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Name: {self.name}\n"
            + f"Node type: Object\n"
            + f"Class: {self.object_class}\n"
            + f"Caption: {caption}\n"
            + f"Capabilities (True): {self.capabilities_str()}\n"
            + f"State (True): {self.state_str()}\n"
        )
        return obj_str
