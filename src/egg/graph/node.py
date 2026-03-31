from datetime import datetime
import numpy as np
from numpy.typing import NDArray
import logging
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Annotated, ClassVar
from typing_extensions import Self

from egg.utils.data import Ai2ThorObjectMetadata, Ai2ThorRoomMetadata
from egg.utils.geometry import Polygon, Position, Odometry
from egg.utils.logger import getLogger
from egg.utils.bounding_box import AxisAlignedBoundingBox
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

    event_description: str
    start: int
    end: int
    timestamped_observation_odom: dict[int, Odometry] = Field(
        default_factory=dict,
        description=(
            """
            Format: {
                timestamp:
                {
                    'position': {'x': <x>, 'y': <y>, 'z': <z>}, 
                    'rotation': {'roll': <roll>, 'pitch': <pitch>, 'yaw': <yaw>}
                }
            }
            """
        ),
    )
    location: str

    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than or equal to start")
        return self

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

    object_class: str = Field(frozen=True)
    timestamped_position: dict[int, Position] = Field(default_factory=dict)
    bounding_box: AxisAlignedBoundingBox

    instance_views: list[NDArray[np.uint8]] = Field(default_factory=list)
    caption: str | None = None

    is_visible: bool = False
    parent_receptacles: list[str] | None = None
    receptacle_object_ids: list[str] | None = None

    # Immutable capability flags (required, no defaults)
    is_pickupable: bool = Field(frozen=True)
    is_moveable: bool = Field(frozen=True)
    is_toggleable: bool = Field(frozen=True)
    is_breakable: bool = Field(frozen=True)
    can_fill_with_liquid: bool = Field(frozen=True)
    is_dirtyable: bool = Field(frozen=True)
    can_be_used_up: bool = Field(frozen=True)
    is_cookable: bool = Field(frozen=True)
    is_heat_source: bool = Field(frozen=True)
    is_cold_source: bool = Field(frozen=True)
    is_sliceable: bool = Field(frozen=True)
    is_openable: bool = Field(frozen=True)
    is_receptacle: bool = Field(frozen=True)

    # Mutable state fields
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
    temperature: str = "RoomTemp"

    openness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @classmethod
    def from_ai2thor(cls, node_id: int, object_metadata: Ai2ThorObjectMetadata) -> Self:
        bounding_box = object_metadata.get_bounding_box()
        return cls(
            node_id=node_id,
            name=object_metadata.objectId,
            object_class=object_metadata.objectType,
            bounding_box=bounding_box,
            is_visible=object_metadata.visible,
            is_pickupable=object_metadata.pickupable,
            is_picked_up=object_metadata.isPickedUp,
            is_moveable=object_metadata.moveable,
            is_moving=object_metadata.isMoving,
            is_toggleable=object_metadata.toggleable,
            is_toggled=object_metadata.isToggled,
            is_breakable=object_metadata.breakable,
            is_broken=object_metadata.isBroken,
            can_fill_with_liquid=object_metadata.canFillWithLiquid,
            is_filled_with_liquid=object_metadata.isFilledWithLiquid,
            is_dirtyable=object_metadata.dirtyable,
            is_dirty=object_metadata.isDirty,
            can_be_used_up=object_metadata.canBeUsedUp,
            is_cookable=object_metadata.cookable,
            is_cooked=object_metadata.isCooked,
            is_heat_source=object_metadata.isHeatSource,
            is_cold_source=object_metadata.isColdSource,
            is_sliceable=object_metadata.sliceable,
            is_openable=object_metadata.openable,
            is_open=object_metadata.isOpen,
            openness=object_metadata.openness,
            is_receptacle=object_metadata.receptacle,
            parent_receptacles=object_metadata.parentReceptacles,
            receptacle_object_ids=object_metadata.receptacleObjectIds,
            timestamped_position={datetime_to_ns(datetime.now()): bounding_box.center},
        )

    def crop_timestamped_position(self, min_timestamp: int, max_timestamp: int):
        """
        Removes positions that fall outside a specified timestamp range.

        :param min_timestamp: Minimum timestamp for retention.
        :type min_timestamp: int
        :param max_timestamp: Maximum timestamp for retention.
        :type max_timestamp: int
        """
        for timestamp in self.timestamped_position.keys():
            if timestamp <= min_timestamp and timestamp >= max_timestamp:
                _ = self.timestamped_position.pop(timestamp)

    def capabilities_str(self) -> str:
        capabilities = {
            "pickupable": self.is_pickupable,
            "moveable": self.is_moveable,
            "toggleable": self.is_toggleable,
            "breakable": self.is_breakable,
            "can_fill_with_liquid": self.can_fill_with_liquid,
            "dirtyable": self.is_dirtyable,
            "can_be_used_up": self.can_be_used_up,
            "cookable": self.is_cookable,
            "is_heat_source": self.is_heat_source,
            "is_cold_source": self.is_cold_source,
            "sliceable": self.is_sliceable,
            "openable": self.is_openable,
        }
        caps_on = [k for k, v in capabilities.items() if v is True]
        return ", ".join(caps_on) if caps_on else "None"

    def state_str(self) -> str:
        state = {
            "is_picked_up": self.is_picked_up,
            "is_moving": self.is_moving,
            "is_toggled": self.is_toggled,
            "is_broken": self.is_broken,
            "is_filled_with_liquid": self.is_filled_with_liquid,
            "is_dirty": self.is_dirty,
            "is_used_up": self.is_used_up,
            "is_cooked": self.is_cooked,
            "is_sliced": self.is_sliced,
            "is_open": self.is_open,
        }
        state_on = [k for k, v in state.items() if v is True]
        return ", ".join(state_on) if state_on else "None"

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
            + f"Visible: {self.is_visible}\n"
            + f"Temperature: {self.temperature}\n"
            + f"Openness: {self.openness:.3f}\n"
            + f"Bounding Box: {self.bounding_box}\n"
            + f"Timestamped Positions: {print_timestamped_position(self.timestamped_position)}\n"
            + f"Capabilities (True): {self.capabilities_str()}\n"
            + f"State (True): {self.state_str()}\n"
            + f"Is Receptacle: {self.is_receptacle}\n"
            + f"Parent Receptacles: {', '.join(self.parent_receptacles) if self.parent_receptacles else None}\n"
            + f"Contained Object IDs: {', '.join(map(str, self.receptacle_object_ids)) if self.receptacle_object_ids else None}\n"
            + f"Instance Views: {len(self.instance_views)}\n"
        )
        return obj_str
