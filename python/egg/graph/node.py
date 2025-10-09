from dataclasses import dataclass, field
import sys
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
import torch
import logging

from egg.utils.logger import getLogger
from egg.utils.timestamp import (
    ns_to_datetime,
    print_timestamped_position,
    print_timestamped_observation_odom,
)


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/node.log",
)


@dataclass
class GraphNode:
    """
    Represents a basic node in a graph with a unique identifier.

    :param node_id: Unique identifier for the node.
    :type node_id: int
    """
    node_id: int


@dataclass
class SpatialNode(GraphNode):
    """
    Represents a spatial node, extending the basic graph node by adding a name.

    :param name: Name of the spatial node.
    :type name: str
    """
    name: str


@dataclass
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
    :type timestamped_observation_odom: Dict[int, Dict[str, List]]
    :param involved_object_ids: List of object IDs involved in the event.
    :type involved_object_ids: List[int]
    :param location: Location where the event takes place.
    :type location: str
    """
    event_description: str
    start: int
    end: int
    timestamped_observation_odom: Dict[int, Dict[str, List]]
    involved_object_ids: List[int]
    location: str

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

    def is_in_location(self, location_list: Optional[List[str]] = None):
        """
        Checks if the event location is in a given list of locations.

        :param location_list: List of locations to check against.
        :type location_list: Optional[List[str]]
        :returns: True if the event location is in the list, False otherwise.
        :rtype: bool
        """
        return location_list is None or self.location in location_list

    def get_first_observation_pos(self) -> NDArray:
        """
        Retrieves the first observed position from the odometry data.

        :returns: The first observed position as a numpy array.
        :rtype: np.ndarray
        """
        obs_pos = next(iter(self.timestamped_observation_odom.values()))["base_odom"][0]
        return np.array(obs_pos)

    def get_first_observation_odom(self) -> Dict:
        """
        Retrieves the odometry data from the first observation.

        :returns: A dictionary containing the first observation odometry data.
        :rtype: Dict
        """
        return next(iter(self.timestamped_observation_odom.values()))

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
            + f"Involved objects: {self.involved_object_ids}\n"
            + f"Timestamped Observation Positions: {print_timestamped_observation_odom(self.timestamped_observation_odom)}\n"
        )
        return event_node_str


@dataclass
class RoomNode(SpatialNode):
    """
    Represents a room node, extending SpatialNode by adding position data.

    :param position: Position of the room as a numpy array.
    :type position: np.ndarray
    """
    position: NDArray

    def pretty_str(self) -> str:
        room_node_str = (
            "\n🏠 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Node type: Room\n"
            + f"Name: {self.name}\n"
            + f"Position: {self.position}\n"
        )
        return room_node_str


@dataclass
class ObjectNode(SpatialNode):
    """
    Represents an object node in the graph, adding specific metadata relevant
    to object tracking and visualization.

    :param object_class: Classification of the object.
    :type object_class: str
    :param timestamped_position: Position data indexed by timestamps.
    :type timestamped_position: Dict[int, np.ndarray]
    :param visual_embedding: [Optional] A visual embedding tensor.
    :type visual_embedding: Optional[torch.Tensor]
    :param instance_views: List of visual observations of the object.
    :type instance_views: List[np.ndarray]
    :param caption: [Optional] A textual description of the object.
    :type caption: Optional[str]
    """
    object_class: str
    timestamped_position: Dict[int, NDArray]
    visual_embedding: Optional[torch.Tensor] = None
    instance_views: List[NDArray] = field(default_factory=list)
    caption: Optional[str] = None

    def is_in_event(self, event_node: EventNode):
        """
        Checks if this object is involved in a specified event.

        :param event_node: The event node to check against.
        :type event_node: EventNode
        :returns: True if the object is involved in the event, False otherwise.
        :rtype: bool
        """
        return self.node_id in event_node.involved_object_ids

    def cut_timestamped_position(self, min_timestamp: int, max_timestamp: int):
        """
        Removes positions that fall outside a specified timestamp range.

        :param min_timestamp: Minimum timestamp for retention.
        :type min_timestamp: int
        :param max_timestamp: Maximum timestamp for retention.
        :type max_timestamp: int
        """
        for timestamp in self.timestamped_position.keys():
            if timestamp <= min_timestamp and timestamp >= max_timestamp:
                self.timestamped_position.pop(timestamp)

    def get_closest_start_end_timestamps(self, start: int, end: int):
        """
        Finds the closest start and end timestamps within the object's timeline.

        :param start: Reference start timestamp.
        :type start: int
        :param end: Reference end timestamp.
        :type end: int
        :returns: Tuple of the closest start and end timestamps found.
        :rtype: Tuple[int, int]
        """
        found_start = False
        closest_start = None
        closest_end = None
        for timestamp in self.timestamped_position.keys():
            if timestamp >= start and timestamp <= end:
                if found_start:
                    closest_end = timestamp
                    break
                if timestamp >= start:
                    closest_start = timestamp
                    found_start = True
        assert (
            closest_start is not None and closest_end is not None
        ), f"Invalid start/end: start: {start} end: {end}"
        return closest_start, closest_end

    def has_been_seen(self, timestamp: int):
        """
        Checks if the object has been observed before a given timestamp.

        :param timestamp: The reference timestamp.
        :type timestamp: int
        :returns: True if the object has been seen, False otherwise.
        :rtype: bool
        """
        first_timestamp = next(iter(self.timestamped_position.keys()))
        return first_timestamp <= timestamp

    def get_previous_timestamp_and_position(self, ref_timestamp: int):
        """
        Gets the previous timestamp and position relative to a reference timestamp.

        :param ref_timestamp: The reference timestamp.
        :type ref_timestamp: int
        :returns: Tuple of the previous timestamp and position, or (None, None).
        :rtype: Tuple[Optional[int], Optional[np.ndarray]]
        """
        if self.has_been_seen:
            prev_timestamp = next(iter(self.timestamped_position.keys()))
            for timestamp in self.timestamped_position.keys():
                if timestamp >= ref_timestamp:
                    if prev_timestamp is not None:
                        return (prev_timestamp, self.timestamped_position[prev_timestamp])
                    else:
                        return (timestamp, self.timestamped_position[timestamp])
                else:
                    prev_timestamp = timestamp
            return (prev_timestamp, self.timestamped_position[prev_timestamp])
        else:
            return (None, None)

    def pretty_str(self) -> str:
        """
        Generates a formatted string representation of the object node details.

        :returns: String representing the object node.
        :rtype: str
        """
        obj_node_str = (
            "\n📦 Node info:\n"
            + f"- Node ID: {self.node_id}\n"
            + f"Node type: Object\n"
            + f"Name: {self.name}\n"
            + f"Object class: {self.object_class}\n"
            + f"Timestamped Positions: {print_timestamped_position(self.timestamped_position)}\n"
            + f"Object description: {self.caption}\n"
        )
        return obj_node_str
