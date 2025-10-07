from dataclasses import dataclass, field
import sys
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
import torch
import logging

from egg.utils.logger import getLogger
from egg.utils.timestamp import (
    datetime_to_ns,
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
    node_id: int


@dataclass
class SpatialNode(GraphNode):
    name: str


@dataclass
class EventNode(GraphNode):
    event_description: str
    start: int
    end: int
    # {timestamp: {"base_odom": [[x,y,z],[x,y,z,w]], "camera_odom": [[x,y,z],[x,y,z,w]]}}
    timestamped_observation_odom: Dict[int, Dict[str, List]]
    involved_object_ids: List[int]
    location: str

    def is_in_time_range(
        self, min_timestamp: int = 0, max_timestamp: int = sys.maxsize
    ):
        if self.start >= min_timestamp and self.end <= max_timestamp:
            return True
        return False

    def is_in_location(self, location_list: Optional[List[str]] = None):
        return location_list is None or self.location in location_list

    def get_first_observation_pos(self) -> NDArray:
        obs_pos = next(iter(self.timestamped_observation_odom.values()))["base_odom"][0]
        return np.array(obs_pos)

    def get_first_observation_odom(self) -> Dict:
        return next(iter(self.timestamped_observation_odom.values()))

    def pretty_str(self) -> str:
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
    object_class: str
    timestamped_position: Dict[int, NDArray]
    visual_embedding: Optional[torch.Tensor] = None
    instance_views: List[NDArray] = field(default_factory=list)
    caption: Optional[str] = None

    def is_in_event(self, event_node: EventNode):
        return self.node_id in event_node.involved_object_ids

    def cut_timestamped_position(self, min_timestamp: int, max_timestamp: int):
        for timestamp in self.timestamped_position.keys():
            if timestamp <= min_timestamp and timestamp >= max_timestamp:
                self.timestamped_position.pop(timestamp)

    def get_closest_start_end_timestamps(self, start: int, end: int):
        # TODO: This is only correct for this case, because we only collect start/end
        # timestamps
        found_start = False
        found_end = False
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
        first_timestamp = next(iter(self.timestamped_position.keys()))
        return first_timestamp <= timestamp

    def get_previous_timestamp_and_position(self, ref_timestamp: int):
        # Get the position that is closest to this timestamp
        if self.has_been_seen:
            prev_timestamp = None
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
