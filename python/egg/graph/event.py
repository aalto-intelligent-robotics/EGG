from copy import deepcopy
from typing import Dict, Optional, List, Tuple
import logging
import sys

from egg.graph.node import EventNode
from egg.utils.logger import getLogger
from egg.utils.timestamp import ns_to_datetime


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/event.log",
)


class EventComponents:
    """
    Manages the event nodes within EGG.
    """
    def __init__(self, event_nodes: Dict[int, EventNode] = {}):
        """
        Initializes EventComponents with a dictionary of event nodes.

        :param event_nodes: A dictionary mapping event IDs to EventNode objects.
        :type event_nodes: Dict[int, EventNode]
        """
        self._event_nodes = event_nodes

    def is_empty(self) -> bool:
        return len(self._event_nodes) == 0

    def get_num_events(self):        
        """
        Returns the number of event nodes.

        :returns: The number of events.
        :rtype: int
        """
        return len(self._event_nodes)

    def get_event_ids(self):
        """
        Returns a list of all event node IDs.

        :returns: List of event node IDs.
        :rtype: List[int]
        """
        return list(self._event_nodes.keys())

    def add_event_node(self, event_node: EventNode):
        """
        Adds an event node.

        :param event_node: The event node to add.
        :type event_node: EventNode
        """
        self._event_nodes.update({event_node.node_id: event_node})

    def replace_event_nodes(self, event_nodes: Dict[int, EventNode]):
        """
        Replaces all existing event nodes with a new set.

        :param event_nodes: Dictionary of new event nodes to set.
        :type event_nodes: Dict[int, EventNode]
        """
        self._event_nodes = event_nodes

    def pretty_str(self) -> str:
        """
        Returns a formatted string representation of all event nodes.

        :returns: Human-readable string of events.
        :rtype: str
        """
        event_str = "🕛🕛🕛 EVENT 🕛🕛🕛\n"
        for node in self._event_nodes.values():
            event_str += node.pretty_str()
        return event_str

    def get_event_node_by_id(self, node_id: int) -> Optional[EventNode]:
        """
        Retrieves an event node by its ID.

        :param node_id: The ID of the event node to retrieve.
        :type node_id: int
        :returns: The event node with the given ID or None.
        :rtype: Optional[EventNode]
        """
        if node_id not in self._event_nodes.keys():
            logger.warning(f"Trying to access non-existent object node {node_id}")
        return self._event_nodes.get(node_id)

    def get_event_nodes_by_objects(
        self, object_node_ids: List[int]
    ) -> Dict[int, EventNode]:
        """
        Returns event nodes that involve specified object IDs.

        :param object_node_ids: List of object node IDs to search for.
        :type object_node_ids: List[int]
        :returns: Dictionary of relevant event nodes.
        :rtype: Dict[int, EventNode]
        """
        relevant_event_nodes = {}
        for event_node in self.get_event_nodes().values():
            for id in object_node_ids:
                if id in event_node.involved_object_ids:
                    relevant_event_nodes.update({event_node.node_id: event_node})
                    break
        return relevant_event_nodes
    
    def get_event_node_by_timestamp(self, timestamp: int) -> Optional[EventNode]:
        """
        Finds an event node active at a particular timestamp.

        :param timestamp: The timestamp to search within.
        :type timestamp: int
        :returns: The event node active at the given timestamp or None.
        :rtype: Optional[EventNode]
        """
        for event_node in self.get_event_nodes().values():
            if timestamp >= event_node.start and timestamp <= event_node.end:
                return event_node

    def get_event_nodes(
        self,
        min_timestamp: int = 0,
        max_timestamp: int = sys.maxsize,
        locations_list: Optional[List[str]] = None,
    ) -> Dict[int, EventNode]:
        """
        Retrieves all event nodes within a certain time range and location.

        :param min_timestamp: Minimum timestamp for search.
        :type min_timestamp: int
        :param max_timestamp: Maximum timestamp for search.
        :type max_timestamp: int
        :param locations_list: List of locations to include.
        :type locations_list: Optional[List[str]]
        :returns: Dictionary of event nodes in the specified range and location.
        :rtype: Dict[int, EventNode]
        """
        relevant_event_nodes = {}
        for event_node in self._event_nodes.values():
            if event_node.is_in_time_range(
                min_timestamp, max_timestamp
            ) and event_node.is_in_location(locations_list):
                relevant_event_nodes.update({event_node.node_id: event_node})
        return relevant_event_nodes

    def serialize(self) -> Dict:
        """
        Serializes the event nodes into a dictionary.

        :returns: Dictionary representation of event nodes.
        :rtype: Dict
        """
        event_data = {}
        for event_node in self.get_event_nodes().values():
            event_attr = {
                "event_description": event_node.event_description,
                "start": str(ns_to_datetime(event_node.start)),
                "end": str(ns_to_datetime(event_node.end)),
                "involved_object_ids": event_node.involved_object_ids,
                "timestamped_observation_odom": {},
                "location": event_node.location,
            }
            for timestamp, pos in event_node.timestamped_observation_odom.items():
                timestamp_datetime = ns_to_datetime(timestamp)
                event_attr["timestamped_observation_odom"].update(
                    {
                        str(timestamp_datetime): {
                            "base_odom": [[round(p, 3) for p in pl] for pl in list(pos["base_odom"])],
                            "camera_odom": [[round(p, 3) for p in pl] for pl in list(pos["camera_odom"])],
                        }
                    }
                )
            event_data.update({event_node.node_id: event_attr})
        return event_data

    def get_time_range(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Determines the time range covered by the event nodes.

        :returns: A tuple containing the earliest start and latest end timestamps.
        :rtype: Tuple[Optional[int], Optional[int]]
        """
        min_timestamp = None
        max_timestamp = None

        for node in self.get_event_nodes().values():
            if min_timestamp is None or node.start < min_timestamp:
                min_timestamp = node.start
            if max_timestamp is None or node.end > max_timestamp:
                max_timestamp = node.end

        return min_timestamp, max_timestamp

    def get_locations(self) -> List[str]:
        """
        Retrieves a list of unique locations from the event nodes.

        :returns: List of unique locations.
        :rtype: List[str]
        """
        locations = []
        for node in self.get_event_nodes().values():
            if node.location not in locations:
                locations.append(node.location)
        return locations
