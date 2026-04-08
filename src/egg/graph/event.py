from copy import deepcopy
import logging
from pydantic import BaseModel, Field
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


class EventComponents(BaseModel):
    """
    Manages the event nodes within EGG.
    """

    event_nodes: dict[int, EventNode] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.event_nodes) == 0

    def get_num_events(self) -> int:
        """
        Returns the number of event nodes.

        :returns: The number of events.
        :rtype: int
        """
        return len(self.event_nodes)

    def get_event_ids(self) -> list[int]:
        """
        Returns a list of all event node IDs.

        :returns: List of event node IDs.
        :rtype: List[int]
        """
        return list(self.event_nodes.keys())

    def add_event_node(self, event_node: EventNode):
        """
        Adds an event node.

        :param event_node: The event node to add.
        :type event_node: EventNode
        """
        self.event_nodes.update({event_node.node_id: event_node})

    def replace_event_nodes(self, event_nodes: dict[int, EventNode]):
        """
        Replaces all existing event nodes with a new set.

        :param event_nodes: Dictionary of new event nodes to set.
        :type event_nodes: Dict[int, EventNode]
        """
        self.event_nodes = event_nodes

    def get_event_node_by_id(self, node_id: int) -> EventNode | None:
        """
        Retrieves an event node by its ID.

        :param node_id: The ID of the event node to retrieve.
        :type node_id: int
        :returns: The event node with the given ID or None.
        :rtype: Optional[EventNode]
        """
        if node_id not in self.event_nodes.keys():
            logger.warning(f"Trying to access non-existent object node {node_id}")
        return self.event_nodes.get(node_id)

    def get_event_nodes_by_objects(
        self, object_node_ids: list[int]
    ) -> dict[int, EventNode]:
        """
        Returns event nodes that involve specified object IDs.

        :param object_node_ids: List of object node IDs to search for.
        :type object_node_ids: List[int]
        :returns: Dictionary of relevant event nodes.
        :rtype: Dict[int, EventNode]
        """
        relevant_event_nodes: dict[int, EventNode] = {}
        raise NotImplementedError
        # for event_node in self.get_event_nodes().values():
        #     for id in object_node_ids:
        #         if id in event_node.involved_object_ids:
        #             relevant_event_nodes.update({event_node.node_id: event_node})
        #             break
        return relevant_event_nodes

    def get_event_nodes(
        self,
        min_timestamp: int = 0,
        max_timestamp: int = sys.maxsize,
        locations_list: list[str] | None = None,
    ) -> dict[int, EventNode]:
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
        relevant_event_nodes: dict[int, EventNode] = {}
        for event_node in self.event_nodes.values():
            if event_node.is_in_time_range(
                min_timestamp, max_timestamp
            ) and event_node.is_in_location(locations_list):
                relevant_event_nodes.update({event_node.node_id: event_node})
        return relevant_event_nodes

    def pretty_str(self) -> str:
        """
        Returns a formatted string representation of all event nodes.

        :returns: Human-readable string of events.
        :rtype: str
        """
        event_str = "🕛🕛🕛 EVENT 🕛🕛🕛\n"
        for node in self.event_nodes.values():
            event_str += node.pretty_str()
        return event_str
