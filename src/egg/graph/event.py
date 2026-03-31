from copy import deepcopy
import logging
from pydantic import BaseModel, Field

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

    def get_num_events(self):
        """
        Returns the number of event nodes.

        :returns: The number of events.
        :rtype: int
        """
        return len(self.event_nodes)

    def get_event_ids(self):
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
