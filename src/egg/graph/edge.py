from dataclasses import dataclass
from enum import Enum
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/edge.log",
)


@dataclass
class GraphEdge:
    """
    Represents a generic edge in a graph, characterized by its identifiers for
    the source and target nodes.

    :param edge_id: Unique identifier for the edge.
    :type edge_id: int
    :param source_node_id: Node identifier where the edge originates.
    :type source_node_id: int
    :param target_node_id: Node identifier where the edge points to.
    :type target_node_id: int
    """
    edge_id: int
    source_node_id: int
    target_node_id: int


@dataclass
class EventObjectEdge(GraphEdge):
    """
    Represents an edge in the event-object graph, extending GraphEdge by adding
    an 'object_role' to describe the relationship between events and objects.

    :param object_role: Description of the object's role in the event context.
    :type object_role: str
    """
    object_role: str

    def pretty_str(self) -> str:
        """
        Constructs a human-readable string representing the event-object edge
        with detailed information including edge identity and object role.

        :returns: A descriptive string of the event-object edge's properties.
        :rtype: str
        """
        event_obj_edge_str = (
            "\n🔗 Edge info:\n"
            + f"- Edge ID: {self.edge_id}\n"
            + f"Object role: {self.object_role}\n"
            + f"From: {self.source_node_id} - To: {self.target_node_id}\n"
        )
        return event_obj_edge_str
