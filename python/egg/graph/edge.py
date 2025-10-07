from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/edge.log",
)


class SpatialRelationship(Enum):
    ON = 1
    SUPPORTS = 2
    NEARBY = 3
    INSIDE = 4
    CONTAINS = 5


@dataclass
class GraphEdge:
    edge_id: int
    source_node_id: int
    target_node_id: int


@dataclass
class EventObjectEdge(GraphEdge):
    object_role: str

    def pretty_str(self) -> str:
        event_obj_edge_str = (
            "\n🔗 Edge info:\n"
            + f"- Edge ID: {self.edge_id}\n"
            + f"Object role: {self.object_role}\n"
            + f"From: {self.source_node_id} - To: {self.target_node_id}\n"
        )
        return event_obj_edge_str


@dataclass
class ObjectObjectEdge(GraphEdge):
    source_target_relationship: int
    target_source_relationship: int
