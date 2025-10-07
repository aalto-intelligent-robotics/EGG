from copy import deepcopy
import logging
import sys
from typing import List, Dict, Optional, Tuple

from egg.graph.node import EventNode, ObjectNode
from egg.graph.egg import EGG
from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="pruning/egg_slicer.log",
)


class EGGSlicer:
    def __init__(self, egg: EGG):
        self.egg: EGG = deepcopy(egg)
        self.pruned_egg: EGG = deepcopy(egg)

    def reset_pruned_egg(self):
        self.pruned_egg = deepcopy(self.egg)

    def get_events_from_object(
        self,
        object_node: ObjectNode,
        min_timestamp: int = 0,
        max_timestamp: int = sys.maxsize,
    ) -> Dict[int, EventNode]:
        relevant_event_nodes = {}
        event_nodes_in_time_range = self.pruned_egg.event_graph.get_event_nodes(
            min_timestamp, max_timestamp
        )
        for event_node in event_nodes_in_time_range.values():
            if object_node.node_id in event_node.involved_object_ids:
                relevant_event_nodes.update({event_node.node_id: event_node})
        return relevant_event_nodes

    def get_objects_from_events(
        self, event_nodes_dict: Dict[int, EventNode]
    ) -> Dict[int, ObjectNode]:
        relevant_object_nodes = {}

        for event_node in event_nodes_dict.values():
            for object_node_id in event_node.involved_object_ids:
                if object_node_id not in relevant_object_nodes.keys():
                    object_node = (
                        self.pruned_egg.get_spatial_graph().get_object_node_by_id(
                            object_node_id
                        )
                    )
                    relevant_object_nodes.update({object_node_id: object_node})
        return relevant_object_nodes

    def prune_graph_by_events(self, event_nodes: Dict[int, EventNode]):
        pruned_edges = []
        self.pruned_egg.spatial_graph.replace_object_nodes(
            self.get_objects_from_events(event_nodes)
        )

        for edge in self.pruned_egg.get_event_object_edges():
            if edge.source_node_id in event_nodes.keys():
                pruned_edges.append(edge)
        self.pruned_egg.set_event_object_edges(pruned_edges)

    def prune_graph_by_objects(self, object_node_ids: List[int]):
        pruned_edges = []
        self.pruned_egg.event_graph.replace_event_nodes(
            self.pruned_egg.event_graph.get_event_nodes_by_objects(object_node_ids)
        )
        for edge in self.pruned_egg.get_event_object_edges():
            if edge.target_node_id in object_node_ids:
                pruned_edges.append(edge)
        self.pruned_egg.set_event_object_edges(pruned_edges)

    def prune_graph_by_location(
        self,
        locations_list: List[str],
    ):
        self.pruned_egg.event_graph.replace_event_nodes(
            self.pruned_egg.event_graph.get_event_nodes(locations_list=locations_list)
        )
        self.prune_graph_by_events(self.pruned_egg.event_graph.get_event_nodes())

        self.pruned_egg.set_event_graph(self.pruned_egg.event_graph)

    def prune_graph_by_time_range(
        self, min_timestamp: int = 0, max_timestamp: int = sys.maxsize
    ):
        self.pruned_egg.event_graph.replace_event_nodes(
            self.pruned_egg.event_graph.get_event_nodes(
                min_timestamp=min_timestamp, max_timestamp=max_timestamp
            )
        )
        self.prune_graph_by_events(
            self.pruned_egg.event_graph.get_event_nodes(
                min_timestamp=min_timestamp, max_timestamp=max_timestamp
            )
        )
        self.pruned_egg.spatial_graph.set_object_nodes_to_time_range(
            min_timestamp, max_timestamp
        )

    def merge_events_and_objects(self, object_ids: List[int], event_ids: List[int]):
        valid_object_ids = set()
        for event_id in event_ids:
            event_node = self.pruned_egg.get_event_graph().get_event_node_by_id(
                event_id
            )
            if event_node is not None:
                for object_id in object_ids:
                    if object_id in event_node.involved_object_ids:
                        valid_object_ids.add(object_id)
        if len(valid_object_ids) == 0:
            valid_object_ids = object_ids

        if len(valid_object_ids) != 0:
            self.prune_graph_by_objects(list(valid_object_ids))

    def get_time_range(self) -> Tuple[Optional[int], Optional[int]]:
        return self.egg.get_event_graph().get_time_range()

    def get_locations(self) -> List[str]:
        return self.egg.get_event_graph().get_locations()
