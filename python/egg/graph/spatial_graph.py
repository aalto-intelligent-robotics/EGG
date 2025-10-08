from typing import Dict, Optional, Tuple
import logging
from numpy.typing import NDArray
from copy import deepcopy

from egg.perception.instance_matching import are_similar_objects
from egg.graph.node import ObjectNode, RoomNode
from egg.utils.timestamp import ns_to_datetime
from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/spatial_graph.log",
)


class SpatialGraph:
    def __init__(
        self,
        object_nodes: Dict[int, ObjectNode] = {},
        room_nodes: Dict[int, RoomNode] = {},
        map_views: Dict[int, NDArray] = {},
    ):
        self._object_nodes = object_nodes
        self._map_views = map_views
        self._room_nodes = room_nodes

    def is_new_node(
        self, new_object_node: ObjectNode, use_gt_id: bool
    ) -> Tuple[bool, int]:
        for object_node in self.get_all_object_nodes().values():
            if are_similar_objects(
                object_node_0=object_node,
                object_node_1=new_object_node,
                use_gt=use_gt_id,
            ):
                return False, object_node.node_id
        return True, new_object_node.node_id

    def add_room_node(self, new_room_node: RoomNode):
        self._room_nodes.update({new_room_node.node_id: new_room_node})

    def remove_room_node(self, room_node_id: int):
        self._room_nodes.pop(room_node_id)

    def replace_room_nodes(self, new_room_nodes: Dict[int, RoomNode]):
        self._room_nodes = new_room_nodes

    def add_object_node(self, new_object_node: ObjectNode):
        self._object_nodes.update({new_object_node.node_id: new_object_node})

    def remove_object_node(self, object_node_id: int):
        self._object_nodes.pop(object_node_id)

    def replace_object_nodes(self, new_object_nodes: Dict[int, ObjectNode]):
        self._object_nodes = new_object_nodes

    def merge_object_nodes(self, object_node_0_id: int, object_node_1: ObjectNode):
        logger.debug(
            f"Merging {object_node_1.name} {object_node_1.node_id} and {object_node_0_id}"
        )
        new_timestamped_position = object_node_1.timestamped_position
        for timestamp, pos in new_timestamped_position.items():
            self._object_nodes[object_node_0_id].timestamped_position.update(
                {timestamp: pos}
            )

    def set_object_nodes_to_time_range(self, min_timestamp: int, max_timestamp: int):
        for id in self._object_nodes.keys():
            self._object_nodes[id].cut_timestamped_position(
                min_timestamp, max_timestamp
            )

    def get_object_node_by_id(self, node_id: int) -> Optional[ObjectNode]:
        if node_id not in self._object_nodes.keys():
            logger.warning(
                f"Trying to access non-existent object node {node_id}, available keys are {self._object_nodes.keys()}"
            )
        return self._object_nodes.get(node_id)

    def get_all_room_nodes(self) -> Dict[int, RoomNode]:
        return deepcopy(self._room_nodes)

    def get_all_object_nodes(self) -> Dict[int, ObjectNode]:
        return deepcopy(self._object_nodes)

    def get_object_nodes_by_class(self, object_class: str) -> Dict[int, ObjectNode]:
        object_nodes_by_class = {}
        for object_node in self.get_all_object_nodes().values():
            if object_node.object_class == object_class:
                object_nodes_by_class.update({object_node.node_id: object_node})
        return object_nodes_by_class

    def get_object_node_by_name(self, node_name: str) -> Optional[ObjectNode]:
        for object_node in self.get_all_object_nodes().values():
            if object_node.name == node_name:
                return object_node
        logger.warning(f"Trying to look for non-existent object {node_name}")
        return None

    def get_room_node_by_name(self, node_name: str) -> Optional[RoomNode]:
        for room_node in self.get_all_room_nodes().values():
            if room_node.name == node_name:
                return room_node
        logger.warning(f"Trying to look for non-existent room {node_name}")
        return None

    def pretty_str(self):
        spatial_graph_str = "📦📦📦 SPATIAL GRAPH 📦📦📦\n"
        for object_node in self._object_nodes.values():
            spatial_graph_str += object_node.pretty_str()
        for room_node in self._room_nodes.values():
            spatial_graph_str += room_node.pretty_str()
        return spatial_graph_str

    def serialize(self) -> Dict:
        # TODO: Add room nodes serialization
        spatial_data = {}

        for object_node in self.get_all_object_nodes().values():
            attr_data = {
                "node_id": object_node.node_id,
                "object_class": object_node.object_class,
                "name": object_node.name,
                "timestamped_position": {},
                "caption": object_node.caption,
            }
            for timestamp, pos in object_node.timestamped_position.items():
                timestamp_datetime = ns_to_datetime(timestamp)
                attr_data["timestamped_position"].update(
                    {str(timestamp_datetime): list(pos)}
                )

            spatial_data.update({object_node.node_id: {"attributes": attr_data}})

        return spatial_data
