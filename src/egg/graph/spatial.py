from typing import ClassVar
import numpy as np
from numpy.typing import NDArray
import logging
from copy import deepcopy
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Any

from egg.graph.node import ObjectNode, RoomNode
from egg.utils.timestamp import ns_to_datetime
from egg.utils.logger import getLogger
from egg.perception.instance_matching import are_similar_objects


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/spatial.log",
)


class SpatialComponents(BaseModel):
    """
    Manages spatial components of EGG, including object nodes and room nodes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    room_nodes: dict[int, RoomNode] = Field(default_factory=dict)
    object_nodes: dict[int, ObjectNode] = Field(default_factory=dict)
    map_views: dict[int, NDArray[np.uint8]] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return bool(self.object_nodes) and bool(self.room_nodes)

    def is_new_node(
        self, new_object_node: ObjectNode, use_gt_id: bool
    ) -> tuple[bool, int]:
        """
        Determines if a given object node is new or similar to existing nodes.

        :param new_object_node: The object node to check.
        :type new_object_node: ObjectNode
        :param use_gt_id: Flag indicating whether to use ground truth ID for comparison.
        :type use_gt_id: bool
        :returns: Tuple containing a boolean indicating if it's new and the object's ID.
        :rtype: Tuple[bool, int]
        """
        for object_node in self.get_all_object_nodes().values():
            if are_similar_objects(
                object_node_0=object_node,
                object_node_1=new_object_node,
                use_gt=use_gt_id,
            ):
                return False, object_node.node_id
        return True, new_object_node.node_id

    def add_room_node(self, new_room_node: RoomNode):
        """
        Adds a room node to the collection.

        :param new_room_node: The room node to add.
        :type new_room_node: RoomNode
        """
        self.room_nodes.update({new_room_node.node_id: new_room_node})

    def remove_room_node(self, room_node_id: int):
        """
        Removes a room node by its ID.

        :param room_node_id: The ID of the room node to remove.
        :type room_node_id: int
        """
        _ = self.room_nodes.pop(room_node_id)

    def replace_room_nodes(self, new_room_nodes: dict[int, RoomNode]):
        """
        Replaces all existing room nodes with a new set.

        :param new_room_nodes: Dictionary of new room nodes to set.
        :type new_room_nodes: Dict[int, RoomNode]
        """
        self.room_nodes = new_room_nodes

    def add_object_node(self, new_object_node: ObjectNode):
        """
        Adds an object node to the collection.

        :param new_object_node: The object node to add.
        :type new_object_node: ObjectNode
        """
        self.object_nodes.update({new_object_node.node_id: new_object_node})

    def remove_object_node(self, object_node_id: int):
        """
        Removes an object node by its ID.

        :param object_node_id: The ID of the object node to remove.
        :type object_node_id: int
        """
        _ = self.object_nodes.pop(object_node_id)

    def replace_object_nodes(self, new_object_nodes: dict[int, ObjectNode]):
        """
        Replaces all existing object nodes with a new set.

        :param new_object_nodes: Dictionary of new object nodes to set.
        :type new_object_nodes: dict[int, ObjectNode]
        """
        self.object_nodes = new_object_nodes

    def merge_object_nodes(self, object_node_0_id: int, object_node_1: ObjectNode):
        """
        Merges two object nodes by updating their timestamp positions.

        :param object_node_0_id: ID of the existing object node.
        :type object_node_0_id: int
        :param object_node_1: The new object node to merge.
        :type object_node_1: ObjectNode
        """
        assert isinstance(self.object_nodes[object_node_0_id], ObjectNode)
        logger.debug(
            f"Merging {object_node_1.name} {object_node_1.node_id} and {object_node_0_id}"
        )
        new_timestamped_position = object_node_1.timestamped_position
        if self.object_nodes[object_node_0_id].timestamped_position:
            for timestamp, pos in new_timestamped_position.items():
                _ = self.object_nodes[object_node_0_id].timestamped_position.update(
                    {timestamp: pos}
                )

    def set_object_nodes_to_time_range(self, min_timestamp: int, max_timestamp: int):
        """
        Trims object nodes' timestamps to within a specified range.

        :param min_timestamp: Starting timestamp of the range.
        :type min_timestamp: int
        :param max_timestamp: Ending timestamp of the range.
        :type max_timestamp: int
        """
        for id in self.object_nodes.keys():
            self.object_nodes[id].crop_timestamped_position(
                min_timestamp, max_timestamp
            )

    def get_object_node_by_id(self, node_id: int) -> ObjectNode | None:
        """
        Retrieves an object node by its ID.

        :param node_id: ID of the object node to retrieve.
        :type node_id: int
        :returns: The object node with the given ID or None.
        :rtype: Optional[ObjectNode]
        """
        if node_id not in self.object_nodes.keys():
            logger.warning(
                f"Trying to access non-existent object node {node_id}, available keys are {self.object_nodes.keys()}"
            )
        return self.object_nodes.get(node_id)

    def get_all_room_nodes(self) -> dict[int, RoomNode]:
        """
        Retrieves all room nodes.

        :returns: A dictionary of all room nodes.
        :rtype: Dict[int, RoomNode]
        """
        return deepcopy(self.room_nodes)

    def get_all_object_nodes(self) -> dict[int, ObjectNode]:
        """
        Retrieves all object nodes.

        :returns: A dictionary of all object nodes.
        :rtype: dict[int, ObjectNode]
        """
        return deepcopy(self.object_nodes)

    def get_object_nodes_by_class(self, object_class: str) -> dict[int, ObjectNode]:
        """
        Retrieves object nodes by their class.

        :param object_class: Class name to retrieve object nodes for.
        :type object_class: str
        :returns: dictionary of object nodes that match the given class.
        :rtype: dict[int, ObjectNode]
        """
        object_nodes_by_class: dict[int, ObjectNode] = {}
        for object_node in self.get_all_object_nodes().values():
            if object_node.object_class == object_class:
                object_nodes_by_class.update({object_node.node_id: object_node})
        return object_nodes_by_class

    def get_object_node_by_name(self, node_name: str) -> ObjectNode | None:
        """
        Retrieves an object node by its name.

        :param node_name: Name of the object node to retrieve.
        :type node_name: str
        :returns: The object node with the given name or None.
        :rtype: Optional[ObjectNode]
        """
        for object_node in self.get_all_object_nodes().values():
            if object_node.name == node_name:
                return object_node
        logger.warning(f"Trying to look for non-existent object {node_name}")
        return None

    def get_room_node_by_name(self, node_name: str) -> RoomNode | None:
        """
        Retrieves a room node by its name.

        :param node_name: Name of the room node to retrieve.
        :type node_name: str
        :returns: The room node with the given name or None.
        :rtype: Optional[RoomNode]
        """
        for room_node in self.get_all_room_nodes().values():
            if room_node.name == node_name:
                return room_node
        logger.warning(f"Trying to look for non-existent room {node_name}")
        return None

    def pretty_str(self) -> str:
        """
        Generates a formatted string representation of all spatial components.

        :returns: A descriptive string of spatial nodes.
        :rtype: str
        """
        spatial_str = "📦📦📦 SPATIAL 📦📦📦\n"
        for object_node in self.object_nodes.values():
            spatial_str += object_node.pretty_str()
        for room_node in self.room_nodes.values():
            spatial_str += room_node.pretty_str()
        return spatial_str
