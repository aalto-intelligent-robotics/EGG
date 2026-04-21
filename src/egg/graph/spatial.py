from datetime import datetime
from typing import ClassVar, get_args
import numpy as np
from numpy.typing import NDArray
import logging
from copy import deepcopy
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Any

from egg.graph.node import AgentNode, ObjectNode, RoomNode
from egg.utils.data import Ai2ThorTemperature
from egg.utils.geometry import AxisAlignedBoundingBox, Position
from egg.utils.timestamp import datetime_to_ns, ns_to_datetime
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

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    agent_nodes: dict[int, AgentNode] = Field(default_factory=dict)
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

    def add_agent(self, new_agent_node: AgentNode):
        """
        Adds a room node to the collection.

        :param new_room_node: The room node to add.
        :type new_room_node: RoomNode
        """
        self.agent_nodes.update({new_agent_node.node_id: new_agent_node})

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

    def update_object_state(
        self,
        object_node_id: int,
        new_object_state: ObjectNode.ObjectState,
        timestamp: int | None,
    ):
        if timestamp is None:
            timestamp = datetime_to_ns(datetime.now())

        assert isinstance(self.object_nodes[object_node_id], ObjectNode)
        self.object_nodes[object_node_id].timestamped_states.update(
            {timestamp: new_object_state}
        )

    def update_agent_state(
        self,
        agent_node_id: int,
        new_agent_state: AgentNode.AgentState,
        timestamp: int | None,
    ):
        if timestamp is None:
            timestamp = datetime_to_ns(datetime.now())

        assert isinstance(self.agent_nodes[agent_node_id], AgentNode)
        self.agent_nodes[agent_node_id].timestamped_states.update(
            {timestamp: new_agent_state}
        )

    def remove_agent_node(self, agent_node_id: int):
        """
        Removes an agent node by its ID.

        :param agent_node_id: The ID of the agent node to remove.
        :type agent_node_id: int
        """
        _ = self.agent_nodes.pop(agent_node_id)

    def replace_agent_nodes(self, new_agent_nodes: dict[int, AgentNode]):
        """
        Replaces all existing agent nodes with a new set.

        :param new_agent_nodes: Dictionary of new agent nodes to set.
        :type new_agent_nodes: dict[int, agentNode]
        """
        self.agent_nodes = new_agent_nodes

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
        new_timestamped_states = object_node_1.timestamped_states
        if self.object_nodes[object_node_0_id].timestamped_states:
            for timestamp, state in new_timestamped_states.items():
                _ = self.object_nodes[object_node_0_id].timestamped_states.update(
                    {timestamp: state}
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
            self.object_nodes[id].crop_timestamped_states(min_timestamp, max_timestamp)

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

    def get_agent_node_by_id(self, node_id: int) -> AgentNode | None:
        """
        Retrieves an agent node by its ID.

        :param node_id: ID of the agent node to retrieve.
        :type node_id: int
        :returns: The agent node with the given ID or None.
        :rtype: Optional[agentNode]
        """
        if node_id not in self.agent_nodes.keys():
            logger.warning(
                f"Trying to access non-existent agent node {node_id}, available keys are {self.agent_nodes.keys()}"
            )
        return self.agent_nodes.get(node_id)

    def get_all_object_classes(self) -> set[str]:
        object_class_set: set[str] = set()
        for object_node in self.get_all_object_nodes().values():
            object_class_set.add(object_node.object_class)
        return object_class_set

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

    def get_all_agent_nodes(self) -> dict[int, AgentNode]:
        """
        Retrieves all object nodes.

        :returns: A dictionary of all object nodes.
        :rtype: dict[int, AgenttNode]
        """
        return deepcopy(self.agent_nodes)

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

    def get_object_node_by_name(
        self, node_name: str
    ) -> tuple[int | None, ObjectNode | None]:
        """
        Retrieves an object node by its name.

        :param node_name: Name of the object node to retrieve.
        :type node_name: str
        :returns: The object node with the given name or None.
        :rtype: Optional[ObjectNode]
        """
        for object_node in self.get_all_object_nodes().values():
            if object_node.name == node_name:
                return object_node.node_id, object_node
        logger.warning(f"Trying to look for non-existent object {node_name}")
        return None, None

    def get_room_node_by_name(
        self, node_name: str
    ) -> tuple[int | None, RoomNode | None]:
        """
        Retrieves a room node by its name.

        :param node_name: Name of the room node to retrieve.
        :type node_name: str
        :returns: The room node with the given name or None.
        :rtype: Optional[RoomNode]
        """
        for room_node in self.get_all_room_nodes().values():
            if room_node.name == node_name:
                return room_node.node_id, room_node
        logger.warning(f"Trying to look for non-existent room {node_name}")
        return None, None

    def get_agent_node_by_name(
        self, node_name: str
    ) -> tuple[int | None, AgentNode | None]:
        """
        Retrieves an object node by its name.

        :param node_name: Name of the object node to retrieve.
        :type node_name: str
        :returns: The object node with the given name or None.
        :rtype: Optional[ObjectNode]
        """
        for agent_node in self.get_all_agent_nodes().values():
            if agent_node.name == node_name:
                return agent_node.node_id, agent_node
        logger.warning(f"Trying to look for non-existent agent {node_name}")
        return None, None

    def get_object_by_capabilities(
        self,
        capabilities: list[str],
        object_nodes_to_search: dict[int, ObjectNode] | None = None,
    ) -> dict[int, ObjectNode]:

        valid_capabilities = list(ObjectNode.ObjectCapabilities.model_fields.keys())
        for cap in capabilities:
            assert (
                cap in valid_capabilities
            ), f"{cap} is not a valid capability. Valid capabilities are {valid_capabilities}"

        object_nodes_with_capabilities: dict[int, ObjectNode] = {}

        if object_nodes_to_search is None:
            object_nodes_to_search = self.get_all_object_nodes()

        for id, object_node in object_nodes_to_search.items():
            capabilities_dict = object_node.capabilities.model_dump(mode="python")
            if all([capabilities_dict[cap] for cap in capabilities]):
                object_nodes_with_capabilities.update({id: object_node})
        return object_nodes_with_capabilities

    def get_object_by_states(
        self,
        desired_states: dict[str, bool | str],
        timestamp: int | None = None,
        object_nodes_to_search: dict[int, ObjectNode] | None = None,
    ) -> dict[int, ObjectNode]:
        invalid_states = {"position", "bounding_box", "instance_view", "openness"}
        valid_states = set(ObjectNode.ObjectState.model_fields.keys()) - invalid_states

        for state_name, state_value in desired_states.items():
            assert (
                state_name in valid_states
            ), f"{state_name} is not a valid state. Valid capabilities are {valid_states}"
            if state_name == "temperature" and state_value not in get_args(
                Ai2ThorTemperature
            ):
                raise ValueError(
                    f"Valid temperature values are: {get_args(Ai2ThorTemperature)}, got {state_value}"
                )

        object_nodes_with_desired_states: dict[int, ObjectNode] = {}

        if object_nodes_to_search is None:
            object_nodes_to_search = self.get_all_object_nodes()
        if timestamp is None:
            timestamp = datetime_to_ns(datetime.now())

        for id, object_node in object_nodes_to_search.items():
            _, prev_state = object_node.get_previous_timestamp_and_states(
                ref_timestamp=timestamp
            )
            if isinstance(prev_state, ObjectNode.ObjectState):
                prev_state_dict = prev_state.model_dump(mode="python")
                if all(
                    [
                        prev_state_dict[state] == desired_states[state]
                        for state in desired_states.keys()
                    ]
                ):
                    object_nodes_with_desired_states.update({id: object_node})
        return object_nodes_with_desired_states

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
        for agent_node in self.agent_nodes.values():
            spatial_str += agent_node.pretty_str()
        return spatial_str
