# pyright: reportExplicitAny=none, reportAny=none
from copy import deepcopy
from datetime import datetime
import tomllib
import logging
from pathlib import Path
from typing import Self, Any, ClassVar
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ConfigDict
from typing_extensions import override

from egg.graph.edge import SpatialEdge
from egg.graph.event import EventComponents
from egg.graph.node import AgentNode, ObjectNode, RoomNode
from egg.graph.spatial import SpatialComponents
from egg.utils.data import (
    Ai2ThorAgentMetadata,
    Ai2ThorObjectMetadata,
    Ai2ThorRoomMetadata,
)
from egg.utils.logger import getLogger
from egg.utils.timestamp import datetime_to_ns

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/egg.log",
)


class EGG(BaseModel):
    """
    EGG (Event-Grounding Graph) framework that grounds events semantic context to spatial geometrics.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    spatial: SpatialComponents
    events: EventComponents
    use_gt_id: bool = Field(default=True, exclude=True)
    use_gt_caption: bool = Field(default=True, exclude=True)
    use_guided_auto_caption: bool = Field(default=True, exclude=True)
    device: str = Field(default="cuda:0", exclude=True)
    do_sample: bool = Field(default=False, exclude=True)
    object_types_config_file: str | None = Field(default=None, exclude=True)
    object_types_config: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        exclude=True,
    )
    node_id_gen: int = Field(default=0, ge=0)

    @override
    def model_post_init(self, context: Any) -> None:
        if self.object_types_config_file:
            try:
                config_path = Path(self.object_types_config_file)
                with open(config_path, "rb") as f_objects:
                    self.object_types_config = tomllib.load(f_objects)
            except FileNotFoundError:
                logger.error(f"Object types config file not found: {self.object_types_config_file}")
                self.object_types_config = {}
            except tomllib.TOMLDecodeError as e:
                logger.error(f"Failed to parse TOML file {self.object_types_config_file}: {e}")
                self.object_types_config = {}
        return super().model_post_init(context)

    def is_empty(self) -> bool:
        return self.spatial.is_empty() and self.events.is_empty()

    def gen_id(self) -> int:
        self.node_id_gen += 1
        return self.node_id_gen

    def get_compatible_receptacles(self, object_name: str) -> list[ObjectNode]:
        if not self.object_types_config:
            return []
        compatible_receptacles: list[ObjectNode] = []
        _, object_node = self.spatial.get_object_node_by_name(node_name=object_name)
        assert isinstance(
            object_node, ObjectNode
        ), f"{object_name} does not exist in EGG"
        object_class = object_node.object_class
        compatible_classes: list[str] = self.object_types_config[object_class][
            "compatible_receptacles"
        ]
        for obj_class in compatible_classes:
            compatible_receptacles += list(
                self.spatial.get_object_nodes_by_class(object_class=obj_class).values()
            )
        return compatible_receptacles

    @classmethod
    def from_ai2thor(
        cls,
        ai2thor_agent_metadata: dict[str, JsonValue],
        ai2thor_object_metadata: list[dict[str, JsonValue]],
        ai2thor_house_metadata: dict[str, JsonValue] | str,
        object_types_config_file: str | None = None,
    ) -> Self:

        node_id_gen: int = 0
        dict_adapter = TypeAdapter(dict[str, JsonValue])
        list_adapter = TypeAdapter(list[dict[str, JsonValue]])

        agent_metadata: dict[str, JsonValue] = dict_adapter.validate_python(
            ai2thor_agent_metadata
        )
        agent_state = AgentNode.AgentState.from_ai2thor(
            agent_metadata=Ai2ThorAgentMetadata.model_validate(agent_metadata)
        )
        agent_node = cls._create_agent_node(
            node_id=node_id_gen,
            agent_state=agent_state,
        )
        node_id_gen += 1

        egg = cls(
            spatial=SpatialComponents(),
            events=EventComponents(),
            object_types_config_file=object_types_config_file,
        )

        egg.spatial.add_agent(new_agent_node=agent_node)

        if isinstance(ai2thor_house_metadata, str):
            # TODO: Implement iTHOR
            raise NotImplementedError
        else:
            rooms_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
                ai2thor_house_metadata["rooms"]
            )
        object_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
            ai2thor_object_metadata
        )

        room_nodes = cls._create_room_nodes(
            rooms_metadata=rooms_metadata,
            start_node_id=node_id_gen,
        )
        node_id_gen += len(room_nodes)
        for room_node in room_nodes:
            egg.spatial.add_room_node(new_room_node=room_node)

        object_nodes = cls._create_object_nodes(
            object_metadata=object_metadata,
            start_node_id=node_id_gen,
        )
        node_id_gen += len(object_nodes)
        for obj_node in object_nodes:
            egg.spatial.add_object_node(new_object_node=obj_node)

        cls._create_receptacle_edges(
            egg=egg,
        )

        egg.node_id_gen = node_id_gen
        return egg

    @classmethod
    def _create_agent_node(
        cls,
        node_id: int,
        agent_state: AgentNode.AgentState,
    ) -> AgentNode:
        """Create an agent node from agent state."""
        return AgentNode(
            node_id=node_id,
            name="agent",
            timestamped_states={datetime_to_ns(datetime.now()): agent_state},
        )

    @classmethod
    def _create_room_nodes(
        cls,
        rooms_metadata: list[dict[str, JsonValue]],
        start_node_id: int,
    ) -> list[RoomNode]:
        """Create room nodes from room metadata."""
        room_nodes: list[RoomNode] = []
        for i, room in enumerate(rooms_metadata):
            room_node = RoomNode.from_ai2thor(
                node_id=start_node_id + i,
                room_metadata=Ai2ThorRoomMetadata.model_validate(room),
            )
            room_nodes.append(room_node)
        return room_nodes

    @classmethod
    def _create_object_nodes(
        cls,
        object_metadata: list[dict[str, JsonValue]],
        start_node_id: int,
    ) -> list[ObjectNode]:
        """Create object nodes from object metadata."""
        object_nodes: list[ObjectNode] = []
        for i, obj in enumerate(object_metadata):
            obj_node = ObjectNode.from_ai2thor(
                node_id=start_node_id + i,
                object_metadata=Ai2ThorObjectMetadata.model_validate(obj),
            )
            object_nodes.append(obj_node)
        return object_nodes

    @classmethod
    def _create_receptacle_edges(
        cls,
        egg: Self,
    ) -> None:
        """Create receptacle edges for objects that are receptacles."""
        receptacle_nodes = egg.spatial.get_object_by_capabilities(
            capabilities=["is_receptacle"]
        )
        for rn in receptacle_nodes.values():
            _, rn_previous_state = rn.get_previous_timestamp_and_states()
            assert rn_previous_state, f"Cannot get previous state of {rn.name}"
            children_names = rn_previous_state.receptacle_object_ids
            if children_names:
                for child_name in children_names:
                    _, child_node = egg.spatial.get_object_node_by_name(
                        node_name=child_name
                    )
                    assert isinstance(child_node, ObjectNode)
                    edge = SpatialEdge(
                        edge_id=f"{rn.name}-{child_node.name}",
                        source_node_id=rn.node_id,
                        target_node_id=child_node.node_id,
                        relationship=SpatialEdge.SpatialRelationship.PARENT,
                    )
                    egg.spatial.add_spatial_edge(new_spatial_edge=edge)

    def set_spatial_components(self, spatial_components: SpatialComponents):
        """
        Updates the spatial component of EGG.

        :param spatial_components: New spatial components.
        :type spatial_components: SpatialComponents
        """
        self.spatial = spatial_components

    def set_event_components(self, event_components: EventComponents):
        """
        Updates the event component of EGG.

        :param event_components: New event configuration.
        :type event_components: EventComponents
        """
        self.events = event_components

    def get_spatial_components(self) -> SpatialComponents:
        """
        Retrieves a copy of the current spatial component.

        :returns: A copy of the spatial components.
        :rtype: SpatialComponents
        """
        return deepcopy(self.spatial)

    def get_event_components(self) -> EventComponents:
        """
        Retrieves a copy of the current event component.

        :returns: A copy of the event components.
        :rtype: EventComponents
        """
        return deepcopy(self.events)

    def serialize(self):
        """
        Serializes the entire EGG state including spatial components, event components,
        and event-object edges.

        :returns: Dictionary representation of the EGG's current state.
        :rtype: Dict
        """
        spatial_data = self.spatial.model_dump(mode="python")
        event_data = self.events.model_dump(mode="python")
        egg_data = {
            "nodes": {"spatial": spatial_data, "events": event_data},
        }
        return egg_data

    def pretty_str(self) -> str:
        """
        Generates a human-readable string representation of the spatial, event,
        and edge components within EGG.

        :returns: String representation of the current graph's state.
        :rtype: str
        """
        egg_str = ""
        egg_str += self.spatial.pretty_str()
        egg_str += self.events.pretty_str()
        return egg_str
