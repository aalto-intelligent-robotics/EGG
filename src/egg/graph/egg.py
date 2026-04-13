from copy import deepcopy
import logging
from typing import Self
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from egg.graph.event import EventComponents
from egg.graph.node import ObjectNode, RoomNode
from egg.graph.spatial import SpatialComponents
from egg.utils.data import Ai2ThorObjectMetadata, Ai2ThorRoomMetadata
from egg.utils.logger import getLogger

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

    spatial: SpatialComponents
    events: EventComponents
    use_gt_id: bool = Field(default=True, exclude=True)
    use_gt_caption: bool = Field(default=True, exclude=True)
    use_guided_auto_caption: bool = Field(default=True, exclude=True)
    device: str = Field(default="cuda:0", exclude=True)
    do_sample: bool = Field(default=False, exclude=True)
    node_id_gen: int = Field(default=0, ge=0)

    def is_empty(self) -> bool:
        return self.spatial.is_empty() and self.events.is_empty()

    def gen_id(self) -> int:
        self.node_id_gen += 1
        return self.node_id_gen

    @classmethod
    def from_ai2thor(
        cls,
        ai2thor_object_metadata: list[dict[str, JsonValue]],
        ai2thor_house_metadata: dict[str, JsonValue],
    ) -> Self:

        node_id_gen: int = 0
        list_adapter = TypeAdapter(list[dict[str, JsonValue]])

        egg = cls(spatial=SpatialComponents(), events=EventComponents())

        rooms_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
            ai2thor_house_metadata["rooms"]
        )
        object_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
            ai2thor_object_metadata
        )

        for room in rooms_metadata:
            room_node = RoomNode.from_ai2thor(
                node_id=node_id_gen,
                room_metadata=Ai2ThorRoomMetadata.model_validate(room),
            )
            egg.spatial.add_room_node(new_room_node=room_node)
            node_id_gen += 1

        for obj in object_metadata:
            obj_node = ObjectNode.from_ai2thor(
                node_id=node_id_gen,
                object_metadata=Ai2ThorObjectMetadata.model_validate(obj),
            )
            egg.spatial.add_object_node(new_object_node=obj_node)
            node_id_gen += 1

        egg.node_id_gen = node_id_gen
        return egg

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
        # edge_str = "\n🔗🔗🔗 EDGES 🔗🔗🔗\n"
        # for edge in self.event_edges:
        #     edge_str += edge.pretty_str()
        # egg_str += edge_str
        return egg_str
