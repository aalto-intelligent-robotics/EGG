from typing import ClassVar
from pydantic import BaseModel, ConfigDict, Field
from ai2thor.controller import Controller
from ai2thor.server import MultiAgentEvent

from egg.graph.egg import EGG
from egg.graph.event import EventComponents
from egg.graph.node import ObjectNode
from egg.graph.spatial import SpatialComponents
from egg.utils.geometry import Position, Rotation


class EventSimulator(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )
    ai2thor_controller: Controller
    egg: EGG = Field(default=EGG(spatial=SpatialComponents(), events=EventComponents()))

    def get_picked_up_oject(self, object_name: str) -> ObjectNode:
        object_node = self.egg.spatial.get_object_node_by_name(object_name)
        assert isinstance(
            object_node, ObjectNode
        ), f"{object_node} does not exist in EGG"
        assert (
            object_node.is_pickupable and object_node.is_moveable
        ), f"Trying pick and place but {object_node} is not pickupable and/or moveable."
        return object_node

    def get_receptacle_object(self, receptacle_name: str) -> ObjectNode:
        receptacle_node = self.egg.spatial.get_object_node_by_name(receptacle_name)
        assert isinstance(
            receptacle_node, ObjectNode
        ), f"{receptacle_node} does not exist in EGG"
        assert receptacle_node.is_receptacle, f"{receptacle_name} is not a receptacle."
        return receptacle_node

    def get_agent_position(self) -> tuple[Position, Rotation]:
        event = self.ai2thor_controller.last_event
        assert isinstance(event, MultiAgentEvent)
        agent_metadata: dict[str, float] = event.metadata["agent"]
        return (
            Position.model_validate(agent_metadata["position"]),
            Rotation.model_validate(agent_metadata["rotation"]),
        )

    def get_reachable_positions(self) -> list[Position]:
        metadata = self.ai2thor_controller.step(action="GetReachablePositions").metadata
        reachable_positions: list[Position] = []
        for pos in metadata["actionReturn"]:
            reachable_positions.append(Position.model_validate(pos))
        return reachable_positions

    def bring_object_to_receptacle(self, object_name: str, receptacle_name: str):
        object_node = self.get_picked_up_oject(object_name=object_name)
        receptacle_node = self.get_receptacle_object(receptacle_name=receptacle_name)
