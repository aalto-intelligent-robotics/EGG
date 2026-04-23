import json
from egg.graph.event import EventComponents
from egg.graph.node import ObjectNode, RoomNode
from egg.graph.spatial import SpatialComponents
from pydantic import TypeAdapter, JsonValue

from egg.graph.egg import EGG

adapter = TypeAdapter(dict[str, JsonValue])
list_adapter = TypeAdapter(list[dict[str, JsonValue]])

with open("./agent.json", "r", encoding="utf-8") as f_agent:
    agent_metadata: dict[str, JsonValue] = adapter.validate_python(json.load(f_agent))

with open("./floor_plan.json", "r", encoding="utf-8") as f_floor:
    floor_metadata: dict[str, JsonValue] = adapter.validate_python(json.load(f_floor))

with open("./obj.json", "r", encoding="utf-8") as f_obj:
    obj_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
        json.load(f_obj)
    )

rooms_metadata: list[dict[str, JsonValue]] = list_adapter.validate_python(
    floor_metadata["rooms"]
)

egg = EGG.from_ai2thor(
    ai2thor_agent_metadata=agent_metadata,
    ai2thor_house_metadata=floor_metadata,
    ai2thor_object_metadata=obj_metadata,
    object_types_config_file="../configs/ai2-thor/object_type_config.toml",
)
pickupables = [
    obj
    for obj in egg.spatial.get_object_by_capabilities(
        capabilities=["is_pickupable"]
    ).values()
]
receptacles = [
    obj
    for obj in egg.spatial.get_object_by_capabilities(
        capabilities=["is_receptacle"]
    ).values()
]


def get_room_containing_object(object: ObjectNode, rooms: dict[int, RoomNode]):
    _, object_prev_state = object.get_previous_timestamp_and_states()
    assert object_prev_state
    for room in rooms.values():
        if room.is_inside_room(object_prev_state.position):
            return room


print(f"Pickupables: {[obj.name for obj in pickupables]}\n")
for p in pickupables:
    p_str = f"- {p.name}:"
    _, prev_state = p.get_previous_timestamp_and_states()
    assert prev_state
    parents = prev_state.parent_receptacles
    if parents:
        p_str += f"\n\t- Initial parent receptacles: {parents}"
    p_str += f"\n\t- Capabilities: {p.capabilities_str()}"
    compatible_receptacles = egg.get_compatible_receptacles(object_name=p.name)
    p_str += f"\n\t- Compatible receptacles: {[cr.name for cr in compatible_receptacles]}"
    print(p_str)

print(f"All receptacles: {[obj.name for obj in receptacles]}\n")
for r in receptacles:
    r_str = f"- {r.name}:"
    _, prev_state = r.get_previous_timestamp_and_states()
    assert prev_state
    r_str += f"\n\t- Initial position: {prev_state.position}"
    r_str += f"\n\t- Capabilities: {r.capabilities_str()}"
    parents = prev_state.parent_receptacles
    if parents:
        r_str += f"\n\t- Initial parent receptacles: {parents}"
    r_str += f"\n\t- Room: {get_room_containing_object(object=r, rooms=egg.spatial.room_nodes).name}"
    print(r_str)


with open("graph_ai2thor.json", "w") as f:
    json.dump(egg.model_dump(mode="python"), f)
