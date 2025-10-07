from copy import deepcopy
import json
import logging
from typing import List, Dict, Tuple, Optional
import cv2
from numpy.typing import NDArray
import yaml
import numpy as np
import os

from egg.graph.node import EventNode, ObjectNode, RoomNode, datetime_to_ns
from egg.graph.spatial_graph import SpatialGraph
from egg.graph.event_graph import EventGraph
from egg.graph.edge import EventObjectEdge
from egg.utils.read_data import get_image_odometry_data
from egg.utils.camera import Camera
from egg.utils.image import (
    xy_to_binary_mask,
    get_instance_view,
    concatenate_images_vertically,
)
from egg.utils.logger import getLogger
from egg.utils.timestamp import ns_to_datetime, str_to_datetime
from egg.language.prompts.image_captioning_prompts import (
    build_image_captioning_messages,
)
from egg.language.llm import LLMAgent


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/egg.log",
)


class EGG:
    def __init__(
        self,
        spatial_graph: SpatialGraph,
        event_graph: EventGraph,
        use_gt_id: bool = True,
        use_gt_caption: bool = True,
        use_guided_auto_caption: bool = True,
        device: str = "cuda:0",
        do_sample: bool = False,
    ):
        self.spatial_graph: SpatialGraph = spatial_graph
        self.event_graph: EventGraph = event_graph
        self.event_object_edges: List[EventObjectEdge] = []
        self._entity_id: int = 0
        self.use_gt_id: bool = use_gt_id
        self.use_gt_caption: bool = use_gt_caption
        if not self.use_gt_caption:
            from egg.language.vlm import VLMAgent
            
            self.vlm_agent = VLMAgent(do_sample=do_sample, device=device)
            self.use_guided_auto_caption: bool = use_guided_auto_caption

    def set_spatial_graph(self, spatial_graph: SpatialGraph):
        self.spatial_graph = spatial_graph

    def set_event_graph(self, event_graph: EventGraph):
        self.event_graph = event_graph

    def set_event_object_edges(self, event_object_edges: List[EventObjectEdge]):
        self.event_object_edges = event_object_edges

    def get_spatial_graph(self) -> SpatialGraph:
        return deepcopy(self.spatial_graph)

    def get_event_graph(self) -> EventGraph:
        return deepcopy(self.event_graph)

    def get_event_object_edges(self) -> List[EventObjectEdge]:
        return deepcopy(self.event_object_edges)

    def get_objects(self) -> Dict[int, Dict[str, str]]:
        objects = {}
        for node in self.spatial_graph.get_all_object_nodes().values():
            objects.update(
                {node.node_id: {"name": node.name, "description": node.caption}}
            )
        return objects

    def get_events(self) -> Dict[int, str]:
        events = {}
        for node in self.event_graph.get_event_nodes().values():
            events.update(
                {
                    node.node_id: {
                        "start": str(ns_to_datetime(node.start)),
                        "description": node.event_description,
                    }
                }
            )
        return events

    def add_event_from_video(self, event_param_file: str, camera_config_file: str):
        # TODO: Somehow do tracking automatically
        # TODO: Match similar object nodes
        with open(event_param_file, "r") as event_param_fh:
            event_data = yaml.safe_load(event_param_fh)
        event_raw_data_path = event_data.get("image_path")
        event_dir = os.path.dirname(os.path.abspath(event_param_file))
        color_frame_file_template = (
            os.path.join(event_dir, event_raw_data_path)
            + "/color/color_frame_{frame_id}.png"
        )
        depth_frame_file_template = (
            os.path.join(event_dir, event_raw_data_path)
            + "/depth/depth_frame_{frame_id}.npy"
        )
        image_odometry_file = os.path.join(
            event_dir, event_data.get("image_odometry_file")
        )
        timestamped_observation_odom, frame_timestamp_map, start_ns, end_ns = (
            get_image_odometry_data(
                image_odometry_file=image_odometry_file,
                from_frame=event_data.get("from_frame"),
                to_frame=event_data.get("to_frame"),
            )
        )
        camera = Camera.from_yaml(yaml_file=camera_config_file)

        event_node_id = self.gen_id()

        if self.use_gt_caption:
            event_description = event_data.get("event_description")
            edge_captions = None
        else:
            event_description, edge_captions = (
                self.vlm_agent.generate_captions_from_yaml(
                    event_param_file, guided=self.use_guided_auto_caption
                )
            )

        object_nodes, event_object_edges, involved_object_ids = (
            self.get_object_nodes_and_edges_from_event(
                event_data=event_data,
                event_node_id=event_node_id,
                frame_timestamp_map=frame_timestamp_map,
                camera=camera,
                color_frame_file_template=color_frame_file_template,
                depth_frame_file_template=depth_frame_file_template,
                timestamped_observation_odom=timestamped_observation_odom,
                edge_captions=edge_captions,
            )
        )
        for new_object_node in object_nodes:
            self.spatial_graph.add_object_node(new_object_node)
        for edge in event_object_edges:
            self.event_object_edges.append(edge)

        self.event_graph.add_event_node(
            event_node=EventNode(
                node_id=event_node_id,
                start=start_ns,
                end=end_ns,
                event_description=event_description,
                timestamped_observation_odom=timestamped_observation_odom,
                involved_object_ids=involved_object_ids,
                location=event_data.get("location"),
            )
        )

    def get_object_cloud(
        self,
        camera: Camera,
        timestamp: int,
        timestamped_observation_odom: Dict[int, Dict[str, List]],
        depth_frame: NDArray,
        mask: NDArray,
    ) -> NDArray:
        camera_odom = timestamped_observation_odom[timestamp].get("camera_odom")
        assert (
            camera_odom is not None
        ), f"Camera odometry at timestamp {timestamp} is None"
        camera.set_T(
            position=camera_odom[0],
            orientation=camera_odom[1],
        )
        object_cloud = camera.depth_to_pointcloud(
            depth_image=depth_frame,
            mask=mask,
        )
        return object_cloud

    def get_first_and_last_object_clouds(
        self,
        first_timestamp: int,
        object_first_binary_mask: NDArray,
        last_timestamp: int,
        object_last_binary_mask: NDArray,
        depth_frame_file_template: str,
        camera: Camera,
        timestamped_observation_odom: Dict[int, Dict[str, List]],
        object_properties: Dict,
    ) -> Tuple[NDArray, NDArray]:
        # Add first frame
        first_depth_frame_file = depth_frame_file_template.format(
            frame_id=str(object_properties.get("first_frame")).zfill(4)
        )
        obj_first_cloud = self.get_object_cloud(
            camera=camera,
            timestamp=first_timestamp,
            timestamped_observation_odom=timestamped_observation_odom,
            depth_frame=np.load(first_depth_frame_file),
            mask=object_first_binary_mask,
        )
        # Add last frame
        last_depth_frame_file = depth_frame_file_template.format(
            frame_id=str(object_properties.get("last_frame")).zfill(4)
        )
        obj_last_cloud = self.get_object_cloud(
            camera=camera,
            timestamp=last_timestamp,
            timestamped_observation_odom=timestamped_observation_odom,
            depth_frame=np.load(last_depth_frame_file),
            mask=object_last_binary_mask,
        )
        return obj_first_cloud, obj_last_cloud

    def get_object_nodes_and_edges_from_event(
        self,
        event_data,
        event_node_id: int,
        frame_timestamp_map: Dict[int, int],
        camera: Camera,
        color_frame_file_template: str,
        depth_frame_file_template: str,
        timestamped_observation_odom: Dict[int, Dict[str, List]],
        edge_captions: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[ObjectNode], List[EventObjectEdge], List[int]]:
        new_object_nodes = []
        event_object_edges = []
        involved_object_ids = []
        for object_name, object_properties in event_data.get(
            "objects_of_interest"
        ).items():
            object_class = object_properties.get("object_class")
            obj_first_frame = object_properties.get("first_frame")
            first_timestamp = frame_timestamp_map[obj_first_frame]
            obj_last_frame = object_properties.get("last_frame")
            last_timestamp = frame_timestamp_map[obj_last_frame]
            obj_first_polygon_mask = object_properties.get("first_mask")
            assert isinstance(obj_first_polygon_mask, List)
            obj_first_binary_mask = xy_to_binary_mask(
                width=camera.width,
                height=camera.height,
                xy_polygon=obj_first_polygon_mask,
            )

            obj_first_color_frame = cv2.imread(
                color_frame_file_template.format(frame_id=str(obj_first_frame).zfill(4))
            )
            assert obj_first_color_frame is not None
            obj_first_instance_view = get_instance_view(
                map_view_img=obj_first_color_frame,
                mask=obj_first_binary_mask[:, :, np.newaxis],
            )

            obj_last_polygon_mask = object_properties.get("last_mask")
            assert isinstance(obj_last_polygon_mask, List)
            obj_last_binary_mask = xy_to_binary_mask(
                width=camera.width,
                height=camera.height,
                xy_polygon=obj_last_polygon_mask,
            )
            obj_last_color_frame = cv2.imread(
                color_frame_file_template.format(frame_id=str(obj_last_frame).zfill(4))
            )
            assert obj_last_color_frame is not None
            obj_last_instance_view = get_instance_view(
                map_view_img=obj_last_color_frame,
                mask=obj_last_binary_mask[:, :, np.newaxis],
                mask_bg=False,
            )

            obj_first_cloud, obj_last_cloud = self.get_first_and_last_object_clouds(
                first_timestamp=first_timestamp,
                object_first_binary_mask=obj_first_binary_mask,
                last_timestamp=last_timestamp,
                object_last_binary_mask=obj_last_binary_mask,
                camera=camera,
                depth_frame_file_template=depth_frame_file_template,
                timestamped_observation_odom=timestamped_observation_odom,
                object_properties=object_properties,
            )
            # TODO: Track all instances, for now only first and last seen
            object_node_id = self.gen_id()
            object_node = ObjectNode(
                node_id=object_node_id,
                object_class=object_class,
                name=object_name,
                timestamped_position={
                    first_timestamp: np.median(obj_first_cloud, axis=0),
                    last_timestamp: np.median(obj_last_cloud, axis=0),
                },
                instance_views=[obj_first_instance_view, obj_last_instance_view],
            )
            is_new_node, sim_node_id = self.spatial_graph.is_new_node(
                new_object_node=object_node, use_gt_id=self.use_gt_id
            )
            if is_new_node:
                new_object_nodes.append(object_node)
            else:
                self.spatial_graph.merge_object_nodes(
                    object_node_0_id=sim_node_id, object_node_1=object_node
                )
            involved_object_ids.append(sim_node_id)
            if self.use_gt_caption:
                object_role = object_properties.get("description")
            else:
                assert edge_captions is not None
                object_role = str(edge_captions.get(object_name))
            event_object_edges.append(
                EventObjectEdge(
                    edge_id=self.gen_id(),
                    source_node_id=event_node_id,
                    target_node_id=sim_node_id,
                    object_role=object_role,
                )
            )
        return new_object_nodes, event_object_edges, involved_object_ids

    def gen_id(self) -> int:
        self._entity_id += 1
        return self._entity_id

    def pretty_str(self) -> str:
        egg_str = ""
        egg_str += self.spatial_graph.pretty_str()
        egg_str += self.event_graph.pretty_str()
        edge_str = "\n🔗🔗🔗 EDGES 🔗🔗🔗\n"
        for edge in self.event_object_edges:
            edge_str += edge.pretty_str()
        egg_str += edge_str
        return egg_str

    def serialize_event_edges(self) -> Dict:
        event_edges_data = {}
        for edge in self.event_object_edges:
            edge_attr_data = {
                "edge_id": edge.edge_id,
                "from_event": edge.source_node_id,
                "to_object": edge.target_node_id,
                "object_role": edge.object_role,
            }
            event_edges_data.update({edge.edge_id: edge_attr_data})

        return event_edges_data

    def serialize(self) -> Dict:
        spatial_data = self.spatial_graph.serialize()
        event_data = self.event_graph.serialize()
        event_object_edges_data = self.serialize_event_edges()
        egg_data = {
            "nodes": {"object_nodes": spatial_data, "event_nodes": event_data},
            "edges": {"event_object_edges": event_object_edges_data},
        }
        return egg_data

    def deserialize(self, json_file: str):
        with open(json_file, "r") as f:
            egg_data = json.load(f)
        nodes_data = egg_data["nodes"]
        spatial_data = nodes_data["object_nodes"]
        for object_id, object_properties in spatial_data.items():
            object_attr = object_properties["attributes"]
            timestamped_position = {}
            for datetime_str, pos in object_attr["timestamped_position"].items():
                timestamped_position.update(
                    {datetime_to_ns(str_to_datetime(datetime_str)): np.array(pos)}
                )
            object_node = ObjectNode(
                node_id=int(object_id),
                name=object_attr["name"],
                timestamped_position=timestamped_position,
                object_class=object_attr["object_class"],
                caption=object_attr["caption"],
            )
            self.spatial_graph.add_object_node(object_node)
        event_data = nodes_data["event_nodes"]
        for event_id, event_attrs in event_data.items():
            timestamped_observation_odom = {}
            for timestamp, odom in event_attrs["timestamped_observation_odom"].items():
                timestamped_observation_odom.update({datetime_to_ns(str_to_datetime(timestamp)): odom})
            event_node = EventNode(
                node_id=int(event_id),
                event_description=event_attrs["event_description"],
                start=datetime_to_ns(str_to_datetime(event_attrs["start"])),
                end=datetime_to_ns(str_to_datetime(event_attrs["end"])),
                involved_object_ids=event_attrs["involved_object_ids"],
                timestamped_observation_odom=timestamped_observation_odom,
                location=event_attrs["location"],
            )
            self.event_graph.add_event_node(event_node)
        edge_data = egg_data["edges"]["event_object_edges"]
        for edge_id, edge_attrs in edge_data.items():
            self.event_object_edges.append(
                EventObjectEdge(
                    edge_id=int(edge_id),
                    source_node_id=int(edge_attrs["from_event"]),
                    target_node_id=int(edge_attrs["to_object"]),
                    object_role=str(edge_attrs["object_role"]),
                )
            )

    def gen_object_captions(self, llm_agent: LLMAgent):
        for obj_node_id in self.spatial_graph.get_all_object_nodes().keys():
            obj_node = self.spatial_graph.get_object_node_by_id(obj_node_id)
            assert obj_node is not None
            obj_views_image = concatenate_images_vertically(
                images=obj_node.instance_views
            )
            image_captioning_messages = build_image_captioning_messages(
                image=obj_views_image, object_class=obj_node.object_class
            )
            obj_caption = llm_agent.send_query(image_captioning_messages)
            assert obj_caption is not None
            obj_node.caption = obj_caption

    def gen_room_nodes(self):
        room_pos: Dict[str, List[NDArray]] = {}
        
        for event in self.get_event_graph().get_event_nodes().values():
            room_name = event.location
            assert isinstance(room_name, str)
            if room_name not in room_pos.keys():
                room_pos[room_name] = [event.get_first_observation_pos()]
            else:
                room_pos[room_name].append(event.get_first_observation_pos())

        for room_name, room_pos_list in room_pos.items():
            self.spatial_graph.add_room_node(
                new_room_node=RoomNode(
                    node_id=self.gen_id(),
                    name=room_name,
                    position=np.mean(room_pos_list, axis=0),
                )
            )
