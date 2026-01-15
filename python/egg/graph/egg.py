from copy import deepcopy
import json
import logging
from typing import List, Dict, Tuple, Optional
import cv2
from numpy.typing import NDArray
import yaml
import numpy as np
import os

from egg.graph.node import EventNode, ObjectNode, RoomNode
from egg.graph.spatial import SpatialComponents
from egg.graph.event import EventComponents
from egg.graph.edge import EventObjectEdge
from egg.utils.read_data import get_image_odometry_data
from egg.utils.camera import Camera
from egg.utils.image import (
    xy_to_binary_mask,
    get_instance_view,
    concatenate_images_vertically,
)
from egg.utils.logger import getLogger
from egg.utils.timestamp import ns_to_datetime, str_to_datetime, datetime_to_ns
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
    """
    EGG (Event-Grounding Graph) framework that grounds events semantic context to spatial geometrics.
    """

    def __init__(
        self,
        spatial: SpatialComponents,
        events: EventComponents,
        use_gt_id: bool = True,
        use_gt_caption: bool = True,
        use_guided_auto_caption: bool = True,
        device: str = "cuda:0",
        do_sample: bool = False,
    ):
        """
        Initializes the EGG framework with specified spatial and event components
        and configuration options for IDs and captions.

        :param spatial: The spatial components.
        :type spatial: SpatialComponents
        :param events: The event component configuration.
        :type events: EventComponents
        :param use_gt_id: Whether to use ground truth IDs for spatial nodes.
        :type use_gt_id: bool
        :param use_gt_caption: Whether to use ground truth captions for images.
        :type use_gt_caption: bool
        :param use_guided_auto_caption: Whether to guided automatic captioning for videos.
        :type use_guided_auto_caption: bool
        :param device: Device for computation requiring hardware acceleration.
        :type device: str
        :param do_sample: Whether sampling is used in GPT4o for image caption generation.
        :type do_sample: bool
        """
        self.spatial: SpatialComponents = spatial
        self.events: EventComponents = events
        self.event_edges: List[EventObjectEdge] = []
        self._entity_id: int = 0
        self.use_gt_id: bool = use_gt_id
        self.use_gt_caption: bool = use_gt_caption
        if not self.use_gt_caption:
            from egg.language.vlm import VLMAgent

            self.vlm_agent = VLMAgent(do_sample=do_sample, device=device)
            self.use_guided_auto_caption: bool = use_guided_auto_caption

    def is_empty(self) -> bool:
        return (self.spatial.is_empty() and self.events.is_empty())

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

    def set_event_edges(self, event_edges: List[EventObjectEdge]):
        """
        Updates the list of edges between events and objects in EGG.

        :param event_edges: List associating events with objects.
        :type event_edges: List[EventObjectEdge]
        """
        self.event_edges = event_edges

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

    def get_event_edges(self) -> List[EventObjectEdge]:
        """
        Retrieves a copy of the current list of event-object edges.

        :returns: A copy of the event-object edges list.
        :rtype: List[EventObjectEdge]
        """
        return deepcopy(self.event_edges)

    def get_objects(self) -> Dict[int, Dict[str, str]]:
        """
        Retrieves object details indexed by their node IDs.

        :returns: Dictionary mapping node IDs to object details.
        :rtype: Dict[int, Dict[str, str]]
        """
        objects = {}
        for node in self.spatial.get_all_object_nodes().values():
            objects.update(
                {node.node_id: {"name": node.name, "description": node.caption}}
            )
        return objects

    def get_events(self) -> Dict[int, str]:
        """
        Retrieves event details indexed by their node IDs.

        :returns: Dictionary mapping node IDs to event descriptions.
        :rtype: Dict[int, str]
        """
        events = {}
        for node in self.events.get_event_nodes().values():
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
        """
        Integrates event data from a video into EGG by parsing input files and extracting
        relevant event and object information.

        :param event_param_file: Path to the YAML file with event parameters.
        :type event_param_file: str
        :param camera_config_file: Path to the camera configuration file in YAML format.
        :type camera_config_file: str
        """
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
            self.spatial.add_object_node(new_object_node)
        for edge in event_object_edges:
            self.event_edges.append(edge)

        self.events.add_event_node(
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
        """
        Converts depth frame data into a point cloud for an object at a specific timestamp.

        :param camera: Camera object for transforming depth frame into spatial data.
        :type camera: Camera
        :param timestamp: Timestamp indicating when the depth data was captured.
        :type timestamp: int
        :param timestamped_observation_odom: Odometry data indexed by timestamps for object localization.
        :type timestamped_observation_odom: Dict[int, Dict[str, List]]
        :param depth_frame: Depth frame image as an array representing scene depth data.
        :type depth_frame: NDArray
        :param mask: Pixel mask specifying the region corresponding to the object.
        :type mask: NDArray
        :returns: A point cloud array representing the object.
        :rtype: NDArray
        """
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
        """
        Generates point clouds for an object's first and last frames.

        :param first_timestamp: Timestamp of the object's initial visible frame.
        :type first_timestamp: int
        :param object_first_binary_mask: Binary mask of the object in the first frame.
        :type object_first_binary_mask: NDArray
        :param last_timestamp: Timestamp of the object's last visible frame.
        :type last_timestamp: int
        :param object_last_binary_mask: Binary mask of the object in the last frame.
        :type object_last_binary_mask: NDArray
        :param depth_frame_file_template: File path template to locate depth frames.
        :type depth_frame_file_template: str
        :param camera: Camera object used for depth-to-point cloud conversion.
        :type camera: Camera
        :param timestamped_observation_odom: Odometry data indexed by timestamps for object localization.
        :type timestamped_observation_odom: Dict[int, Dict[str, List]]
        :param object_properties: Properties and metadata associated with the object.
        :type object_properties: Dict
        :returns: Tuple containing the point clouds for the object's first and last frames.
        :rtype: Tuple[NDArray, NDArray]
        """
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
        """
        Extracts object nodes and event edges involved in an event from the given event data.

        :param event_data: Data containing details about the event and associated objects.
        :type event_data: dict
        :param event_node_id: Unique ID for the event node.
        :type event_node_id: int
        :param frame_timestamp_map: Mapping from frame numbers to timestamps.
        :type frame_timestamp_map: Dict[int, int]
        :param camera: Camera object for image processing and point cloud generation.
        :type camera: Camera
        :param color_frame_file_template: Template for locating color frame image files.
        :type color_frame_file_template: str
        :param depth_frame_file_template: Template for locating depth frame image files.
        :type depth_frame_file_template: str
        :param timestamped_observation_odom: Odometry data indexed by timestamps for object localization.
        :type timestamped_observation_odom: Dict[int, Dict[str, List]]
        :param edge_captions: [Optional] Captions for edges between event and object nodes.
        :type edge_captions: Optional[Dict[str, str]]
        :returns: A tuple containing lists of new object nodes, event-object edges, and involved object IDs.
        :rtype: Tuple[List[ObjectNode], List[EventObjectEdge], List[int]]
        """
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
            is_new_node, sim_node_id = self.spatial.is_new_node(
                new_object_node=object_node, use_gt_id=self.use_gt_id
            )
            if is_new_node:
                new_object_nodes.append(object_node)
            else:
                self.spatial.merge_object_nodes(
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
        """
        Generates a unique identifier for nodes within EGG.

        :returns: A new unique identifier for an entity.
        :rtype: int
        """
        self._entity_id += 1
        return self._entity_id

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
        edge_str = "\n🔗🔗🔗 EDGES 🔗🔗🔗\n"
        for edge in self.event_edges:
            edge_str += edge.pretty_str()
        egg_str += edge_str
        return egg_str

    def serialize_event_edges(self) -> Dict:
        """
        Serializes the event-object edges within EGG into a dictionary form.

        :returns: Dictionary representation of event-object edges.
        :rtype: Dict
        """
        event_edges_data = {}
        for edge in self.event_edges:
            edge_attr_data = {
                "edge_id": edge.edge_id,
                "from_event": edge.source_node_id,
                "to_object": edge.target_node_id,
                "object_role": edge.object_role,
            }
            event_edges_data.update({edge.edge_id: edge_attr_data})

        return event_edges_data

    def serialize(self) -> Dict:
        """
        Serializes the entire EGG state including spatial components, event components,
        and event-object edges.

        :returns: Dictionary representation of the EGG's current state.
        :rtype: Dict
        """
        spatial_data = self.spatial.serialize()
        event_data = self.events.serialize()
        event_object_edges_data = self.serialize_event_edges()
        egg_data = {
            "nodes": {"object_nodes": spatial_data, "event_nodes": event_data},
            "edges": {"event_object_edges": event_object_edges_data},
        }
        return egg_data

    def deserialize(self, json_file: str):
        """
        Reconstructs the spatial and event components, along with edges, from a JSON file
        representation into the EGG.

        :param json_file: Path to the JSON file containing the serialized EGG data.
        :type json_file: str
        """
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
            self.spatial.add_object_node(object_node)
        event_data = nodes_data["event_nodes"]
        for event_id, event_attrs in event_data.items():
            timestamped_observation_odom = {}
            for timestamp, odom in event_attrs["timestamped_observation_odom"].items():
                timestamped_observation_odom.update(
                    {datetime_to_ns(str_to_datetime(timestamp)): odom}
                )
            event_node = EventNode(
                node_id=int(event_id),
                event_description=event_attrs["event_description"],
                start=datetime_to_ns(str_to_datetime(event_attrs["start"])),
                end=datetime_to_ns(str_to_datetime(event_attrs["end"])),
                involved_object_ids=event_attrs["involved_object_ids"],
                timestamped_observation_odom=timestamped_observation_odom,
                location=event_attrs["location"],
            )
            self.events.add_event_node(event_node)
        edge_data = egg_data["edges"]["event_object_edges"]
        for edge_id, edge_attrs in edge_data.items():
            self.event_edges.append(
                EventObjectEdge(
                    edge_id=int(edge_id),
                    source_node_id=int(edge_attrs["from_event"]),
                    target_node_id=int(edge_attrs["to_object"]),
                    object_role=str(edge_attrs["object_role"]),
                )
            )

    def gen_object_captions(self, llm_agent: LLMAgent):
        """
        Generates captions for objects within EGG using the provided language model agent.

        :param llm_agent: Language model agent used for generating image captions.
        :type llm_agent: LLMAgent
        """
        for obj_node_id in self.spatial.get_all_object_nodes().keys():
            obj_node = self.spatial.get_object_node_by_id(obj_node_id)
            assert obj_node is not None
            obj_views_image = concatenate_images_vertically(
                images=obj_node.instance_views
            )
            image_captioning_messages = build_image_captioning_messages(
                image=obj_views_image, object_class=obj_node.object_class
            )
            obj_caption, _, _ = llm_agent.query(image_captioning_messages)
            assert obj_caption is not None
            obj_node.caption = obj_caption

    def gen_room_nodes(self):
        """
        Generates room nodes in the spatial graph based on event location data
        and object observations.

        This method aggregates object positions within event-defined locations,
        creates room nodes in the graph with computed average positions.
        """
        room_pos: Dict[str, List[NDArray]] = {}

        for event in self.get_event_components().get_event_nodes().values():
            room_name = event.location
            assert isinstance(room_name, str)
            if room_name not in room_pos.keys():
                room_pos[room_name] = [event.get_first_observation_pos()]
            else:
                room_pos[room_name].append(event.get_first_observation_pos())

        for room_name, room_pos_list in room_pos.items():
            self.spatial.add_room_node(
                new_room_node=RoomNode(
                    node_id=self.gen_id(),
                    name=room_name,
                    position=np.mean(room_pos_list, axis=0),
                )
            )
