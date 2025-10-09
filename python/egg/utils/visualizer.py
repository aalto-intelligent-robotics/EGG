from dataclasses import dataclass
import re
from typing import Optional, List, Tuple
import tf.transformations as tr
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import logging

from egg.graph.egg import EGG
from egg.graph.node import ObjectNode, EventNode
from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/visualizer.log",
)

PCD_COLOR = [0.2, 0.5, 0.7]
EVENT_COLOR = [0.2, 0.2, 0.6]
EVENT_NODE_DIM = 0.2
OBJECT_COLOR = [0.8, 0.3, 0.5]
INACTIVE_OBJECT_COLOR = [0.3, 0.0, 0.0]
OBJECT_NODE_DIM = 0.05
EDGE_COLOR = [0.2, 0.2, 0.2]
BASE_CAM_COLOR = [0.2, 0.2, 0.2]
ROOM_COLOR = [0.7, 0.2, 0.6]


@dataclass
class VizElement:
    name: str
    geometry: o3d.geometry


class EGGVisualizer:
    def __init__(
        self,
        egg: EGG,
        pcd_path: Optional[str] = None,
        panel_height: int = 120,
        window_size: Tuple[int, int] = (1024, 768),
        title: str = "EGG Viewer",
        pcd_z_filter: float = 2.0,
    ):
        self.egg = egg
        self.event_ids = self.egg.get_event_components().get_event_ids()
        self.room_offset = 3
        self.building_offset = 4

        self.slider_values = list(range(0, self.egg.get_event_components().get_num_events()))

        self.pcd_path = pcd_path
        self.panel_height = panel_height
        self.window_size = window_size
        self.title = title

        # Open3D GUI components
        self.app = gui.Application.instance
        self.window = None
        self.scene_widget = None
        self.panel = None
        self.slider = None
        self.label = None

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self.pcd = None
        self.pcd_z_filter = pcd_z_filter

    def load_and_filter_pcd(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Load point cloud and remove points with z >= pcd_z_filter.

        Preserves colors if present.
        """
        if not self.pcd_path:
            return None

        pcd = o3d.io.read_point_cloud(self.pcd_path)
        if len(pcd.points) == 0:
            return pcd

        pts = np.asarray(pcd.points)
        mask = pts[:, 2] < self.pcd_z_filter
        filtered_pts = pts[mask]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_pts)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            filtered_colors = colors[mask]
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        else:
            filtered_pcd.paint_uniform_color([0.2, 0.5, 0.7])

        return filtered_pcd

    def _on_slider_value_changed(self, v):
        self.update_event(self.event_ids[int(round(v))])

    def _on_layout(self, ctx):
        r = self.window.content_rect
        h = self.panel_height
        # Bottom panel
        self.panel.frame = gui.Rect(r.x, r.y + r.height - h, r.width, h)
        # Scene above the panel
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width, r.height - h)

    def setup_ui(self):
        # Initialize Open3D app and window
        self.app.initialize()
        self.window = self.app.create_window(self.title, *self.window_size)

        # Scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        # Bottom panel: slider on top, label underneath
        self.panel = gui.Vert(8, gui.Margins(8, 8, 8, 8))
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(0, len(self.slider_values) - 1))
        self.slider.set_on_value_changed(self._on_slider_value_changed)

        self.label = gui.Label("")  # set after first update

        self.panel.add_child(self.slider)
        self.panel.add_child(self.label)
        self.window.add_child(self.panel)

        # Load and filter PCD if path provided
        self.pcd = self.load_and_filter_pcd()

        # Initial geometry
        self.update_event(self.event_ids[0])
        bounds = self.pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        self.scene_widget.setup_camera(60.0, bounds, center)

        # Layout
        self.window.set_on_layout(self._on_layout)

    def run(self):
        self.setup_ui()
        self.app.run()

    def update_event(self, event_id: int):
        event_viz = self.draw_event_node(event_id)
        event_viz += self.draw_room_nodes()
        event_node = self.egg.get_event_components().get_event_node_by_id(event_id)
        assert event_node is not None
        self.scene_widget.scene.clear_geometry()
        if self.pcd is not None:
            self.scene_widget.scene.add_geometry("Scene Cloud", self.pcd, self.material)
        for element in event_viz:
            self.scene_widget.scene.add_geometry(
                element.name, element.geometry, self.material
            )
        self.label.text = event_node.pretty_str()

    def draw_room_nodes(self) -> List[VizElement]:
        rooms_viz = []
        room_nodes = self.egg.spatial.get_all_room_nodes()
        room_pos = []
        for node in room_nodes.values():
            room_pos.append(node.position)
            node_viz_position = node.position
            node_viz_position[2] = self.room_offset
            rooms_viz += [
                VizElement(
                    name=f"{node.name} node",
                    geometry=self.draw_cube(center=node_viz_position),
                ),
                VizElement(
                    name=f"{node.name} label",
                    geometry=self.draw_text_mesh(
                        text=node.name, position=node_viz_position
                    ),
                ),
            ]
        building_node_pos_viz = np.mean(room_pos, axis=0)
        building_node_pos_viz[2] = self.building_offset
        rooms_viz += [
            VizElement(
                name=f"Building node",
                geometry=self.draw_cube(center=building_node_pos_viz),
            ),
            VizElement(
                name=f"Building label",
                geometry=self.draw_text_mesh(
                    text="Building", position=building_node_pos_viz
                ),
            ),
        ]
        for room_node in room_nodes.values():
            room_node_viz_position = room_node.position
            room_node_viz_position[2] = self.room_offset
            rooms_viz.append(
                VizElement(
                    name=f"Building - {room_node.name} edge",
                    geometry=self.draw_line(
                        source=building_node_pos_viz, target=room_node_viz_position
                    ),
                )
            )
        return rooms_viz

    def draw_involved_objects(
        self,
        event_node: EventNode,
        start_timestamp: int,
        end_timestamp: int,
        room_node_viz_pos: NDArray,
        cam_pos: NDArray,
    ) -> List[VizElement]:
        obj_viz = []
        for obj_node_id in event_node.involved_object_ids:
            obj_node = self.egg.spatial.get_object_node_by_id(obj_node_id)
            assert obj_node is not None
            _, obj_end = obj_node.get_closest_start_end_timestamps(
                start_timestamp, end_timestamp
            )
            obj_viz += [
                VizElement(
                    name=f"Object {obj_node_id} End Node",
                    geometry=self.draw_sphere(
                        center=obj_node.timestamped_position[obj_end],
                        dim=OBJECT_NODE_DIM,
                        color=OBJECT_COLOR,
                    ),
                ),
                VizElement(
                    name=f"Object {obj_node_id} End Node label",
                    geometry=self.draw_text_mesh(
                        text=f"{obj_node.name}",
                        position=obj_node.timestamped_position[obj_end],
                    ),
                ),
                VizElement(
                    name=f"Object {obj_node_id} - Room Edge",
                    geometry=self.draw_line(
                        source=obj_node.timestamped_position[obj_end],
                        target=room_node_viz_pos,
                    ),
                ),
                VizElement(
                    name=f"Event {event_node.node_id} - Object {obj_node_id} edge",
                    geometry=self.draw_line(
                        source=obj_node.timestamped_position[obj_end],
                        target=cam_pos,
                    ),
                ),
            ]
        return obj_viz

    def draw_non_involved_objects(self, event_node: EventNode) -> List[VizElement]:
        obj_viz = []
        non_involved_ids = [
            id
            for id in list(self.egg.spatial.get_all_object_nodes().keys())
            if id not in event_node.involved_object_ids
        ]
        for obj_node_id in non_involved_ids:
            obj_node = self.egg.spatial.get_object_node_by_id(obj_node_id)
            assert obj_node is not None
            if obj_node.has_been_seen(event_node.start):
                prev_timestamp, prev_pos = obj_node.get_previous_timestamp_and_position(
                    event_node.start
                )
                assert (
                    prev_pos is not None and prev_timestamp is not None
                ), f"Node {obj_node.name} failed, timestamps {list(obj_node.timestamped_position.keys())} ref_timestamp {event_node.start}"
                prev_event_node = self.egg.events.get_event_node_by_timestamp(
                    prev_timestamp
                )
                assert prev_event_node is not None
                prev_room_node = self.egg.spatial.get_room_node_by_name(
                    prev_event_node.location
                )
                assert prev_room_node is not None
                prev_room_pos_viz = prev_room_node.position
                prev_room_pos_viz[2] = self.room_offset
                obj_viz += [
                    VizElement(
                        name=f"Object {obj_node_id} Prev Node",
                        geometry=self.draw_sphere(
                            center=prev_pos,
                            dim=OBJECT_NODE_DIM,
                            color=INACTIVE_OBJECT_COLOR,
                        ),
                    ),
                    VizElement(
                        name=f"Object {obj_node_id} Prev Node Label",
                        geometry=self.draw_text_mesh(
                            text=f"{obj_node.name}",
                            position=prev_pos,
                        ),
                    ),
                    VizElement(
                        name=f"Object {obj_node_id} - {prev_room_node.name} Edge",
                        geometry=self.draw_line(
                            source=prev_pos,
                            target=prev_room_pos_viz,
                        ),
                    ),
                ]
        return obj_viz

    def draw_event_node(self, event_id: int) -> List[VizElement]:
        event_viz = []
        event_node = self.egg.events.get_event_node_by_id(event_id)
        assert isinstance(event_node, EventNode)
        start_timestamp = event_node.start
        end_timestamp = event_node.end
        obs_odom = event_node.get_first_observation_odom()
        cam_pos = np.array(obs_odom["camera_odom"][0])
        cam_orientation = tr.quaternion_matrix(
            quaternion=np.array(obs_odom["camera_odom"][1])
        )[:3, :3]

        base_pos = np.array(obs_odom["base_odom"][0])
        base_orientation = tr.quaternion_matrix(
            quaternion=np.array(obs_odom["base_odom"][1])
        )[:3, :3]

        room_node = self.egg.spatial.get_room_node_by_name(event_node.location)
        assert room_node is not None
        room_node_viz_pos = room_node.position
        room_node_viz_pos[2] = self.room_offset
        obj_viz = self.draw_involved_objects(
            event_node=event_node,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            room_node_viz_pos=room_node_viz_pos,
            cam_pos=cam_pos,
        )
        obj_viz += self.draw_non_involved_objects(event_node=event_node)
        event_viz += obj_viz
        event_viz += [
            VizElement(
                name=f"Camera Frame Event {event_id}",
                geometry=self.draw_frame(pos=cam_pos, orientation=cam_orientation),
            ),
            VizElement(
                name=f"Base Frame Event {event_id}",
                geometry=self.draw_frame(pos=base_pos, orientation=base_orientation),
            ),
            VizElement(
                name=f"Base_Cam Event {event_id}",
                geometry=self.draw_line(
                    source=base_pos, target=cam_pos, line_color=BASE_CAM_COLOR
                ),
            ),
            VizElement(
                name=f"Event Node {event_id}",
                geometry=self.draw_cube(center=cam_pos, color=EVENT_COLOR),
            ),
            VizElement(
                name=f"Event Node {event_id} label",
                geometry=self.draw_text_mesh(
                    text=f"Event {event_id}", position=cam_pos
                ),
            ),
            VizElement(
                name=f"Event {event_id} - {room_node.name} Edge",
                geometry=self.draw_line(
                    source=cam_pos,
                    target=room_node_viz_pos,
                ),
            ),
        ]
        return event_viz

    def draw_cube(
        self, center: NDArray, color: List = [0.5, 0.5, 0.5], dim: float = 0.1
    ) -> o3d.geometry.TriangleMesh:
        """Draw a cube at a given center with specified dimensions and color.

        :param center: The center position of the cube.
        :param color: The color of the cube. Defaults to [0.5, 0.5,
            0.5].
        :param dim: The dimension of the cube. Defaults to 0.1.
        :return: An o3d.geometry.TriangleMesh object representing the
            cube.
        """
        # Create the box mesh
        cube = o3d.geometry.TriangleMesh.create_box(dim, dim, dim)
        # Translate the box to the specified center
        translation = center - np.array([dim / 2, dim / 2, dim / 2])
        cube.translate(translation)
        cube.paint_uniform_color(color)
        return cube

    def draw_sphere(
        self, center: np.ndarray, color: List = [0.5, 0.5, 0.5], dim: float = 0.1
    ) -> o3d.geometry.TriangleMesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=dim, resolution=20)
        # Translate the sphere to the specified center
        translation = center - np.array([dim / 2, dim / 2, dim / 2])
        sphere.translate(translation)
        sphere.paint_uniform_color(color)
        return sphere

    def draw_line(self, source: NDArray, target: NDArray, line_color: List = [0, 0, 0]):
        points = [source, target]
        lines = [[0, 1]]
        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        lineset.paint_uniform_color(line_color)
        return lineset

    def draw_frame(self, pos=np.zeros([3, 1]), orientation=np.eye(3), size=0.2):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coordinate_frame.translate(pos)
        coordinate_frame.rotate(orientation, center=pos)
        return coordinate_frame

    def draw_text_mesh(
        self, text: str, position: np.ndarray
    ) -> o3d.geometry.TriangleMesh:
        """Create and draw a 3D text mesh at a specified position.

        :param text: The text to display in the mesh.
        :param position: The position where the text should be placed.
        :return: An o3d.geometry.TriangleMesh object representing the
            text.
        """
        # Create a 3D text mesh
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.5)
        text_mesh = text_mesh.to_legacy()
        text_mesh.scale(0.005, center=[0.0, 0.0, 0.0])
        text_mesh.rotate(
            text_mesh.get_rotation_matrix_from_xyz((np.pi / 3, 0, 0)), center=(0, 0, 0)
        )
        text_mesh.translate(position + np.array([0, 0, 0.1]))
        text_mesh.paint_uniform_color([0.0, 0.0, 0.0])
        return text_mesh
