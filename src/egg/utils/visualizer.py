import json
import logging
from typing import ClassVar
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np
from numpy.typing import NDArray

from egg.utils.geometry import AxisAlignedBoundingBox
from egg.utils.logger import getLogger
from egg.graph.egg import EGG
from egg.graph.node import ObjectNode, EventNode

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/visualizer.log",
)


class EGGVizConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    pcd_color: tuple[float, float, float] = Field(default=(0.2, 0.5, 0.7), frozen=True)
    event_color: tuple[float, float, float] = Field(
        default=(0.2, 0.2, 0.6), frozen=True
    )
    object_color: tuple[float, float, float] = Field(
        default=(0.8, 0.3, 0.5), frozen=True
    )
    inactive_object_color: tuple[float, float, float] = Field(
        default=(0.3, 0.0, 0.0), frozen=True
    )
    edge_color: tuple[float, float, float] = Field(default=(0.2, 0.2, 0.2), frozen=True)
    base_cam_color: tuple[float, float, float] = Field(
        default=(0.2, 0.2, 0.2), frozen=True
    )
    room_color: tuple[float, float, float] = Field(default=(0.7, 0.2, 0.6), frozen=True)

    event_node_dim: float = Field(default=0.2, ge=0, frozen=True)
    spatial_node_dim: float = Field(default=0.05, ge=0, frozen=True)
    room_offset: float = Field(default=3, ge=0, frozen=True)
    building_offset: float = Field(default=4, ge=0, frozen=True)
    panel_height: int = Field(default=120, ge=0, frozen=True)
    window_size: tuple[int, int] = Field(default=(1024, 768), frozen=True)
    ignore_classes: list[str] = Field(default_factory=list, frozen=True)

    # Validators
    @field_validator(
        "pcd_color",
        "event_color",
        "object_color",
        "inactive_object_color",
        "edge_color",
        "base_cam_color",
        "room_color",
    )
    @classmethod
    def _validate_color(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for x in v:
            if not (0.0 <= x <= 1.0):
                raise ValueError("each RGB component must be between 0.0 and 1.0")
        return v

    @field_validator("window_size")
    @classmethod
    def _validate_color(cls, v: tuple[int, int]) -> tuple[int, int]:
        for x in v:
            if not (x > 0):
                raise ValueError("Window dimensions need to be positive")
        return v

    @classmethod
    def from_json(cls, path: str) -> "EGGVizConfig":
        with open(path, "r") as f:
            viz_config = cls.model_validate(json.load(f))
        return viz_config


class VizElement(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str
    geometry: o3d.geometry.Geometry


class EGGViz(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )
    egg: EGG

    # event_ids: list[int] = self.egg.get_event_components().get_event_ids()
    # slider_values: list[int] = list(
    #     range(0, self.egg.get_event_components().get_num_events())
    # )
    event_ids: list[int] = Field(default_factory=list)
    slider_values: list[int] = Field(default_factory=list)

    pcd_path: str | None = None
    title: str = "EGG Viz"

    # Open3D GUI components
    app: gui.Application = gui.Application.instance
    window: gui.Window | None = None
    scene_widget: gui.SceneWidget | None = None
    panel: gui.Vert | None = None
    slider: gui.Slider | None = None
    label: gui.Label = gui.Label("")

    material: rendering.MaterialRecord = rendering.MaterialRecord()
    material.shader = "defaultLit"

    pcd: o3d.geometry.PointCloud | None = None
    pcd_z_filter: float = Field(default=2.0, gt=0, frozen=True)

    viz_config: EGGVizConfig

    def _on_slider_value_changed(self, v):
        self.update_event(self.event_ids[int(round(v))])

    def _on_layout(self, ctx):
        assert self.window is not None
        assert self.panel is not None
        assert self.scene_widget is not None

        r = self.window.content_rect
        h = self.viz_config.panel_height
        # Bottom panel
        self.panel.frame = gui.Rect(r.x, r.y + r.height - h, r.width, h)
        # Scene above the panel
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width, r.height - h)

    def load_and_filter_pcd(self) -> o3d.geometry.PointCloud | None:
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

    def draw_event_node(self, event_id: int) -> list[VizElement]:
        # TODO: Implement
        events_viz: list[VizElement] = []
        event_node = self.egg.events.get_event_node_by_id(event_id)
        assert isinstance(event_node, EventNode)

        events_viz += self.draw_uninvolved_objects(event_node=event_node)
        return events_viz

    def draw_room_nodes(self) -> list[VizElement]:
        # TODO: Implement
        rooms_viz: list[VizElement] = []
        return rooms_viz

    def draw_uninvolved_objects(self, event_node: EventNode) -> list[VizElement]:
        obj_viz = []
        non_involved_ids = [
            id
            for id in list(self.egg.spatial.get_all_object_nodes().keys())
            # if id not in event_node.involved_object_ids
        ]
        for obj_node_id in non_involved_ids:
            obj_node = self.egg.spatial.get_object_node_by_id(obj_node_id)
            assert obj_node is not None
            if obj_node.has_been_seen(
                event_node.start
            ) and obj_node.object_class.lower() not in self.viz_config.ignore_classes:
                prev_timestamp, prev_pos = obj_node.get_previous_timestamp_and_states(
                    event_node.start
                )
                assert (
                    prev_pos is not None and prev_timestamp is not None
                ), f"Node {obj_node.name} failed, timestamps {list(obj_node.timestamped_states.keys())} ref_timestamp {event_node.start}"
                # prev_event_node = self.egg.events.get_event_node_by_timestamp(
                #     prev_timestamp
                # )
                # assert prev_event_node is not None
                # prev_room_node = self.egg.spatial.get_room_node_by_name(
                #     prev_event_node.location
                # )
                # assert prev_room_node is not None
                # prev_room_pos_viz = prev_room_node.position
                # prev_room_pos_viz[2] = self.room_offset
                obj_viz += [
                    VizElement(
                        name=f"Object {obj_node_id} Prev Node",
                        geometry=self.draw_aabb(
                            aabb=obj_node.timestamped_states[prev_timestamp].bounding_box,
                            color=self.viz_config.inactive_object_color,
                            # center=prev_pos,
                            # dim=OBJECT_NODE_DIM,
                            # color=INACTIVE_OBJECT_COLOR,
                        ),
                    ),
                    VizElement(
                        name=f"Object {obj_node_id} Prev Node Label",
                        geometry=self.draw_text_mesh(
                            text=f"{obj_node.name}",
                            position=obj_node.timestamped_states[prev_timestamp].bounding_box.center.as_numpy(),
                        ),
                    ),
                    # VizElement(
                    #     name=f"Object {obj_node_id} - {prev_room_node.name} Edge",
                    #     geometry=self.draw_line(
                    #         source=prev_pos,
                    #         target=prev_room_pos_viz,
                    #     ),
                    # ),
                ]
        return obj_viz

    def update_event(self, event_id: int):
        event_viz = self.draw_event_node(event_id)
        event_viz += self.draw_room_nodes()
        event_node = self.egg.get_event_components().get_event_node_by_id(event_id)
        assert self.scene_widget is not None
        self.scene_widget.scene.clear_geometry()
        if self.pcd is not None:
            self.scene_widget.scene.add_geometry("Scene Cloud", self.pcd, self.material)
        for element in event_viz:
            self.scene_widget.scene.add_geometry(
                element.name, element.geometry, self.material
            )
        if event_node is not None:
            self.label.text = event_node.pretty_str()
        else:
            self.label.text = "N/A"

    def setup_ui(self):
        self.event_ids = self.egg.get_event_components().get_event_ids()
        num_events = self.egg.get_event_components().get_num_events()
        if num_events == 0:
            self.event_ids = [0]
            num_events = 1
        self.slider_values = list(range(0, num_events))

        self.app.initialize()
        self.window = self.app.create_window(self.title, *self.viz_config.window_size)
        assert self.window is not None

        # Scene widget
        self.scene_widget = gui.SceneWidget()
        assert self.scene_widget is not None
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        # Bottom panel: slider on top, label underneath
        self.panel = gui.Vert(8, gui.Margins(8, 8, 8, 8))
        assert self.panel is not None
        self.slider = gui.Slider(gui.Slider.INT)
        assert self.slider is not None
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
        if self.pcd is not None:
            bounds = self.pcd.get_axis_aligned_bounding_box()
        else:
            # TODO: initialize some other defaults
            bounds = o3d.geometry.AxisAlignedBoundingBox(
                np.array([-8.725, -13.425, -0.225]), np.array([7.925, 7.725, 1.99998])
            )
        center = bounds.get_center()
        self.scene_widget.setup_camera(60.0, bounds, center)

        # Layout
        self.window.set_on_layout(self._on_layout)

    def draw_aabb(
        self,
        aabb: AxisAlignedBoundingBox,
        color: tuple[float, float, float],
    ) -> o3d.geometry.AxisAlignedBoundingBox:
        """Create an axis-aligned bounding box centered at `center`.

        :param center: The center position of the bounding box.
        :param color: The color of the bounding box (RGB in [0,1]).
        :param dim: The box dimension. If a float, uses the same length
            for x, y, z. If a 3-sequence, interpreted as (dx, dy, dz).
        :return: An o3d.geometry.AxisAlignedBoundingBox object.
        """
        center = aabb.center.as_numpy()

        half = aabb.size.as_numpy() / 2.0
        min_bound = center - half
        max_bound = center + half

        viz_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        viz_aabb.color = color
        return viz_aabb

    def draw_cube(
        self,
        center: NDArray[np.float32],
        color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        dim: float = 0.1,
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
        self,
        center: NDArray[np.float32],
        color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        dim: float = 0.1,
    ) -> o3d.geometry.TriangleMesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=dim, resolution=20)
        # Translate the sphere to the specified center
        translation = center - np.array([dim / 2, dim / 2, dim / 2])
        sphere.translate(translation)
        sphere.paint_uniform_color(color)
        return sphere

    def draw_line(
        self,
        source: NDArray[np.float32],
        target: NDArray[np.float32],
        line_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
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
        self, text: str, position: NDArray[np.float32]
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

    def run(self):
        self.setup_ui()
        self.app.run()
