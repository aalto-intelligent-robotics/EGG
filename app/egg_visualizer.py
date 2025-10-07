from dataclasses import dataclass
import json
from typing import Optional, List, Tuple
import tf.transformations as tr
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from egg.utils.visualizer import EGGVisualizer
from egg.graph.spatial_graph import SpatialGraph
from egg.graph.event_graph import EventGraph
from egg.graph.egg import EGG

PCD_COLOR = [0.2, 0.5, 0.7]
EVENT_COLOR = [0.2, 0.2, 0.6]
EVENT_NODE_DIM = 0.2
OBJECT_COLOR = [0.8, 0.3, 0.5]
OBJECT_NODE_DIM = 0.2
EDGE_COLOR = [0.2, 0.2, 0.2]


if __name__ == "__main__":
    spatial_graph = SpatialGraph()
    event_graph = EventGraph()
    egg = EGG(spatial_graph, event_graph)
    egg.deserialize("./graph_gt.json")
    egg.gen_room_nodes()
    # Customize these as needed:
    PCD_PATH = "/home/ros/data/map_cloud.pcd"

    vis = EGGVisualizer(
        egg=egg,
        pcd_path=PCD_PATH,
        panel_height=320,
        window_size=(1024, 768),
    )
    vis.run()
