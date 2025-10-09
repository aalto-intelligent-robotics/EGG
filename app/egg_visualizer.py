#!/usr/bin/env python3

import argparse

from egg.utils.visualizer import EGGVisualizer
from egg.graph.spatial import SpatialComponents
from egg.graph.event_graph import EventComponents
from egg.graph.egg import EGG

PCD_COLOR = [0.2, 0.5, 0.7]
EVENT_COLOR = [0.2, 0.2, 0.6]
EVENT_NODE_DIM = 0.2
OBJECT_COLOR = [0.8, 0.3, 0.5]
OBJECT_NODE_DIM = 0.2
EDGE_COLOR = [0.2, 0.2, 0.2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="./graph_gt.json")
    parser.add_argument("-p", "--pcd", default="/home/ros/data/map_cloud.pcd")
    args = parser.parse_args()
    spatial_graph = SpatialComponents()
    event_graph = EventComponents()
    egg = EGG(spatial_graph, event_graph)
    egg.deserialize(args.file)
    egg.gen_room_nodes()
    # Customize these as needed:

    vis = EGGVisualizer(
        egg=egg,
        pcd_path=args.pcd,
        panel_height=320,
        window_size=(1024, 768),
    )
    vis.run()
