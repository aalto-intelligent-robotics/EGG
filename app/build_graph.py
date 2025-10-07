#!/usr/bin/env python3
import glob
import os
import logging
import json
import argparse

from egg.graph.spatial_graph import SpatialGraph
from egg.graph.event_graph import EventGraph
from egg.graph.egg import EGG
from egg.utils.logger import getLogger
from egg.language.llm import LLMAgent

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--auto", action="store_true")
parser.add_argument("-u", "--unguided", action="store_true")
args = parser.parse_args()

viz_elements = []

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="build_graph.log",
)
spatial_graph = SpatialGraph()
event_graph = EventGraph()
use_gt_caption = not args.auto
use_guided_auto_caption = not args.unguided

egg = EGG(
    spatial_graph,
    event_graph,
    use_gt_caption=use_gt_caption,
    use_guided_auto_caption=use_guided_auto_caption,
)

event_dirs = [
    "/home/ros/data/coffee_room_events/batch_1/events_gt/",
    "/home/ros/data/coffee_room_events/batch_2/events_gt/",
    "/home/ros/data/coffee_room_events/batch_3/events_gt/",
    "/home/ros/data/coffee_room_events/batch_4/events_gt/",
    "/home/ros/data/coffee_room_events/batch_5/events_gt/",
    "/home/ros/data/coffee_room_events/batch_6/events_gt/",
    "/home/ros/data/coffee_room_events/batch_7/events_gt/",
]

yaml_files = []
cam_config_file = "../configs/camera/astra2.yaml"
for directory in event_dirs:
    yaml_files += glob.glob(os.path.join(directory, "*.yaml")) + glob.glob(
        os.path.join(directory, "*.yml")
    )

for event_param_file in sorted(yaml_files):
    egg.add_event_from_video(
        event_param_file=event_param_file,
        camera_config_file=cam_config_file,
    )

egg.gen_room_nodes()

# llm_agent = LLMAgent(use_mini=False)
# egg.gen_object_captions(llm_agent=llm_agent)

logger.info(egg.pretty_str())
# if use_gt_caption:
#     graph_filename = "graph_gt.json"
# elif use_guided_auto_caption:
#     graph_filename = "graph_auto_guided.json"
# else:
#     graph_filename = "graph_auto_unguided.json"
# with open(graph_filename, "w") as f:
#     json.dump(egg.serialize(), f)
