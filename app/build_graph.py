#!/usr/bin/env python3

import glob
import os
import logging
import argparse

from egg.graph.spatial import SpatialComponents
from egg.graph.event import EventComponents
from egg.graph.egg import EGG
from egg.utils.logger import getLogger
from egg.language.openai_agent import OpenaiAgent

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--auto", action="store_true")
parser.add_argument("--aalto", action="store_true")
parser.add_argument("-u", "--unguided", action="store_true")
parser.add_argument("-d", "--data-path", default="/home/ros/data/")
args = parser.parse_args()

viz_elements = []

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="build_graph.log",
)
spatial_graph = SpatialComponents()
event_graph = EventComponents()
use_gt_caption = not args.auto
use_guided_auto_caption = not args.unguided

egg = EGG(
    spatial_graph,
    event_graph,
    use_gt_caption=use_gt_caption,
    use_guided_auto_caption=use_guided_auto_caption,
)

event_dirs = [
    f"{args.data_path}/batch_1/",
    f"{args.data_path}/batch_2/",
    f"{args.data_path}/batch_3/",
    f"{args.data_path}/batch_4/",
    f"{args.data_path}/batch_5/",
    f"{args.data_path}/batch_6/",
    f"{args.data_path}/batch_7/",
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
llm_agent = OpenaiAgent(use_mini=False, aalto=args.aalto)
egg.gen_object_captions(llm_agent=llm_agent)

logger.info(egg.pretty_str())
