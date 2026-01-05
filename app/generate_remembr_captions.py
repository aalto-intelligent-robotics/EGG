import os
import glob
import json
import logging
from scipy.spatial.transform import Rotation as R

from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="generate_remembr_captions.log",
)

def quaternion_to_yaw(quaternion):
    rotation = R.from_quat(quaternion)
    euler_angles = rotation.as_euler('xyz')
    yaw = euler_angles[2]
    return yaw

from egg.language.vlm import VLMAgent


vlm_agent = VLMAgent()

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

remembr_data = {}
for idx, event_param_file in enumerate(sorted(yaml_files)):
    summary_caption, timestamped_observation_odom = vlm_agent.generate_remembr_data_from_yaml(yaml_param_file=event_param_file)
    obs_timestamp = next(iter(timestamped_observation_odom))
    obs_odom = timestamped_observation_odom[obs_timestamp]
    obs_pos = [round(p, 3) for p in obs_odom["base_odom"][0]]
    theta = round(quaternion_to_yaw(obs_odom['base_odom'][1]), 3)
    data_dict = {idx: {"summary": summary_caption, "time": obs_timestamp, "position": obs_pos, "theta": theta}}
    remembr_data.update(data_dict)
    logger.debug(f"Param file: {event_param_file}\n-> ReMEmbR data: {data_dict}\n")

with open("remember_data.json", "w+") as f:
    json.dump(remembr_data, f)
