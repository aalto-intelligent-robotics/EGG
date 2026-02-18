import numpy as np
from numpy.typing import NDArray
import cv2
import json
from typing import Dict, Tuple, List
import yaml
import logging
import pandas as pd

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/read_data.log",
)


def get_hydra_data(dsg_path) -> Tuple[Dict, Dict, Dict]:
    """
    Load Hydra dynamic scene graph (dsg) data from the specified 3dsg output path.

    :param dsg_path: The path to the directory containing the 3dsg data
        files.
    :return: A tuple containing three dictionaries with instance views
        data, map views data, and 3DSG data respectively.
    """
    with open(f"{dsg_path}/instance_views/instance_views.json") as f:
        instance_views_data = json.load(f)
    with open(f"{dsg_path}/map_views/map_views.json") as f:
        map_views_data = json.load(f)
    with open(f"{dsg_path}/backend/dsg_with_mesh.json") as f:
        dsg_data = json.load(f)
    return instance_views_data, map_views_data, dsg_data


def get_map_views(map_views_data) -> Dict[int, NDArray]:
    """Register map views from provided data and load them as images.

    :param map_views_data: A list of dictionaries containing map view
        data.
    :return: A dictionary mapping map view IDs to their corresponding
        images as numpy arrays.
    """
    map_views = {}
    for view_data in map_views_data:
        map_view_file = view_data["file"]
        map_view_id = view_data["map_view_id"]
        map_views[map_view_id] = cv2.imread(map_view_file)
    return map_views


def get_node_attrs(dsg_data, node_id) -> Dict:
    """Retrieve attributes for a specific node in the 3DSG data loaded from
    dsg_with_mesh.json file.

    :param dsg_data: The DSG data  in which the node is located.
    :param node_id: The ID of the node whose attributes are to be
        retrieved.
    :return: A dictionary of attributes for the specified node. Returns
        an empty dictionary if the node is not found.
    """
    for node_data in dsg_data["nodes"]:
        if node_data["id"] == node_id:
            return node_data["attributes"]
    return {}


def get_image_odometry_data(
    image_odometry_file: str, from_frame: int, to_frame: int
) -> Tuple[Dict[int, Dict[str, List]], Dict[int, int], int, int]:
    assert (
        from_frame < to_frame
    ), f"from_frame < to_frame, but got from_frame={from_frame} >= to_frame={to_frame}"
    with open(image_odometry_file, "r") as image_odometry_fh:
        image_odometry_data = json.load(image_odometry_fh)
    frame_timestamp_map = {}
    timestamped_observation_positions = {}
    start = None
    end = None
    for frame_id in range(from_frame, to_frame + 1):
        odom_data = image_odometry_data[str(frame_id)]
        timestamp = odom_data.get("timestamp")
        if frame_id == from_frame:
            start = timestamp
        elif frame_id == to_frame:
            end = timestamp
        base_odom = odom_data.get("base_odom")
        camera_odom = odom_data.get("camera_odom")

        if base_odom is not None and camera_odom is not None:
            timestamped_observation_positions.update(
                {int(timestamp): {"base_odom": base_odom, "camera_odom": camera_odom}}
            )
        frame_timestamp_map.update({frame_id: int(timestamp)})
    assert (
        start is not None and end is not None
    ), "Unable to get start and end timestamp"
    return timestamped_observation_positions, frame_timestamp_map, start, end


def get_event_data(yaml_param_file: str):
    with open(yaml_param_file, "r") as event_fh:
        event_data = yaml.safe_load(event_fh)
    verify_event_data(event_data)
    return event_data


def verify_event_data(event_data: Dict):
    required_keys = [
        "clip_path",
        "image_path",
        "objects_of_interest",
        "image_odometry_file",
        "from_frame",
        "to_frame",
        "first_person_mask",
        "location",
    ]
    for key in required_keys:
        if key not in event_data.keys():
            raise AssertionError(
                f"{key} is missing in event data. Available keys {event_data.keys()}"
            )


def read_qa_data(qa_file: str):
    assert qa_file.lower().endswith(
        ".csv"
    ), f"QA data file needs to end with '.csv', but provided {qa_file}"
    with open(qa_file, "r") as f:
        qa_data = pd.read_csv(qa_file, delimiter="|")
    return qa_data
