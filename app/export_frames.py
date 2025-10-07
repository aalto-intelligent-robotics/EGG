#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import json
import signal
import os
import numpy as np


class ImageOdometrySaver:
    def __init__(
        self,
        out_directory,
        map_frame: str = "/map",
        base_frame: str = "/base_link",
        camera_frame: str = "/astra2_link",
    ):
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.saved_data = {}
        self.image_count = 0
        self.out_directory = out_directory
        self.image_directory = os.path.join(self.out_directory, "images")
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.depth_img = None
        self.prev_cam_odom = None
        self.prev_base_odom = None
        os.makedirs(self.image_directory, exist_ok=True)
        os.makedirs(self.image_directory + "/color", exist_ok=True)
        os.makedirs(self.image_directory + "/depth", exist_ok=True)

    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Generate a unique filename
            color_filename = f"color_frame_{self.image_count:04d}.png"
            color_path = os.path.join(self.image_directory, "color", color_filename)
            # Save the image
            cv2.imwrite(color_path, cv_image)

            if self.depth_image is not None:
                depth_filename = f"depth_frame_{self.image_count:04d}.npy"
                depth_path = os.path.join(self.image_directory, "depth", depth_filename)
                np.save(depth_path, self.depth_image)
            else:
                depth_path = None

            # Get odometry for camera_link and base_link at the closest timestamp
            timestamp = msg.header.stamp
            camera_odom, base_odom = self.get_odom_data(
                timestamp, prev_cam=self.prev_cam_odom, prev_base=self.prev_base_odom
            )
            self.prev_cam_odom = camera_odom
            self.prev_base_odom = base_odom

            # Save data to dictionary
            self.saved_data[self.image_count] = {
                "color_image": os.path.abspath(color_path),
                "depth_image": (
                    os.path.abspath(depth_path) if depth_path is not None else None
                ),
                "timestamp": timestamp.to_nsec(),
                "camera_odom": camera_odom,
                "base_odom": base_odom,
            }

            self.image_count += 1
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def depth_callback(self, msg):
        cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        self.depth_image = cv_depth.astype(np.float32) / 1000.0

    def get_odom_data(self, timestamp, prev_cam=None, prev_base=None):
        camera_odom = prev_cam
        base_odom = prev_base

        try:
            # Get transformations for camera_link and base_link
            camera_odom = self.listener.lookupTransform(
                self.map_frame, self.camera_frame, timestamp
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logwarn(f"Failed to fetch {self.camera_frame} transformation")

        try:
            base_odom = self.listener.lookupTransform(
                self.map_frame, self.base_frame, timestamp
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logwarn(f"Failed to fetch {self.base_frame} transformation")

        return camera_odom, base_odom

    def save_json(self):
        with open(os.path.join(self.out_directory, "image_odometry_data.json"), "w") as json_file:
            json.dump(self.saved_data, json_file, indent=4)
        rospy.loginfo(f"Saved image data to {self.out_directory}/image_odometry_data.json")


def signal_handler(signal, frame):
    rospy.loginfo("Ctrl+C pressed. Saving JSON file...")
    saver.save_json()
    rospy.signal_shutdown("Shutting down")


if __name__ == "__main__":
    rospy.init_node("image_odometry_saver", anonymous=True)

    color_topic = rospy.get_param("~color_topic", "/astra2/color/image_raw")
    depth_topic = rospy.get_param("~depth_topic", "/astra2/depth/image_raw")
    map_frame = rospy.get_param("~map_frame", "/map")
    base_frame = rospy.get_param("~base_frame", "/base_link")
    camera_frame = rospy.get_param("~camera_frame", "/astra2_link")
    data_directory = rospy.get_param("~data_directory", "/home/ros/data/coffee_room_events/")
    data_name = rospy.get_param("~data_name", "batch_5")
    out_directory = os.path.join(str(data_directory), str(data_name))
    saver = ImageOdometrySaver(
        map_frame=str(map_frame),
        base_frame=str(base_frame),
        camera_frame=str(camera_frame),
        out_directory=str(out_directory),
    )
    signal.signal(signal.SIGINT, signal_handler)

    rospy.Subscriber(color_topic, Image, saver.image_callback)
    rospy.Subscriber(depth_topic, Image, saver.depth_callback)

    rospy.loginfo("Started image and odometry saver node.")
    rospy.spin()
