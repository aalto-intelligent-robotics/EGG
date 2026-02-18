from dataclasses import dataclass
from typing import Union
import yaml
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from numpy.typing import NDArray
import logging

from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/camera.log",
)


@dataclass
class Camera:
    """
    Camera model that encapsulates intrinsic and extrinsic parameters for a pinhole
    camera.

    :param fx: Focal length in the x-axis.
    :type fx: float
    :param fy: Focal length in the y-axis.
    :type fy: float
    :param cx: Principal point offset in the x-axis.
    :type cx: float
    :param cy: Principal point offset in the y-axis.
    :type cy: float
    :param width: Width of the camera image.
    :type width: int
    :param height: Height of the camera image.
    :type height: int
    :param T: Extrinsic transformation matrix.
    :type T: NDArray

    Methods
    -------
    from_yaml(yaml_file)
        Creates a Camera instance from a YAML file.
    set_T(position, orientation)
        Sets the extrinsic transformation matrix using position and orientation.
    depth_to_pointcloud(depth_image, mask)
        Converts a depth image to a 3D point cloud.

    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    transformation_matrix: NDArray[np.float32] = np.eye(4).astype(np.float32)

    @staticmethod
    def from_yaml(yaml_file: str):
        """
        Creates a Camera instance from a YAML file containing camera intrinsic
        parameters.

        :param yaml_file: Path to the YAML file with camera specifications.
        :type yaml_file: str

        :return: Camera instance initialized with values from the YAML file.
        :rtype: Camera

        .. note::
            The transformation matrix `T` is initialized to an identity matrix.
        """
        with open(yaml_file, "r") as f:
            camera_info = yaml.safe_load(f)

        return Camera(
            fx=camera_info["fx"],
            fy=camera_info["fy"],
            cx=camera_info["cx"],
            cy=camera_info["cy"],
            width=camera_info["width"],
            height=camera_info["height"],
            transformation_matrix=np.eye(4).astype(np.float32),
        )

    def set_T(self, position: NDArray[np.float32], orientation: NDArray[np.float32]):
        """
        Sets the camera's extrinsic transformation matrix using given position and
        orientation.

        :param position: Translation vector for the camera.
        :type position: NDArray
        :param orientation: Quaternion representing camera orientation.
        :type orientation: NDArray
        """
        transformation_matrix = np.eye(4).astype(np.float32)
        transformation_matrix[:3, :3] = R.from_quat(orientation).as_matrix()
        transformation_matrix[:3, 3] = position
        self.transformation_matrix = transformation_matrix

    def depth_to_pointcloud(
        self,
        depth_image: NDArray[np.float32],
        mask: Union[NDArray[np.uint8], None] = None,
    ) -> NDArray[np.float32]:
        """
        Converts a depth image to a 3D point cloud using the camera's intrinsic parameters.

        :param depth_image: Input depth image, representing the distance of each pixel
                            from the camera.
        :type depth_image: NDArray[np.float32]
        :param mask: Optional mask to filter out specific areas in the depth image.
        :type mask: Union[NDArray[np.uint8], None]

        :return: Array of 3D points derived from the depth image.
        :rtype: NDArray[np.float32]

        :raises AssertionError: If the dimensions of the depth image do not match the
                                camera model.

        .. note::
            The depth image is assumed to align with the camera's field of view. Each pixel's
            depth value is converted into a 3D point using intrinsic camera parameters. If a
            mask is provided, it is applied to the depth image before conversion.
        """
        rows, cols = depth_image.shape
        if rows != self.height or cols != self.width:
            raise AssertionError(
                f"Depth image dimensions ({cols}, {rows}) do not match camera model "
                + f"dimensions ({self.width}, {self.height})"
            )
        # Apply mask to the depth image if provided
        processed_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        # Generate grid of pixel coordinates (u, v)
        u_coords, v_coords = np.meshgrid(np.arange(cols), np.arange(rows))

        # Flatten (u, v) arrays and extract valid depth values
        depth_values = processed_depth.flatten()
        valid_depth_indices = depth_values > 0
        valid_depth_values = depth_values[valid_depth_indices]
        u_valid = u_coords.flatten()[valid_depth_indices]
        v_valid = v_coords.flatten()[valid_depth_indices]

        # Calculate x and y coordinates in the camera plane
        x_coords = (u_valid - self.cx) * valid_depth_values / self.fx
        y_coords = (v_valid - self.cy) * valid_depth_values / self.fy

        # Combine x, y, and depth into homogeneous coordinates
        homogeneous_coordinates = np.vstack(
            (
                x_coords,
                y_coords,
                valid_depth_values,
                np.ones_like(valid_depth_values),
            )
        )

        # Apply the extrinsic transformation matrix to compute world coordinates
        world_coordinates = self.transformation_matrix @ homogeneous_coordinates

        # Return 3D points by extracting the x, y, z components
        point_cloud = world_coordinates[:3].T
        return point_cloud
