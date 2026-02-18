import math
import base64
from typing import List
import numpy as np
from numpy.typing import NDArray
import cv2
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import pad
from PIL import Image
import logging

from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/image.log",
)


def get_instance_view(
    map_view_img: NDArray,
    mask: NDArray,
    mask_bg: bool = True,
    crop: bool = True,
    padding: int = 5,
) -> np.ndarray:
    """Get a view of the instance from the provided map view image and mask.

    This function extracts the view of an instance from a map view image
    using the provided mask. It can also crop, mask the background and
    apply padding to the view if specified.

    :param map_view_img: The map view image from which the instance view
        is to be extracted. This is a 3D numpy array with shape (height,
        width, channels).
    :param mask: The mask indicating the region of the instance in the
        map view image.
    :param mask_bg: If True, applies a black mask to the background of
        the image. Default is True.
    :param crop: If True, crops the image to the bounding box of the
        instance. Default is True.
    :param padding: The padding to be applied to the cropped view.
        Default is 5 pixels.
    :return: An image of the view of the instance.
    """
    coords = cv2.findNonZero(mask)
    # Get bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(coords)
    # Crop the image using the bounding box
    if mask_bg:
        image = cv2.bitwise_and(map_view_img, map_view_img, mask=mask)
    else:
        image = map_view_img
    if crop:
        image = image[
            max(y - padding, 0) : min(y + padding + h, map_view_img.shape[0]),
            max(x - padding, 0) : min(x + padding + w, map_view_img.shape[1]),
        ]
    return image


class SquarePad:
    """Class to apply square padding to an image.

    This class pads an image to make its dimensions square by adding
    equal padding to all sides.
    """

    def __call__(self, image) -> Tensor:
        """Apply square padding to the given image.

        :param image: The image to be padded. It is expected to be a PIL
            or torch.Tensor image.
        :return: The padded image as a Tensor.
        """
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return pad(image, padding, 0, "constant")


def preprocess_image(image: NDArray, to_cuda: bool = True) -> Tensor:
    """Preprocess the input image for model inference.

    This function preprocesses the input image by applying square
    padding, resizing, normalization, and converting it to a tensor. It
    optionally moves the tensor to GPU.

    :param image: The input image to be preprocessed. It is expected to
        be a numpy array with shape (height, width, channels).
    :param to_cuda: If True, moves the preprocessed image tensor to CUDA
        (GPU). Default is True.
    :return: The preprocessed image tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    transform_val = Compose(
        [
            SquarePad(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image_tensor = transform_val(image_pil)
    assert isinstance(image_tensor, Tensor)
    if to_cuda:
        return image_tensor[None, :].cuda()
    else:
        return image_tensor[None, :]


def xy_to_binary_mask(width: int, height: int, xy_polygon: List[List[int]]) -> NDArray:
    mask = np.zeros((height, width))
    cv2.fillPoly(img=mask, pts=[np.array(xy_polygon, dtype=np.int32)], color=1)
    return mask.astype(np.uint8)


def encode_image(image: NDArray, image_type: str = "image/png") -> str:
    _, buffer = cv2.imencode(".jpg", image)
    encoded_string = base64.b64encode(buffer).decode("utf-8")
    return f"data:{image_type};base64,{encoded_string}"

def pad_images_to_width(images, target_width):
    """Pad images horizontally to have the same width."""
    padded_images = []
    for img in images:
        if img.shape[1] < target_width:  # If the image width is less than target width
            padding = (0, target_width - img.shape[1])  # Calculate padding needed
            padded_img = np.pad(img, ((0, 0), padding, (0, 0)), mode='constant')  # Pad the image
        else:
            padded_img = img  # No padding needed
        padded_images.append(padded_img)
    return padded_images

def concatenate_images_vertically(images):
    """Concatenate images vertically."""
    if not images:
        raise ValueError("The list of images is empty")
    
    # Find the maximum width among all images
    max_width = max(img.shape[1] for img in images)
    
    # Pad images to the maximum width
    padded_images = pad_images_to_width(images, max_width)
    
    # Concatenate the images vertically
    concatenated_image = np.concatenate(padded_images, axis=0)
    
    return concatenated_image
