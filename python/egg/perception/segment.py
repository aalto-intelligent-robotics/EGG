from torch import Tensor
from ultralytics import YOLOWorld
from ultralytics import SAM
from typing import List, Union
import numpy
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="perception/segment.log",
)


class YOLOWorld_SAM:
    """
    YOLOWorld_SAM integrates YOLOWorld object detection and SAM segmentation.

    :param yolo_path: Path to the YOLOWorld model file.
    :type yolo_path: str
    :param sam_path: Path to the SAM model file.
    :type sam_path: str
    :param classes: List of class names for detection.
    :type classes: List[str]
    :param conf: Confidence threshold for object detection, defaults to 0.5.
    :type conf: float, optional

    :ivar detector: Instance of YOLOWorld for object detection.
    :vartype detector: ultralytics.YOLOWorld
    :ivar segmenter: Instance of SAM for mask segmentation.
    :vartype segmenter: ultralytics.SAM
    :ivar conf: Confidence threshold for detection.
    :vartype conf: float

    .. note::
        This class requires pre-trained YOLOWorld and SAM models.

    Methods
    -------
    get_masks(image)
        Detects objects in the image and returns their masks.

    """

    def __init__(
        self, yolo_path: str, sam_path: str, classes: List[str], conf: float = 0.5
    ):
        self.detector = YOLOWorld(yolo_path)
        self.segmenter = SAM(sam_path)
        self.detector.set_classes(classes)
        self.conf = conf

    def get_masks(self, image: numpy.ndarray) -> Union[numpy.ndarray, None]:
        """
        Detects objects in the input image and computes segmentation masks for them.

        :param image: Input image for detection and segmentation.
        :type image: numpy.ndarray

        :return: Array of segmentation masks of detected objects, or None if no objects
        are detected.
        :rtype: Union[numpy.ndarray, None]

        .. note::
            This method first performs object detection using the YOLOWorld detector.
            If objects are detected within the confidence threshold, it uses their
            bounding boxes to compute segmentation masks with the SAM segmenter.

        """
        result = self.detector.predict(image, conf=self.conf)[0]
        boxes = []
        names = []

        if result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Convert to integers
                class_id = int(cls)
                name = result.names[class_id]
                names.append(name)
                # Prepare SAM
                boxes.append([x1, y1, x2, y2])
            seg_results = self.segmenter(result.orig_img, bboxes=boxes)
            result.masks = seg_results[0].masks

        masks = None
        if result.masks is not None:
            if isinstance(result.masks.data, Tensor):
                masks = result.masks.data.cpu().detach().numpy().astype(numpy.uint8)
        return masks
