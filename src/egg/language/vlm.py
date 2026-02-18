import os
import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List
import logging
from transformers import logging as trf_logging

from videorefer_videollama3 import model_init, get_model_output
from videorefer_videollama3.mm_utils import load_video

from egg.utils.image import xy_to_binary_mask
from egg.language.prompts.video_captioning_prompts import (
    build_video_summary_caption_query,
    build_video_object_role_caption_query,
    build_remembr_video_summary_query,
)
from egg.utils.read_data import get_event_data, get_image_odometry_data

from egg.utils.logger import getLogger

trf_logging.set_verbosity_error()

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/vlm.log",
)


class VLMAgent:
    def __init__(
        self,
        model_path: str = "DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B",
        do_sample: bool = False,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)

        self.model, self.processor, self.tokenizer = model_init(
            model_path, device_map={"": device}
        )
        logger.info(f"👀 Using {model_path} VLM")

        assert self.model.generation_config is not None
        self.model.generation_config.top_k = None
        self.model.generation_config.top_p = None
        self.model.generation_config.temperature = None
        self.model.generation_config.do_sample = False

        for m in self.model.modules():
            m.tokenizer = self.tokenizer
        self.do_sample = do_sample

    def generate_video_caption(
        self, video_tensor: Tuple, masks: Tensor, query: str
    ) -> str:
        return str(
            get_model_output(
                video_tensor,
                query,
                model=self.model,
                tokenizer=self.tokenizer,
                masks=masks,
                mask_ids=[0 for _ in range(len(masks))],
                modal="video",
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
            )
        )

    def generate_image_caption(
        self, image_data: NDArray, masks: Tensor, query: str
    ) -> str:
        return str(
            get_model_output(
                [image_data],
                query,
                model=self.model,
                tokenizer=self.tokenizer,
                masks=masks,
                mask_ids=[0 for _ in range(len(masks))],
                modal="image",
                image_downsampling=1,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
            )
        )

    def generate_captions_from_yaml(
        self,
        yaml_param_file: str,
        guided: bool = True,
    ) -> Tuple[str, Dict[str, str]]:
        event_data = get_event_data(yaml_param_file)

        event_dir = os.path.dirname(os.path.abspath(yaml_param_file))
        video_path = os.path.join(event_dir, event_data.get("clip_path"))

        frame_idx = 0

        objects = []
        object_properties = []
        first_frame = int(event_data.get("from_frame"))
        for obj, obj_properties in event_data.get("objects_of_interest").items():
            objects.append(obj)
            object_properties.append(obj_properties)

        query = build_video_summary_caption_query(objects=objects, guided=guided)
        video_tensor = load_video(
            video_path, fps=5, max_frames=768, frame_ids=[frame_idx]
        )
        _, video_height, video_width = video_tensor[0][0].shape
        person_mask_np = xy_to_binary_mask(
            width=video_width,
            height=video_height,
            xy_polygon=event_data.get("first_person_mask"),
        )
        masks = []
        masks.append(person_mask_np)
        masks = np.array(masks)
        masks = torch.from_numpy(masks).to(torch.uint8)
        summary_caption = self.generate_video_caption(
            video_tensor=video_tensor, masks=masks, query=query
        )
        edge_captions = {}
        for obj_name, obj_attr in zip(objects, object_properties):
            obj_name = obj_name
            instance_query = build_video_object_role_caption_query(
                summary=summary_caption, object_of_interest=obj_name
            )
            frame_idx = int(obj_attr.get("first_frame")) - first_frame
            video_tensor = load_video(
                video_path, fps=5, max_frames=768, frame_ids=[frame_idx]
            )
            masks = []
            object_mask_np = xy_to_binary_mask(
                width=video_width,
                height=video_height,
                xy_polygon=obj_attr.get("first_mask"),
            )
            masks = np.array([person_mask_np, object_mask_np])
            masks = torch.from_numpy(masks).to(torch.uint8)
            caption = self.generate_video_caption(
                video_tensor=video_tensor, masks=masks, query=instance_query
            )
            edge_captions.update({obj_name: caption})
        return (summary_caption, edge_captions)

    def generate_remembr_data_from_yaml(
        self,
        yaml_param_file: str,
    ) -> Tuple[str, Dict[int, Dict[str, List]]]:
        event_data = get_event_data(yaml_param_file)

        event_dir = os.path.dirname(os.path.abspath(yaml_param_file))
        video_path = os.path.join(event_dir, event_data.get("clip_path"))

        image_odometry_file = os.path.join(
            event_dir, event_data.get("image_odometry_file")
        )
        timestamped_observation_odom, _, _, _ = (
            get_image_odometry_data(
                image_odometry_file=image_odometry_file,
                from_frame=event_data.get("from_frame"),
                to_frame=event_data.get("to_frame"),
            )
        )

        frame_idx = 0

        objects = []
        for obj, _ in event_data.get("objects_of_interest").items():
            objects.append(obj)
        query = build_remembr_video_summary_query(objects=objects)
        video_tensor = load_video(
            video_path, fps=5, max_frames=768, frame_ids=[frame_idx]
        )
        _, video_height, video_width = video_tensor[0][0].shape
        person_mask_np = xy_to_binary_mask(
            width=video_width,
            height=video_height,
            xy_polygon=event_data.get("first_person_mask"),
        )
        masks = []
        masks.append(person_mask_np)
        masks = np.array(masks)
        masks = torch.from_numpy(masks).to(torch.uint8)
        summary_caption = self.generate_video_caption(
            video_tensor=video_tensor, masks=masks, query=query
        )
        return summary_caption, timestamped_observation_odom
