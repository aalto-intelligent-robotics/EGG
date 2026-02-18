from copy import deepcopy
from typing import List
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/prompts/video_captioning_prompts.log",
)

GUIDED_VIDEO_SUMMARY_CAPTION_TEMPLATE = "<video>\nIn the video, the person is doing something with: {objects}. Describe what the person <object0><region> is doing in the video. Return your answer in natural language and do not use <object> to identify which object is which."

UNGUIDED_VIDEO_SUMMARY_CAPTION_TEMPLATE = "<video>\nIn the video, the person is doing something. Describe what the person <object0><region> is doing in the video. Return your answer in natural language and do not use <object> to identify which object is which."


def build_video_summary_caption_query(objects: List[str], guided: bool = True) -> str:
    if guided:
        message = deepcopy(GUIDED_VIDEO_SUMMARY_CAPTION_TEMPLATE)
        return message.format(objects=objects)
    else:
        message = deepcopy(UNGUIDED_VIDEO_SUMMARY_CAPTION_TEMPLATE)
        return message


VIDEO_OBJECT_ROLE_CAPTION_TEMPLATE = "<video>\nIn the video, the person <object0><region> performing the action: {summary}. Describe the role of the {object_of_interest} <object1><region> in the person's action in the video. Return your answer in natural language and do not use <object> to identify which object is which."

def build_video_object_role_caption_query(summary: str, object_of_interest: str) -> str:
    message = deepcopy(VIDEO_OBJECT_ROLE_CAPTION_TEMPLATE)
    return message.format(
        summary=summary, object_of_interest=object_of_interest
    )

# NOTE: Adopted and modified from https://github.com/NVIDIA-AI-IOT/remembr/blob/964faab296fe70f8a8dbb9135adf20fecb758525/examples/nova_carter_demo/python/captioner_node.py#L26 to suit the evalutaion

REMEMBR_VIDEO_SUMMARY_CAPTION_TEMPLATE = """
    <video>\nYou are a wandering around inside a building. You are observing the actions of a person <object0><region> doing something with {objects}.
    The person is either inside the coffee room or the office.
    Describe the person's actions in the video in as much detail as possible.
    Specifically focus on the actions of the person <object0><region>, the appearances of the objects, events/ectivities, and other interesting details.
    Think step by step about these details and be very specific.
"""

def build_remembr_video_summary_query(objects: List[str]) -> str:
    message = deepcopy(REMEMBR_VIDEO_SUMMARY_CAPTION_TEMPLATE)
    return message.format(objects=objects)
