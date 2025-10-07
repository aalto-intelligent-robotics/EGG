from copy import deepcopy
from typing import List, Dict
from egg.utils.image import encode_image
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/prompts/image_captioning_prompts.log",
)

IMAGE_CAPTION_PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """
        You are a mobile robotic navigation assistant analyzing indoor scenes.
        You are given an image containing multiple views of a single object stacked vertically. Each views are taken from an image and masked out, so some part of the object might be occluded in each view. Also, only focus on the object to be described, and ignore objects nearby, inside of or on top of it.
        Your task is to describe the object in the given image.
        Outputs an image caption aiming to identify the object. Try to focus on what makes the object unique. E.g., "blue, ceramic, with white interior, has 'Something' printed on it".
        
        Only use the visual information from the image and do not make assumptions beyond what is visible.
        """,
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Here is the picture with one object. The object to be described is a {object_class}. Return a caption that describe it. Ignore other objects in the picture and only focus on the object to be described",
            },
            {
                "type": "image_url",
                "image_url": {},
            },
        ],
    },
]


def build_image_captioning_messages(image, object_class) -> List[Dict]:
    messages = deepcopy(IMAGE_CAPTION_PROMPT_TEMPLATE)

    image_url = encode_image(image)

    # Replace the placeholder in the content string with the data
    messages[-1]["content"][0]["text"] = messages[-1]["content"][0]["text"].format(
        object_class=object_class
    )
    messages[-1]["content"][1]["image_url"] = {"url": image_url}

    return messages
