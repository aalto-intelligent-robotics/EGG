import json
from typing import Optional, Dict
import re
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/language_utils.log",
)


def remove_explanation_and_convert(json_string: str) -> Optional[Dict]:
    lines = json_string.splitlines()
    filtered_lines = [line for line in lines if "explanation" not in line]
    filtered_json_string = "\n".join(filtered_lines)
    try:
        python_dict = json.loads(filtered_json_string)
    except json.JSONDecodeError:
        filtered_json_string = re.sub(r",\s*}", "}", filtered_json_string)
        filtered_json_string = re.sub(r",\s*\]", "]", filtered_json_string)
        try:
            python_dict = json.loads(filtered_json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            return None
    return python_dict


def get_eval_accuracy(json_string: str) -> float:
    eval_data = remove_explanation_and_convert(json_string)[0]
    assert (
        "accuracy" in eval_data.keys()
    ), f"Invalid eval data: {eval_data}. 'accuracy' key missing"
    return float(eval_data["accuracy"])
