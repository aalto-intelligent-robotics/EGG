import json
import logging
from typing import Dict

from egg.utils.logger import getLogger
from egg.eval.qa_ground_truth import Modality


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="eval/evaluator.log",
)


class EGGAnalyzer:
    """
    Class to evaluate the results of EGG
    """
    def __init__(self, eval_data_file: str):
        with open(eval_data_file, "r") as fp:
            self.eval_data: Dict = json.load(fp)

    def get_failure_eval_data(self):
        failure_data = {}
        for q_id, qa_data in self.eval_data.items():
            if qa_data["accuracy"] != 1:
                failure_data.update({q_id: qa_data})
        return failure_data

    def get_eval_data_by_modality(self, modality: str):
        eval_data_by_modality = {}
        if modality == "time":
            modality_list = [
                Modality.TIME_INTERVAL.name.lower(),
                Modality.TIME_POINT.name.lower(),
            ]
        elif modality == "all":
            modality_list = [m.name.lower() for m in Modality]
        else:
            modality_list = [modality]
        for q_id, qa_data in self.eval_data.items():
            if qa_data["modality"] in modality_list:
                eval_data_by_modality.update({q_id: qa_data})
        return eval_data_by_modality


    def get_token_usage(self):
        total_input_tokens = 0
        total_output_tokens = 0
        for qa_data in self.eval_data.values():
            total_input_tokens += qa_data["input_tokens"]
            total_output_tokens += qa_data["output_tokens"]
        return total_input_tokens, total_output_tokens
