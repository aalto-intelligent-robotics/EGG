import json
import logging

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
            self.eval_data = json.load(fp)

    def get_failure_eval_data(self):
        failure_data = {}
        for q_id, qa_data in self.eval_data.items():
            if q_id != "usage_metadata":
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
            if q_id != "usage_metadata":
                if qa_data["modality"] in modality_list:
                    eval_data_by_modality.update({q_id: qa_data})
        return eval_data_by_modality
