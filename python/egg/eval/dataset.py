import ast
from typing import List
import logging

from egg.eval.qa_ground_truth import Modality, QAGroundTruth
from egg.utils.read_data import read_qa_data
from egg.graph.egg import EGG
from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="eval/dataset.log",
)


class QADataset:
    def __init__(self, qa_file: str, egg: EGG):
        self.egg = egg
        qa_gt_df = read_qa_data(qa_file=qa_file)
        self.qa_ground_truth_list: List[QAGroundTruth] = []
        for qa in qa_gt_df.values:
            query, modality, answer = qa
            if modality == "text":
                self.qa_ground_truth_list.append(
                    QAGroundTruth(query=query, modality=Modality.TEXT, answer=answer)
                )
            elif modality == "node":
                object_names = ast.literal_eval(answer)
                is_valid_answer = True
                for name in object_names:
                    obj_id = self.egg.get_spatial_graph().get_object_node_by_name(name)
                    if obj_id is None:
                        is_valid_answer = False
                if is_valid_answer:
                    self.qa_ground_truth_list.append(
                        QAGroundTruth(
                            query=query, modality=Modality.NODE, answer=object_names
                        )
                    )
            elif modality == "binary":
                self.qa_ground_truth_list.append(
                    QAGroundTruth(
                        query=query, modality=Modality.BINARY, answer=bool(int(answer))
                    )
                )
            elif modality == "time_point":
                self.qa_ground_truth_list.append(
                    QAGroundTruth(
                        query=query, modality=Modality.TIME_POINT, answer=answer
                    )
                )
            elif modality == "time_interval":
                self.qa_ground_truth_list.append(
                    QAGroundTruth(
                        query=query, modality=Modality.TIME_INTERVAL, answer=answer
                    )
                )
            else:
                raise AssertionError(f"Invalid modality {modality}")
    def pretty_str(self) -> str:
        dataset_str = ""
        for qa in self.qa_ground_truth_list:
            dataset_str += qa.pretty_str() + "\n"
        return dataset_str
