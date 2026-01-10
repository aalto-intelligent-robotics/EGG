import logging
from typing import Dict, List, Optional, Tuple
import json
from ast import literal_eval
from datetime import datetime

from egg.eval.qa_ground_truth import Modality, QAGroundTruth
from egg.language.openai_agent import OpenaiAgent
from egg.language.prompts.answer_templates import EVALUATOR_RESPONSE_FORMAT
from egg.utils.logger import getLogger
from egg.utils.language_utils import get_eval_accuracy
from egg.language.prompts.evaluator_prompts import (
    build_evaluator_messages,
)
from torch import Value


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="eval/evaluator.log",
)


def compute_f1_score_nodes(gt: List, pred: List) -> float:
    gt_set = set(gt)
    pred_set = set(pred)
    correct_guesses = gt_set.intersection(pred_set)
    precision = len(correct_guesses) / len(pred_set) if pred_set else 0
    recall = len(correct_guesses) / len(gt_set) if gt_set else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    return f1_score


class EGGEvaluator:
    def __init__(self, llm_agent: OpenaiAgent, eval_data: Dict = {}):
        self.agent = llm_agent
        self.eval_data = eval_data
        self._qa_id = 0

    def reset(self):
        self.eval_data = {}
        self._qa_id = 0

    def get_id(self):
        self._qa_id += 1
        return self._qa_id

    def eval_qa(
        self,
        qa_gt: QAGroundTruth,
        gen_answer: str,
        optimal_subgraph: Optional[Dict],
        input_tokens: int,
        output_tokens: int,
    ) -> Tuple[str, float]:
        eval_response = "None"
        accuracy = 0.0
        if qa_gt.modality in [Modality.TEXT, "text"]:
            # If text, use llm to judge
            eval_messages = build_evaluator_messages(
                query=qa_gt.query, gt_answer=str(qa_gt.answer), gen_answer=gen_answer
            )
            eval_response, _, _ = self.agent.query_with_structured_output(
                llm_message=eval_messages,
                count_tokens=False,
                response_format=EVALUATOR_RESPONSE_FORMAT,
            )
            eval_response = json.loads(str(eval_response))
            accuracy = eval_response["accuracy"]
        elif qa_gt.modality in [Modality.BINARY, "binary"]:
            invalid_ans = False
            if str(gen_answer).lower() in ["yes", "true"]:
                gen_answer = "1"
            elif (gen_answer).lower() in ["no", "false"]:
                gen_answer = "0"
            else:
                logger.warning(f"Invalid binary answer: {gen_answer}")
                accuracy = 0.0
                invalid_ans = True
            assert isinstance(
                qa_gt.answer, bool
            ), f"GT answer for modality binary must be bool but got {qa_gt.answer}"
            if not invalid_ans:
                accuracy = 1.0 if int(qa_gt.answer) == int(gen_answer) else 0.0
        elif qa_gt.modality in [Modality.NODE, "node"]:
            try:
                gen_answer = literal_eval(str(gen_answer))
            except ValueError:
                gen_answer = [str(gen_answer)]
                logger.warning(
                    f"Gen answer for modality 'node' must be List, but got {gen_answer}"
                )
            if not isinstance(gen_answer, List):
                gen_answer = [str(gen_answer)]
                logger.warning(
                    f"Gen answer for modality 'node' must be List, but got {gen_answer}"
                )
            assert isinstance(
                qa_gt.answer, List
            ), f"GT answer for modality 'node' must be List, but got {qa_gt.answer}"
            accuracy = compute_f1_score_nodes(gt=qa_gt.answer, pred=gen_answer)
            logger.debug(f"Nodes GT {qa_gt.answer}")
            logger.debug(f"Nodes pred: {gen_answer}")
            logger.debug(f"F1-score is {accuracy}")
        elif qa_gt.modality in [Modality.TIME_POINT, "time_point"]:
            gt_answer = datetime.strptime(str(qa_gt.answer), "%Y-%m-%d %H:%M:%S")
            try:
                gen_time = datetime.strptime(str(gen_answer), "%Y-%m-%d %H:%M:%S")
                time_diff = abs((gen_time - gt_answer).total_seconds() / 60.0)
                logger.debug(f"Time diff is {time_diff}")
                accuracy = 1.0 if time_diff < 2.0 else 0.0
                time_diff = abs((gen_time - gt_answer).total_seconds() / 60.0)
                logger.debug(f"Time diff is {time_diff}")
                accuracy = 1.0 if time_diff < 2.0 else 0.0
            except ValueError:
                accuracy = 0.0
        else:
            logger.error(f"Invalid modality: {qa_gt.modality}")
            raise NotImplementedError
        self.eval_data.update(
            {
                self.get_id(): {
                    "query": qa_gt.query,
                    "gt_answer": qa_gt.answer,
                    "modality": (
                        qa_gt.modality.name.lower()
                        if isinstance(qa_gt.modality, Modality)
                        else qa_gt.modality
                    ),
                    "gen_answer": gen_answer,
                    "eval_response": eval_response,
                    "accuracy": accuracy,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "optimal_subgraph": optimal_subgraph,
                }
            }
        )
        return eval_response, accuracy

    def save_eval_data(self, output_file: str):
        with open(output_file, "w") as fp:
            json.dump(self.eval_data, fp)

    def load_eval_data(self, data_file: str):
        with open(data_file, "r") as fp:
            self.eval_data = json.load(fp)
