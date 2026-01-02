import logging
from typing import Dict
import json

from egg.eval.dataset import QADataset
from egg.eval.qa_ground_truth import QAGroundTruth
from egg.graph.egg import EGG
from egg.utils.logger import getLogger
from egg.utils.language_utils import get_eval_accuracy
from egg.language.llm import LLMAgent
from egg.language.prompts.evaluator_prompts import build_evaluator_messages


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="eval/evaluator.log",
)


class EGGEvaluator:
    def __init__(
        self, dataset: QADataset, egg: EGG, llm_agent: LLMAgent, eval_data: Dict = {}
    ):
        self.dataset = dataset
        self.egg = egg
        self.agent = llm_agent
        self.eval_data = eval_data
        self._qa_id = 0

    def reset(self):
        self.eval_data = {}
        self._qa_id = 0

    def get_id(self):
        self._qa_id += 1
        return self._qa_id

    def eval_qa(self, qa_gt: QAGroundTruth, gen_answer: str, optimal_subgraph: Dict) -> str:
        eval_messages = build_evaluator_messages(
            query=qa_gt.query, gt_answer=str(qa_gt.answer), gen_answer=gen_answer
        )
        eval_response = self.agent.query(
            llm_message=eval_messages, count_tokens=False
        )
        self.eval_data.update(
            {
                self.get_id(): {
                    "query": qa_gt.query,
                    "gt_answer": qa_gt.answer,
                    "modality": qa_gt.modality.name.lower(),
                    "gen_answer": gen_answer,
                    "eval_response": str(eval_response),
                    "accuracy": get_eval_accuracy(str(eval_response)),
                    "optimal_subgraph": optimal_subgraph,
                }
            }
        )
        return str(eval_response)

    def save_eval_data(self, output_file: str):
        with open(output_file, "w") as fp:
            json.dump(self.eval_data, fp)

    def load_eval_data(self, data_file:str):
        with open(data_file, "r") as fp:
            self.eval_data = json.load(fp)
