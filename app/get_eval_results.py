#!/usr/bin/env python

import numpy as np
from typing import Dict
import json
from tqdm import tqdm
import logging
import argparse

from egg.eval.dataset import QAGroundTruth
from egg.language.openai_agent import OpenaiAgent
from egg.eval.evaluator import EGGEvaluator
from egg.utils.logger import getLogger
from egg.utils.language_utils import get_eval_accuracy

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="app/get_eval_results.log",
)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default="./eval_trial_1_remembr_model_gpt-4o.json")
args = parser.parse_args()

with open(args.file, "r") as f:
    benchmark_data: Dict = json.load(f)

llm_agent = OpenaiAgent()
evaluator = EGGEvaluator(llm_agent=llm_agent)

accuracy_list = []
for result in tqdm(benchmark_data.values()):
    if "optimal_subgraph" in result.keys():
        optimal_subgraph = result["optimal_subgraph"]
    else:
        optimal_subgraph = None
    qa_gt = QAGroundTruth(
        query=result["query"], modality=result["modality"], answer=result["gt_answer"]
    )
    eval_response, accuracy = evaluator.eval_qa(
        qa_gt=qa_gt,
        gen_answer=result["gen_answer"],
        optimal_subgraph=optimal_subgraph,
        input_tokens=result["input_tokens"] if "remembr" not in args.file else 0,
        output_tokens=result["output_tokens"] if "remembr" not in args.file else 0,
    )
    accuracy_list.append(accuracy)
mean_accuracy = np.mean(accuracy_list)
logger.info(f"Mean Accuracy: {mean_accuracy}")
evaluator.save_eval_data(args.file.replace(".json", "_eval_results.json"))
