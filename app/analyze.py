#!/usr/bin/env python3

import json
import logging
import argparse

from egg.utils.logger import getLogger
from egg.eval.analyzer import EGGAnalyzer
import numpy

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="app/analyze.log",
)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modality", type=str, default="all")
parser.add_argument("-f", "--file", type=str, default="./eval_pruning_unified_autocaption_False_guided_True.json")
args = parser.parse_args()

with open("./graph_gt.json") as fp:
    full_graph = json.load(fp)

analyzer = EGGAnalyzer(args.file)
if args.modality == "failure":
    failure_data = analyzer.get_failure_eval_data()
    for id, eval_sample in failure_data.items():
        logger.info(
            f"{id} - {eval_sample['query']}\n"
            + f"Accuracy: {eval_sample['accuracy']}\n"
            + f"Gen Answer: {eval_sample['gen_answer']}\n"
            + f"Eval Response: {eval_sample['eval_response']}\n"
            + f"Graph: {eval_sample['optimal_subgraph']}\n\n"
        )
else:
    for m in ["text", "binary", "node", "time", "all"]:
        modality_data = analyzer.get_eval_data_by_modality(modality=m)
        accuracy_list = []
        compression_list = []
        for id, eval_sample in modality_data.items():
            graph_percentage = len(str(eval_sample["optimal_subgraph"])) / len(str(full_graph))
            accuracy_list.append(eval_sample["accuracy"])
            compression_list.append(graph_percentage)
        logger.info(
            f"Average accuracy of modality {m}: {numpy.mean(accuracy_list)}"
        )
        if m == "all":
            logger.info(
                f"Average compression % of modality {m}: {100 - numpy.mean(compression_list) * 100}%"
            )
