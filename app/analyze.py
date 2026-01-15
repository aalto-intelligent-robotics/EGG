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
parser.add_argument("-r", "--results-file", type=str)
parser.add_argument("-g", "--graph-file", type=str, default="./graph_gt.json")
args = parser.parse_args()

with open(args.graph_file) as fp:
    fg = json.load(fp)

# NOTE: Convert event ids to int
event_ids = list(fg["nodes"]["event_nodes"].keys())
for e_id in event_ids:
    fg["nodes"]["event_nodes"][int(e_id)] = fg["nodes"]["event_nodes"].pop(e_id)
object_ids = list(fg["nodes"]["object_nodes"].keys())
for o_id in object_ids:
    fg["nodes"]["object_nodes"][int(o_id)] = fg["nodes"]["object_nodes"].pop(o_id)
edge_ids = list(fg["edges"]["event_object_edges"].keys())
for ed_id in edge_ids:
    fg["edges"]["event_object_edges"][int(ed_id)] = fg["edges"][
        "event_object_edges"
    ].pop(ed_id)

analyzer = EGGAnalyzer(args.results_file)
if args.modality == "failure":
    failure_data = analyzer.get_failure_eval_data()
    for id, eval_sample in failure_data.items():
        logger.info(
            f"{id} - {eval_sample['query']}\n"
            + f"GT Answer: {eval_sample['gt_answer']}\n"
            + f"Accuracy: {eval_sample['accuracy']}\n"
            + f"Gen Answer: {eval_sample['gen_answer']}\n"
            + f"Gen Answer Explanation: {eval_sample['gen_answer_explanation']}\n"
            + f"Eval Response: {eval_sample['eval_response']}\n"
            + f"Graph: {eval_sample['optimal_subgraph']}\n\n"
        )
else:
    for m in ["all", "text", "binary", "node", "time"]:
        modality_data = analyzer.get_eval_data_by_modality(modality=m)
        accuracy_list = []
        compression_list = []
        if m != "binary":
            for id, eval_sample in modality_data.items():
                graph_percentage = len(str(eval_sample["optimal_subgraph"])) / len(
                    str(fg)
                )
                accuracy_list.append(eval_sample["accuracy"])
                compression_list.append(graph_percentage)
            logger.info(
                f"Average accuracy of modality {m}: {numpy.mean(accuracy_list)}"
            )
        else:
            f1_score = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for id, eval_sample in modality_data.items():
                bool_gt_ans = bool(int(eval_sample["gt_answer"]))
                bool_gen_ans = bool(int(eval_sample["gen_answer"]))
                if bool_gt_ans == True and bool_gen_ans == True:
                    tp += 1
                elif bool_gt_ans == False and bool_gen_ans == True:
                    fp += 1
                elif bool_gt_ans == False and bool_gen_ans == False:
                    tn += 1
                elif bool_gt_ans == True and bool_gen_ans == False:
                    fn += 1
            if 2 * tp + fp + fn == 0:
                f1_score = 0
            else:
                f1_score = 2 * tp / (2 * tp + fp + fn)
            logger.info(f"F1 score of modality {m}: {f1_score}")
        if m == "all":
            logger.info(
                f"Average compression % of modality {m}: {100 - (numpy.mean(compression_list) * 100)}%"
            )
total_input_tokens, total_output_tokens = analyzer.get_token_usage()
logger.info(f"Total input tokens: {total_input_tokens}")
logger.info(f"Total output tokens: {total_output_tokens}")
logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}\n\n")
