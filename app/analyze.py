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

# NOTE: Convert event ids to 0
event_ids = list(fg["nodes"]["event_nodes"].keys())
for e_id in event_ids:
    fg["nodes"]["event_nodes"][int(e_id)] = fg["nodes"]["event_nodes"].pop(e_id)
object_ids = list(fg["nodes"]["object_nodes"].keys())
for o_id in object_ids:
    fg["nodes"]["object_nodes"][int(o_id)] = fg["nodes"]["object_nodes"].pop(o_id)
edge_ids = list(fg["edges"]["event_object_edges"].keys())
for ed_id in edge_ids:
    fg["edges"]["event_object_edges"][int(ed_id)] = fg["edges"]["event_object_edges"].pop(ed_id)

# NOTE: Involved object ID set by edges, remove to avoid confusion.
for e_id in fg["nodes"]["event_nodes"].keys():
    fg["nodes"]["event_nodes"][e_id].pop("involved_object_ids")

analyzer = EGGAnalyzer(args.results_file)
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
    for m in ["all", "text", "binary", "node", "time"]:
        modality_data = analyzer.get_eval_data_by_modality(modality=m)
        accuracy_list = []
        compression_list = []
        for id, eval_sample in modality_data.items():
            graph_percentage = len(str(eval_sample["optimal_subgraph"])) / len(str(fg))
            accuracy_list.append(eval_sample["accuracy"])
            compression_list.append(graph_percentage)
        logger.info(f"Average accuracy of modality {m}: {numpy.mean(accuracy_list)}")
        if m == "all":
            logger.info(
                f"Average compression % of modality {m}: {100 - (numpy.mean(compression_list) * 100)}%"
            )
total_input_tokens, total_output_tokens = analyzer.get_token_usage()
logger.info(f"Total input tokens: {total_input_tokens}")
logger.info(f"Total output tokens: {total_output_tokens}")
logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}")
