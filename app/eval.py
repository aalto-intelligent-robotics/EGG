#!/usr/bin/env python

import os
import logging
import numpy as np
import argparse

from egg.eval.evaluator import EGGEvaluator
from egg.pruning.egg_slicer import EGGSlicer
from egg.pruning.query_processor import QueryProcessor
from egg.eval.dataset import QADataset
from egg.graph.spatial import SpatialComponents
from egg.graph.event import EventComponents
from egg.graph.egg import EGG
from egg.language.openai_agent import OpenaiAgent
from egg.language.ollama_agent import OllamaAgent
from egg.pruning.strategies import RetrievalStrategy
from egg.utils.logger import getLogger
from egg.utils.language_utils import get_eval_accuracy

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="app/eval.log",
)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--strategy", type=str, default="pruning_unified")
parser.add_argument("-a", "--auto", action="store_true")
parser.add_argument("-u", "--unguided", action="store_true")
parser.add_argument("--mini", action="store_true")
parser.add_argument("-t", "--trial", type=int, default=1)
parser.add_argument("--model", type=str, default="gpt4")
args = parser.parse_args()

spatial_graph = SpatialComponents()
event_graph = EventComponents()
egg = EGG(spatial_graph, event_graph)

if not args.auto:
    graph_file = "./graph_gt.json"
elif args.unguided:
    graph_file = "./graph_auto_unguided.json"
else:
    graph_file = "./graph_auto_guided.json"
egg.deserialize(json_file=graph_file)

# logger.info(egg.get_events())

qa_file = "/home/ros/data/egg_qa.csv"
qa_dataset = QADataset(qa_file=qa_file, egg=egg)

if "gpt" in args.model:
    llm_agent = OpenaiAgent(use_mini=args.mini)
else:
    llm_agent = OllamaAgent(model=args.model)

evaluator = EGGEvaluator(llm_agent=llm_agent)

if args.strategy.lower() == "full_unified":
    strategy = RetrievalStrategy.FULL_UNIFIED
elif args.strategy.lower() == "pruning_unified":
    strategy = RetrievalStrategy.PRUNING_UNIFIED
elif args.strategy.lower() == "pruning_unified_no_edge":
    strategy = RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE
elif args.strategy.lower() == "spatial":
    strategy = RetrievalStrategy.SPATIAL_ONLY
elif args.strategy.lower() == "event":
    strategy = RetrievalStrategy.EVENT_ONLY
elif args.strategy.lower() == "no_edge":
    strategy = RetrievalStrategy.NO_EDGE
else:
    raise AssertionError(
        "Valid strategies are: ['pruning_unified', 'pruning_unified_no_edge', 'spatial', 'event', 'no_edge', 'full_unified']"
    )

egg_slicer = EGGSlicer(egg=egg)
processor = QueryProcessor(
    egg_slicer=egg_slicer,
    llm_agent=llm_agent,
    retrieval_strategy=strategy,
)

accuracy = []
logger.info(f"graph file used: {graph_file}")
logger.info(f"Auto: {args.auto}")
logger.info(f"Strategy: {args.strategy.lower()}")
for qa_gt in qa_dataset.qa_ground_truth_list:
    _, _, gen_answer, _, _ = processor.process_query(qa_gt.query, qa_gt.modality.name.lower())

    eval_response = evaluator.eval_qa(
        qa_gt=qa_gt,
        gen_answer=gen_answer,
        optimal_subgraph=processor.serialized_optimal_subgraph,
    )
    logger.debug(
        f"Eval Response: {eval_response}\n"
        + f"Query: {qa_gt.query}\n"
        + f"GT Answer: {qa_gt.answer}\n"
        + f"Gen answer: {gen_answer}\n"
    )
    accuracy.append(get_eval_accuracy(eval_response))

mean_accuracy = np.mean(accuracy)
logger.info(f"Mean Accuracy: {mean_accuracy}")
total_input_tokens, total_output_tokens = processor.get_used_tokens()
logger.info(f"Input tokens: {total_input_tokens}")
logger.info(f"Output tokens: {total_output_tokens}")
logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}")

os.makedirs(f"trial_{args.trial}", exist_ok=True)
evaluator.save_eval_data(
    f"trial_{args.trial}/eval_{args.strategy.lower()}_autocaption_{args.auto}_guided_{not args.unguided}_trial_{args.trial}.json"
)
