#!/usr/bin/env python

import logging
import argparse
import pickle

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

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="app/run_benchmark.log",
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
current_time = "30 August 2025 23:59:59"
processor = QueryProcessor(
    egg_slicer=egg_slicer,
    current_time=current_time,
    llm_agent=llm_agent,
    retrieval_strategy=strategy,
)

accuracy = []
logger.info(f"graph file used: {graph_file}")
logger.info(f"Auto: {args.auto}")
logger.info(f"Strategy: {args.strategy.lower()}")
benchmark_data = {}
for id, qa_gt in enumerate(qa_dataset.qa_ground_truth_list):
    _, _, gen_answer = processor.process_query(qa_gt.query, qa_gt.modality.name.lower())
    gen_data = {
        "query": qa_gt.query,
        "modality": qa_gt.modality,
        "gt_answer": qa_gt.answer,
        "gen_answer": gen_answer,
        "optimal_subgraph": processor.serialized_optimal_subgraph,
    }
    benchmark_data.update({id: gen_data})
total_input_tokens, total_output_tokens = processor.get_used_tokens()
logger.info(f"Input tokens: {total_input_tokens}")
logger.info(f"Output tokens: {total_output_tokens}")
logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}")

with open(f"trial_{args.trial}/eval_{args.strategy.lower()}_model_{args.model}_autocaption_{args.auto}_guided_{not args.unguided}_trial_{args.trial}.pkl", "wb") as f:
    pickle.dump(benchmark_data, f)
    f.close()
