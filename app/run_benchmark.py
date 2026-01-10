#!/usr/bin/env python

import logging
import argparse
import json
import os
from tqdm import tqdm

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

qa_file = "/home/ros/data/egg_qa_remembr.csv"
qa_dataset = QADataset(qa_file=qa_file, egg=egg)

if "gpt" in args.model:
    llm_agent = OpenaiAgent(use_mini=args.mini, temperature=0.001)
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
current_time = "30th August 2025 23:59:00"
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
output_dir = f"trial_{args.trial}"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/eval_trial_{args.trial}_{args.strategy.lower()}_model_{args.model}_autocaption_{args.auto}_guided_{not args.unguided}.json"
if os.path.isfile(output_file):
    with open(output_file, "r") as f:
        benchmark_data = json.load(f)
        f.close()
        logger.info(f"Loading prev benchmark data from: {output_file}")
else:
    logger.info(
        f"No prev benchmark data from: {output_file}\nCreating new benchmark data file"
    )
    benchmark_data = {}

for id, qa_gt in enumerate(tqdm(qa_dataset.qa_ground_truth_list)):
    if str(id) not in benchmark_data.keys():
        _, _, gen_data, query_input_tokens, query_output_tokens = (
            processor.process_query(qa_gt.query, qa_gt.modality.name.lower())
        )
        gen_answer = json.loads(gen_data)["answer"]
        logger.debug(f"Query {id}: {qa_gt.query}")
        logger.debug(f"modality: {qa_gt.modality.name.lower()}")
        logger.debug(f"Gen answer: {gen_answer}")
        gen_data = {
            "query": qa_gt.query,
            "modality": qa_gt.modality.name.lower(),
            "gt_answer": qa_gt.answer,
            "gen_answer": gen_answer,
            "input_tokens": query_input_tokens,
            "output_tokens": query_output_tokens,
            "optimal_subgraph": str(processor.serialized_optimal_subgraph),
        }
        benchmark_data.update({id: gen_data})

        with open(output_file, "w+") as f:
            json.dump(benchmark_data, f)
            f.close()
    else:
        logger.info(f"Skipping query {id}")
total_input_tokens, total_output_tokens = processor.get_used_tokens()
logger.info(f"Input tokens: {total_input_tokens}")
logger.info(f"Output tokens: {total_output_tokens}")
logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}")
