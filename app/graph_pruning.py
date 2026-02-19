#!/usr/bin/env python3

import datetime
import logging
import argparse

from egg.graph.spatial import SpatialComponents
from egg.graph.event import EventComponents
from egg.graph.egg import EGG
from egg.pruning.strategies import RetrievalStrategy
from egg.utils.logger import getLogger
from egg.pruning.egg_slicer import EGGSlicer
from egg.pruning.query_processor import QueryProcessor
from egg.language.openai_agent import OpenaiAgent
from egg.language.ollama_agent import OllamaAgent

viz_elements = []

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="build_graph.log",
)
spatial_graph = SpatialComponents()
event_graph = EventComponents()
egg = EGG(spatial_graph, event_graph)

egg.deserialize("./graph_auto_guided.json")

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", type=str)
parser.add_argument("-m", "--modality", type=str)
parser.add_argument("--mini", action="store_true")
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--aalto", action="store_true")
args = parser.parse_args()

current_time = "30th August 2025 23:59:00"
# current_time = str(datetime.datetime.now())
query = args.query

if "gpt" in args.model:
    llm_agent = OpenaiAgent(
        use_mini=args.mini, temperature=0.001, model_name=args.model, aalto=args.aalto
    )
else:
    llm_agent = OllamaAgent(model=args.model)

egg_slicer = EGGSlicer(egg=egg)
processor = QueryProcessor(
    egg_slicer=egg_slicer,
    current_time=current_time,
    llm_agent=llm_agent,
    retrieval_strategy=RetrievalStrategy.PRUNING_UNIFIED,
)
phase_1_response, phase_2_response, phase_3_response, _, _ = processor.process_query(
    args.query, args.modality
)
logger.info(f"Phase 1: {phase_1_response}")
logger.info(f"Phase 2: {phase_2_response}")
logger.info(f"Phase 3: {phase_3_response}")
