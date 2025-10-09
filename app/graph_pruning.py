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
from egg.language.llm import LLMAgent

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

egg.deserialize("./graph_gt.json")

# egg.gen_captions(llm_agent=agent)
# logger.info(egg.pretty_str())

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", type=str)
parser.add_argument("-m", "--modality", type=str)
parser.add_argument("--mini", action="store_true")
args = parser.parse_args()

current_time = datetime.datetime.now()
query = args.query

agent = LLMAgent(use_mini=args.mini)
egg_slicer = EGGSlicer(egg=egg)
processor = QueryProcessor(
    egg_slicer=egg_slicer,
    llm_agent=agent,
    retrieval_strategy=RetrievalStrategy.NO_EDGE,
)
phase_1_response, phase_2_response, phase_3_response = processor.process_query(
    args.query, args.modality
)
logger.info(f"Phase 1: {phase_1_response}")
logger.info(f"Phase 2: {phase_2_response}")
logger.info(f"Phase 3: {phase_3_response}")
