from copy import deepcopy
import sys
from typing import Optional, Tuple
import datetime
import logging
from egg.utils.language_utils import remove_explanation_and_convert

from egg.pruning.egg_slicer import EGGSlicer
from egg.language.prompts.pruning_unified_prompts import (
    PRUNING_UNIFIED_SYSTEM_PROMPT,
    PRUNING_UNIFIED_PHASE_1_PROMPT,
    PRUNING_UNIFIED_PHASE_2_PROMPT,
    PRUNING_UNIFIED_PHASE_3_PROMPT_TEMPLATE,
)
from egg.language.prompts.pruning_unified_no_edge_prompts import (
    PRUNING_UNIFIED_NO_EDGE_SYSTEM_PROMPT,
    PRUNING_UNIFIED_NO_EDGE_PHASE_1_PROMPT,
    PRUNING_UNIFIED_NO_EDGE_PHASE_2_PROMPT,
    PRUNING_UNIFIED_NO_EDGE_PHASE_3_PROMPT_TEMPLATE,
)
from egg.language.prompts.full_unified_prompts import (
    FULL_UNIFIED_SYSTEM_PROMPT,
    FULL_UNIFIED_USER_PROMPT,
)
from egg.language.prompts.spatial_only_prompts import (
    SPATIAL_ONLY_SYSTEM_PROMPT,
    SPATIAL_ONLY_USER_PROMPT,
)
from egg.language.prompts.event_only_prompts import (
    EVENT_ONLY_SYSTEM_PROMPT,
    EVENT_ONLY_USER_PROMPT,
)
from egg.language.prompts.no_edge_prompts import (
    NO_EDGE_SYSTEM_PROMPT,
    NO_EDGE_USER_PROMPT,
)
from egg.utils.timestamp import datetime_to_ns
from egg.language.llm import LLMAgent
from egg.utils.logger import getLogger
from egg.pruning.strategies import RetrievalStrategy

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="pruning/query_processor.log",
)


class QueryProcessor:
    """
    A class that processes queries to manipulate EGG using EGGSlicer, leveraging
    large language models (LLM) to perform various information retrieval strategies.
    """
    def __init__(
        self,
        egg_slicer: EGGSlicer,
        llm_agent: LLMAgent,
        current_time: str = str(datetime.datetime.now()),
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.PRUNING_UNIFIED,
    ):
        """
        Initializes the QueryProcessor with an EGGSlicer and language model agent.

        :param egg_slicer: EGGSlicer instance to manage EGG processing.
        :type egg_slicer: EGGSlicer
        :param llm_agent: The language model agent for information retrieval.
        :type llm_agent: LLMAgent
        :param current_time: Current time for use in query processing.
        :type current_time: str
        :param retrieval_strategy: Strategy for information retrieval.
        :type retrieval_strategy: RetrievalStrategy
        """
        self.egg_slicer = egg_slicer
        self.retrieval_strategy = retrieval_strategy
        self.current_time = current_time
        self.agent = llm_agent
        self.min_timestamp = 0
        self.max_timestamp = sys.maxsize
        self.serialized_optimal_subgraph = {}
        self.reset()

    def reset(self):
        """
        Resets the processor by restoring the pruned EGG and initializing prompts.
        """
        self.egg_slicer.reset_pruned_egg()
        self.messages = []
        if self.retrieval_strategy == RetrievalStrategy.PRUNING_UNIFIED:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(PRUNING_UNIFIED_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(PRUNING_UNIFIED_PHASE_1_PROMPT),
            }
            self.phase_2_prompt = {
                "role": "user",
                "content": deepcopy(PRUNING_UNIFIED_PHASE_2_PROMPT),
            }
            self.phase_3_prompt = deepcopy(PRUNING_UNIFIED_PHASE_3_PROMPT_TEMPLATE)
        elif self.retrieval_strategy == RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(PRUNING_UNIFIED_NO_EDGE_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(PRUNING_UNIFIED_NO_EDGE_PHASE_1_PROMPT),
            }
            self.phase_2_prompt = {
                "role": "user",
                "content": deepcopy(PRUNING_UNIFIED_NO_EDGE_PHASE_2_PROMPT),
            }
            self.phase_3_prompt = deepcopy(
                PRUNING_UNIFIED_NO_EDGE_PHASE_3_PROMPT_TEMPLATE
            )
        elif self.retrieval_strategy == RetrievalStrategy.SPATIAL_ONLY:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(SPATIAL_ONLY_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(SPATIAL_ONLY_USER_PROMPT),
            }
        elif self.retrieval_strategy == RetrievalStrategy.EVENT_ONLY:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(EVENT_ONLY_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(EVENT_ONLY_USER_PROMPT),
            }
        elif self.retrieval_strategy == RetrievalStrategy.NO_EDGE:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(NO_EDGE_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(NO_EDGE_USER_PROMPT),
            }
        elif self.retrieval_strategy == RetrievalStrategy.FULL_UNIFIED:
            self.system_prompt = {
                "role": "system",
                "content": deepcopy(FULL_UNIFIED_SYSTEM_PROMPT),
            }
            self.phase_1_prompt = {
                "role": "user",
                "content": deepcopy(FULL_UNIFIED_USER_PROMPT),
            }
        else:
            raise AssertionError(
                f"Invalid retrieval strategy: {self.retrieval_strategy}"
            )

    def process_query(
        self,
        query: str,
        modality: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Processes a query to manipulate EGG and retrieve information based on the current strategy.

        :param query: The query string to process.
        :type query: str
        :param modality: The modality of the query (e.g., spatial or event).
        :type modality: str
        :returns: Tuple of phase 1, phase 2, and phase 3 responses.
        :rtype: Tuple[Optional[str], Optional[str], str]
        """
        self.reset()
        self.system_prompt["content"] = self.system_prompt["content"].format(
            current_time=self.current_time,
            query=query,
            modality=modality,
        )

        self.messages.append(self.system_prompt)

        if self.retrieval_strategy in [
            RetrievalStrategy.PRUNING_UNIFIED,
            RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE,
        ]:

            phase_1_response_content = self.phase_1()
            phase_2_response_content = self.phase_2()
            if self.retrieval_strategy == RetrievalStrategy.PRUNING_UNIFIED:
                self.messages[0] = {
                    "role": "system",
                    "content": deepcopy(PRUNING_UNIFIED_SYSTEM_PROMPT),
                }
            elif self.retrieval_strategy == RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE:
                self.messages[0] = {
                    "role": "system",
                    "content": deepcopy(PRUNING_UNIFIED_NO_EDGE_SYSTEM_PROMPT),
                }
            phase_3_response_content = self.phase_3(query=query, modality=modality)
            return (
                phase_1_response_content,
                phase_2_response_content,
                phase_3_response_content,
            )
        elif self.retrieval_strategy == RetrievalStrategy.SPATIAL_ONLY:
            response_content = self.spatial_only()
            return None, None, response_content
        elif self.retrieval_strategy == RetrievalStrategy.EVENT_ONLY:
            response_content = self.event_only()
            return None, None, response_content
        elif self.retrieval_strategy == RetrievalStrategy.NO_EDGE:
            response_content = self.no_edge()
            return None, None, response_content
        else:
            response_content = self.full_graph()
            return None, None, response_content

    def add_response(self, response: str):
        """
        Adds a response to the list of messages.

        :param response: The response content to add.
        :type response: str
        """
        self.messages.append({"role": "assistant", "content": response})

    def set_phase_1_message(self):
        """
        Sets the message for phase 1 based on the retrieval strategy.
        """
        locations = self.egg_slicer.get_locations()
        self.phase_1_prompt["content"] = self.phase_1_prompt["content"].format(
            locations=locations,
        )
        self.messages.append(self.phase_1_prompt)

    def phase_1(self) -> str:
        """
        Executes phase 1 of the query process.

        :returns: Phase 1 response content.
        :rtype: str
        """
        self.set_phase_1_message()
        phase_1_response_content = self.agent.send_query(
            self.messages, count_tokens=True
        )
        assert phase_1_response_content is not None
        logger.debug(phase_1_response_content)
        phase_1_response_dict = remove_explanation_and_convert(phase_1_response_content)
        assert phase_1_response_dict is not None
        if phase_1_response_dict[0]["start_year"] != 0:
            self.min_timestamp = datetime_to_ns(
                datetime.datetime(
                    phase_1_response_dict[0]["start_year"],
                    phase_1_response_dict[0]["start_month"],
                    phase_1_response_dict[0]["start_day"],
                    phase_1_response_dict[0]["start_hour"],
                    phase_1_response_dict[0]["start_minute"],
                )
            )
        else:
            self.min_timestamp = 0
        if phase_1_response_dict[0]["end_year"] != "inf":
            self.max_timestamp = datetime_to_ns(
                datetime.datetime(
                    phase_1_response_dict[0]["end_year"],
                    phase_1_response_dict[0]["end_month"],
                    phase_1_response_dict[0]["end_day"],
                    phase_1_response_dict[0]["end_hour"],
                    phase_1_response_dict[0]["end_minute"],
                )
            )
        else:
            self.max_timestamp = sys.maxsize
        self.egg_slicer.prune_graph_by_time_range(
            min_timestamp=self.min_timestamp,
            max_timestamp=self.max_timestamp,
        )
        self.egg_slicer.prune_graph_by_location(
            locations_list=phase_1_response_dict[0]["locations"]
        )
        self.add_response(phase_1_response_content)
        return phase_1_response_content

    def set_phase_2_message(self):
        """
        Sets the message for phase 2 based on the retrieval strategy and current graph state.
        """
        objects = self.egg_slicer.pruned_egg.get_objects()
        events = self.egg_slicer.pruned_egg.get_events()

        if self.retrieval_strategy in [
            RetrievalStrategy.PRUNING_UNIFIED,
            RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE,
        ]:
            self.phase_2_prompt["content"] = self.phase_2_prompt["content"].format(
                objects=objects,
                events=events,
            )
        self.messages.append(self.phase_2_prompt)

    def phase_2(self) -> str:
        """
        Executes phase 2 of the query process, refining results using objects and events.

        :returns: Phase 2 response content.
        :rtype: str
        """
        self.set_phase_2_message()
        phase_2_response_content = self.agent.send_query(
            self.messages, count_tokens=True
        )
        assert phase_2_response_content is not None
        logger.debug(phase_2_response_content)
        phase_2_response_dict = remove_explanation_and_convert(phase_2_response_content)
        assert phase_2_response_dict is not None
        if self.retrieval_strategy in [
            RetrievalStrategy.PRUNING_UNIFIED,
            RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE,
        ]:
            relevant_object_ids = phase_2_response_dict[0]["object_nodes"]
            relevant_event_ids = phase_2_response_dict[0]["event_nodes"]

            self.egg_slicer.reset_pruned_egg()
            self.egg_slicer.merge_events_and_objects(
                object_ids=relevant_object_ids, event_ids=relevant_event_ids
            )
            self.egg_slicer.prune_graph_by_time_range(
                self.min_timestamp, self.max_timestamp
            )

            self.add_response(phase_2_response_content)
        return phase_2_response_content

    def set_phase_3_message(self, query: str, modality: str):
        """
        Sets the message for phase 3, tailoring the prompts to the query and available data.

        :param query: The query string influencing the current phase.
        :type query: str
        :param modality: The modality of the query (e.g., spatial or event).
        :type modality: str
        """
        self.messages = self.phase_3_prompt
        subgraph = self.egg_slicer.pruned_egg.serialize()
        for event_id in subgraph["nodes"]["event_nodes"].keys():
            subgraph["nodes"]["event_nodes"][event_id].pop("involved_object_ids")
        logger.debug(f"Optimal subgraph: {self.egg_slicer.pruned_egg.pretty_str()}")
        self.messages[0]["content"] = self.messages[0]["content"].format(
            current_time=self.current_time, query=query, modality=modality
        )
        if self.retrieval_strategy == RetrievalStrategy.PRUNING_UNIFIED_NO_EDGE:
            subgraph.pop("edges")
            # for event_id in subgraph["nodes"]["event_nodes"].keys():
            #     subgraph["nodes"]["event_nodes"][event_id].pop("involved_object_ids")
        self.serialized_optimal_subgraph = subgraph
        logger.debug(f"Optimal subgraph serialized: {self.serialized_optimal_subgraph}")
        self.messages[-1]["content"] = self.messages[-1]["content"].format(
            subgraph=self.serialized_optimal_subgraph
        )

    def phase_3(self, query: str, modality: str) -> str:
        """
        Executes phase 3 of the query process, finalizing the optimal subgraph serialization.

        :param query: The query string influencing the results.
        :type query: str
        :param modality: The modality of the query.
        :type modality: str
        :returns: Phase 3 response content.
        :rtype: str
        """
        self.set_phase_3_message(query=query, modality=modality)
        phase_3_response_content = self.agent.send_query(
            self.messages, count_tokens=True
        )
        assert phase_3_response_content is not None
        logger.debug(phase_3_response_content)
        return phase_3_response_content

    def full_graph(self) -> str:
        """
        Retrieves and processes the full graph, removing object involvement to focus on events.

        :returns: Response content from full graph processing.
        :rtype: str
        """
        full_graph_data = self.egg_slicer.egg.serialize()
        self.phase_1_prompt["content"] = self.phase_1_prompt["content"].format(
            full_graph=full_graph_data
        )
        for event_id in full_graph_data["nodes"]["event_nodes"].keys():
            full_graph_data["nodes"]["event_nodes"][event_id].pop("involved_object_ids")
        logger.debug(f"Graph data: {full_graph_data}")
        self.messages = [self.system_prompt, self.phase_1_prompt]
        response_content = self.agent.send_query(self.messages, count_tokens=True)
        assert response_content is not None
        logger.debug(response_content)
        self.serialized_optimal_subgraph = full_graph_data
        return response_content

    def spatial_only(self) -> str:
        """
        Execute SPATIAL_ONLY strategy: EGG with only spatial component.

        :returns: Response content from spatial processing.
        :rtype: str
        """
        full_graph_data = self.egg_slicer.egg.serialize()
        full_graph_data.pop("edges")
        full_graph_data["nodes"].pop("event_nodes")
        self.phase_1_prompt["content"] = self.phase_1_prompt["content"].format(
            full_graph=full_graph_data
        )
        logger.debug(f"Graph data: {full_graph_data}")
        self.messages = [self.system_prompt, self.phase_1_prompt]
        response_content = self.agent.send_query(self.messages, count_tokens=True)
        assert response_content is not None
        logger.debug(response_content)
        self.serialized_optimal_subgraph = full_graph_data
        return response_content

    def event_only(self) -> str:
        """
        Execute EVENT_ONLY strategy: EGG with only event component.

        :returns: Response content from event processing.
        :rtype: str
        """
        full_graph_data = self.egg_slicer.egg.serialize()
        full_graph_data.pop("edges")
        full_graph_data["nodes"].pop("object_nodes")
        for event_id in full_graph_data["nodes"]["event_nodes"].keys():
            full_graph_data["nodes"]["event_nodes"][event_id].pop("involved_object_ids")
        self.phase_1_prompt["content"] = self.phase_1_prompt["content"].format(
            full_graph=full_graph_data
        )
        logger.debug(f"Graph data: {full_graph_data}")
        self.messages = [self.system_prompt, self.phase_1_prompt]
        response_content = self.agent.send_query(self.messages, count_tokens=True)
        assert response_content is not None
        logger.debug(response_content)
        self.serialized_optimal_subgraph = full_graph_data
        return response_content

    def no_edge(self) -> str:
        """
        Execute NO_EDGE strategy: EGG without edges connecting spatial and event components.

        :returns: Response content from edge-free processing.
        :rtype: str
        """
        full_graph_data = self.egg_slicer.egg.serialize()
        full_graph_data.pop("edges")
        for event_id in full_graph_data["nodes"]["event_nodes"].keys():
            full_graph_data["nodes"]["event_nodes"][event_id].pop("involved_object_ids")
        self.phase_1_prompt["content"] = self.phase_1_prompt["content"].format(
            full_graph=full_graph_data
        )
        logger.debug(f"Graph data: {full_graph_data}")
        self.messages = [self.system_prompt, self.phase_1_prompt]
        response_content = self.agent.send_query(self.messages, count_tokens=True)
        assert response_content is not None
        logger.debug(response_content)
        self.serialized_optimal_subgraph = full_graph_data
        return response_content

    def get_used_tokens(self) -> Tuple[int, int]:
        """
        Retrieves the number of input and output tokens used by the LLM agent.

        :returns: Tuple of input and output token counts.
        :rtype: Tuple[int, int]
        """
        return self.agent.total_input_tokens, self.agent.total_output_tokens
