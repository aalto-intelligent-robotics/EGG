from copy import deepcopy
from typing import List, Dict
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/prompts/evaluator_prompt.log",
)

EVALUATOR_PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """
        You are a judge for a QA system that answers human questions in a natural way.
        The answers could be give the following modalities:
        - node: return the list of node names of the object nodes that responds to the query. e.g., [mug_0, bowl_0, faucet_3]. Your answer could contain only one or multiple node names.
        - text: Return the answer in natural language responding to the query.
        - binary: Return either True or False.
        - time_point: Return the answer in the form of a point in time return the timestamp in the format yyyy-mm-dd hh:mm:ss
        - time_range: Return the answer in the form of a time range, return in the form yyyy-mm-dd hh:mm:ss - yyyy-mm-dd hh:mm:ss (start timestamp - end timestamp)
        - time_duration: Return the answer in the form hh:mm:ss.
        - position: Return the answer in the form of a point in space, return the answer in the form of a 3D coordinate [x, y, z].

        You are given a query, a ground truth answer, and a generated answer. You need to evaluate how accurate the generated answer is compared to the ground truth answer on a scale of 0 to 1.

        """,
    },
    {
        "role": "user",
        "content": """
            Here is the query: {query}.
            Here is the ground truth answer: {gt_answer}
            Here is the generated answer: {gen_answer}
            Return your response in this JSON format:
            [
                {{
                    "accuracy": <How semantically similar is the generated answer to the ground truth answer on a scale of 0 to 1>
                    "explanation": <explanation for the accuracy evaluation>
                }}
            ]
            """,
    },
]


def build_evaluator_messages(query: str, gt_answer: str, gen_answer: str) -> List[Dict]:
    messages = deepcopy(EVALUATOR_PROMPT_TEMPLATE)
    messages[-1]["content"] = messages[-1]["content"].format(
        query=query, gt_answer=gt_answer, gen_answer=gen_answer
    )
    return messages
