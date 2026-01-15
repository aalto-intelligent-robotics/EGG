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

        You are given a query, a ground truth answer, and a generated answer. 

        You need to evaluate how semantically accurate the generated answer is compared to the ground truth answer on a scale of 0 to 1.
        """,
    },
    {
        "role": "user",
        "content": """
            Here is the query: {query}.
            Here is the ground truth answer: {gt_answer}
            Here is the generated answer: {gen_answer}
            """,
    },
]


def build_evaluator_messages(query: str, gt_answer: str, gen_answer: str) -> List[Dict]:
    messages = deepcopy(EVALUATOR_PROMPT_TEMPLATE)
    messages[-1]["content"] = messages[-1]["content"].format(
        query=query, gt_answer=gt_answer, gen_answer=gen_answer
    )
    return messages
