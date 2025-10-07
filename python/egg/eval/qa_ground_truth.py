from dataclasses import dataclass
from numpy.typing import NDArray
from datetime import datetime
from typing import Union, List
from enum import Enum
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="eval/qa_ground_truth.log",
)


class Modality(Enum):

    TEXT = 0
    NODE = 1
    BINARY = 2
    TIME_POINT = 3
    TIME_DURATION = 4
    TIME_INTERVAL = 5
    POSITION = 6


@dataclass
class QAGroundTruth:
    query: str
    modality: Modality
    answer: Union[str, NDArray, List[datetime], List[str], bool]

    def pretty_str(self) -> str:
        return (
            f"Query: {self.query}\n"
            + f"Modality: {self.modality}\n"
            + f"Answer: {self.answer}\n"
        )
