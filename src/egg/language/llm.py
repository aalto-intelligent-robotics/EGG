from abc import ABC, abstractmethod
import logging
from typing import Sequence, Optional, Tuple

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/llm.log",
)


class LLMAgent(ABC):
    def __init__(
        self,
        temperature: float = 0,
    ):
        self._temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def temperature(self):
        return self._temperature

    @abstractmethod
    def query(self, llm_message: Sequence, count_tokens: bool = False) -> Tuple[Optional[str], int, int]:
        ...
