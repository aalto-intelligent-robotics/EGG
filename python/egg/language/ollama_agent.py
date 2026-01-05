from typing import Optional, Sequence
from langchain_ollama import ChatOllama
from langchain_core.callbacks import get_usage_metadata_callback
import logging

from egg.utils.logger import getLogger
from egg.language.llm import LLMAgent

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/ollama_agent.log",
)


class OllamaAgent(LLMAgent):
    def __init__(
        self,
        model: str = "command-r",
        num_ctx: int = 128000,
        *args,
        **kwargs,
    ):
        super(OllamaAgent, self).__init__(*args, **kwargs)
        self._model_name = model
        self._model = ChatOllama(
            model=self._model_name,
            temperature=self.temperature,
            num_ctx=num_ctx,
        )
        logger.info(f"🧠 Using {self._model.model}")

    def query(self, llm_message: Sequence, count_tokens: bool = False) -> Optional[str]:
        with get_usage_metadata_callback() as cb:
            response = self._model.invoke(llm_message)
            if count_tokens:
                self.total_input_tokens += cb.usage_metadata[self._model_name][
                    "input_tokens"
                ]
                self.total_input_tokens += cb.usage_metadata[self._model_name][
                    "output_tokens"
                ]
        return str(response.content)
