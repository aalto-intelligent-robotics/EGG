import os
from typing import Sequence, Optional, Tuple, Dict
import httpx
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
import tiktoken
import logging

from egg.utils.logger import getLogger
from egg.language.llm import LLMAgent


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/openai_agent.log",
)


class OpenaiAgent(LLMAgent):
    def __init__(
        self,
        use_gpt4: bool = True,
        use_mini: bool = False,
        aalto: bool = False,
        model_name = "gpt-4o",
        *args,
        **kwargs,
    ):
        super(OpenaiAgent, self).__init__(*args, **kwargs)
        self.aalto = aalto
        if self.aalto:
            if use_gpt4:
                if use_mini:
                    base_url = "https://aalto-openai-apigw.azure-api.net/v1/openai/deployments/gpt-4o-mini-2024-07-18"
                    logger.info(f"🧠 Using GPT4o-mini from {base_url}")
                else:
                    base_url = "https://aalto-openai-apigw.azure-api.net/v1/openai/deployments/gpt-4o-2024-11-20"
                    logger.info(f"🧠 Using GPT4 from {base_url}")
                openai_endpoint_url = "/chat/completions"
            else:
                base_url = "https://aalto-openai-apigw.azure-api.net"
                openai_endpoint_url = "/v1/chat/gpt-35-turbo-1106"
                logger.info(f"🧠 Using GPT3.5")

            # Set API key in terminal: export AALTO_OPENAI_API_KEY=""
            api_key = os.environ.get("AALTO_OPENAI_API_KEY")
            assert (
                api_key is not None
            ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
            """
            Rewrite the base path with Aalto mappings
            For all endpoints see https://www.aalto.fi/en/services/azure-openai#6-available-api-s
            """

            def update_base_url(request: httpx.Request) -> None:
                if request.url.path == "/chat/completions":
                    request.url = request.url.copy_with(path=openai_endpoint_url)

            self._model = OpenAI(
                base_url=base_url,
                api_key="False",  # API key not used, and rather set below
                default_headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                },
                http_client=httpx.Client(event_hooks={"request": [update_base_url]}),
            )
        else:
            self.model_name = model_name
            self._model = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

    def query(
        self,
        llm_message: Sequence,
        count_tokens: bool = False,
    ) -> Tuple[Optional[str], int, int]:
        # Send query
        if self.aalto:
            model_name = "no_effect"
        else:
            model_name = self.model_name
        completion = self._model.chat.completions.create(
            model=model_name,  # the model variable must be set, but has no effect, model selection done with URL
            messages=llm_message,
            temperature=self.temperature,
        )
        # Get Content of the response
        response_content = completion.choices[0].message.content
        if response_content is None:
            return response_content, 0, 0

        # Count tokens
        encoding_name = "cl100k_base"  # For GPT-3.5-turbo-1106 and GPT-4o
        encoding = tiktoken.get_encoding(encoding_name)

        input_tokens = 0
        output_tokens = 0
        if count_tokens:
            for message in llm_message:
                input_tokens += len(encoding.encode(message["content"]))

            output_tokens = len(encoding.encode(response_content))

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

        return response_content, input_tokens, output_tokens

    def query_with_structured_output(
        self,
        response_format: ResponseFormat,
        llm_message: Sequence,
        count_tokens: bool = False,
    ) -> Tuple[Optional[str], int, int]:
        # Send query
        if self.aalto:
            model_name = "no_effect"
        else:
            model_name = self.model_name
        completion = self._model.chat.completions.create(
            model=model_name,  # the model variable must be set, but has no effect, model selection done with URL
            messages=llm_message,
            temperature=self.temperature,
            response_format=response_format,
        )
        # Get Content of the response
        response_content = completion.choices[0].message.content
        if response_content is None:
            return response_content, 0, 0

        # Count tokens
        encoding_name = "cl100k_base"  # For GPT-3.5-turbo-1106 and GPT-4o
        encoding = tiktoken.get_encoding(encoding_name)

        input_tokens = 0
        output_tokens = 0
        if count_tokens:
            for message in llm_message:
                input_tokens += len(encoding.encode(message["content"]))

            output_tokens = len(encoding.encode(response_content))

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

        return response_content, input_tokens, output_tokens
