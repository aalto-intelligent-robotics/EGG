import os
from typing import Iterable
import httpx
from openai import OpenAI
import tiktoken
import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/llm.log",
)


class LLMAgent:
    def __init__(
        self,
        use_gpt4o: bool = True,
        use_mini: bool = True,
        temperature: float = 0,
    ):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.temperature = temperature
        # Change the `path` variable to the endpoint listed at https://www.aalto.fi/en/services/aalto-ai-apis
        if use_gpt4o:
            if use_mini:
                base_url = "https://aalto-openai-apigw.azure-api.net/v1/openai/deployments/gpt-4o-mini-2024-07-18"
                logger.info(f"Using GPT4o-mini from {base_url}")
            else:
                # base_url = "https://aalto-openai-apigw.azure-api.net/v1/openai/deployments/gpt-4o-2024-11-20"
                base_url = "https://aalto-openai-apigw.azure-api.net/v1/openai/gpt4o"
                logger.info(f"Using GPT4o from {base_url}")
            aalto_openai_endpoint_url = "/chat/completions"
        else:
            base_url = "https://aalto-openai-apigw.azure-api.net"
            aalto_openai_endpoint_url = "/v1/chat/gpt-35-turbo-1106"
            logger.info(f"Using GPT3.5")

        # Set API key in terminal: export AALTO_OPENAI_API_KEY=""
        aalto_api_key = os.environ.get("AALTO_OPENAI_API_KEY")
        assert (
            aalto_api_key is not None
        ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."

        """
        Rewrite the base path with Aalto mappings
        For all endpoints see https://www.aalto.fi/en/services/azure-openai#6-available-api-s
        """

        def update_base_url(request: httpx.Request) -> None:
            if request.url.path == "/chat/completions":
                request.url = request.url.copy_with(path=aalto_openai_endpoint_url)

        self.client = OpenAI(
            base_url=base_url,
            api_key="False",  # API key not used, and rather set below
            default_headers={
                "Ocp-Apim-Subscription-Key": aalto_api_key,
            },
            http_client=httpx.Client(event_hooks={"request": [update_base_url]}),
        )

    def send_query(self, llm_message: Iterable, count_tokens: bool = False):
        # Send query
        completion = self.client.chat.completions.create(
            model="no_effect",  # the model variable must be set, but has no effect, model selection done with URL
            messages=llm_message,
            temperature=self.temperature,
        )

        # Get Content of the response
        response_content = completion.choices[0].message.content

        if response_content is None:
            return response_content

        # Count tokens
        encoding_name = "cl100k_base"  # For GPT-3.5-turbo-1106 and GPT-4o
        encoding = tiktoken.get_encoding(encoding_name)

        if count_tokens:
            input_tokens = 0
            for message in llm_message:
                input_tokens += len(encoding.encode(message["content"]))

            output_tokens = len(encoding.encode(response_content))

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

        return response_content
