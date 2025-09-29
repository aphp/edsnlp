import asyncio
import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
)

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

logger = logging.getLogger(__name__)


class AsyncLLM:
    """
    AsyncLLM is an asynchronous interface for interacting with Large Language Models
    (LLMs) via an API, supporting concurrent requests and batch processing.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-8B",
        api_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 50,
        extra_body: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        api_key: str = "EMPTY_API_KEY",
        n_completions: int = 1,
        timeout: float = 15.0,
        n_concurrent_tasks: int = 4,
        **kwargs,
    ):
        """
        Initializes the AsyncLLM class with configuration parameters for interacting
        with an OpenAI-compatible API server.

        Parameters
        ----------
        model_name : str, optional
            Name of the model to use (default: "Qwen/Qwen3-8B").
        api_url : str, optional
            Base URL of the API server (default: "http://localhost:8000/v1").
        temperature : float, optional
            Sampling temperature for generation (default: 0.0).
        max_tokens : int, optional
            Maximum number of tokens to generate per completion (default: 50).
        extra_body : Optional[Dict[str, Any]], optional
            Additional parameters to include in the API request body (default: None).
        response_format : Optional[Dict[str, Any]], optional
            Format specification for the API response (default: None).
        api_key : str, optional
            API key for authentication (default: "EMPTY_API_KEY").
        n_completions : int, optional
            Number of completions to request per prompt (default: 1).
        timeout : float, optional
            Timeout for API requests in seconds (default: 15.0).
        n_concurrent_tasks : int, optional
            Maximum number of concurrent tasks for API requests (default: 4).
        **kwargs
            Additional keyword arguments for further customization.
        """

        # Set OpenAI's API key and API base to use vLLM's API server.
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.response_format = response_format
        self.n_completions = n_completions
        self.timeout = timeout
        self.kwargs = kwargs
        self.n_concurrent_tasks = n_concurrent_tasks
        self.responses = []
        self._lock = None

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_url,
            default_headers={"Connection": "close"},
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - properly close the client."""
        await self.aclose()

    async def aclose(self):
        """Properly close the AsyncOpenAI client to prevent resource leaks."""
        if hasattr(self, "client") and self.client is not None:
            await self.client.close()

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def async_id_message_generator(
        self, batch_messages: List[List[Dict[str, str]]]
    ) -> AsyncIterator[Tuple[int, List[Dict[str, str]]]]:
        """
        Generator
        """
        for i, messages in enumerate(batch_messages):
            yield (i, messages)

    def parse_messages(
        self, response: ChatCompletion, response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse the response from the LLM and return the content.
        """
        if (response_format is not None) and (isinstance(response, ChatCompletion)):
            prediction = [
                parse_json_response(
                    choice.message.content, response_format=response_format
                )
                for choice in response.choices
            ]
            if self.n_completions == 1:
                prediction = prediction[0]

            return prediction

        else:
            return response

    async def call_llm(
        self, id: int, messages: List[Dict[str, str]]
    ) -> Tuple[int, ChatCompletion]:
        """
        Call the LLM with the given messages and return the response.

        Parameters
        ----------
        id : int
            Unique identifier for the call
        messages : List[Dict[str, str]]
            List of messages to send to the LLM, where each message is a dictionary
            with keys 'role' and 'content'.

        Returns
        -------
        Tuple[int, ChatCompletion]
            The id of the call and the ChatCompletion object corresponding to the
            LLM response
        """

        raw_response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                n=self.n_completions,
                temperature=self.temperature,
                stream=False,
                response_format=self.response_format,
                extra_body=self.extra_body,
                **self.kwargs,
            ),
            timeout=self.timeout,
        )

        # Parse the response
        parsed_response = self.parse_messages(raw_response, self.response_format)

        return id, parsed_response

    def store_responses(self, p_id, abbreviation_list):
        """ """
        self.responses.append((p_id, abbreviation_list))

    # async def async_worker(
    #     self,
    #     name: str,
    #     id_messages_tuples: AsyncIterator[Tuple[int, List[List[Dict[str, str]]]]],
    # ):
    #     """ """

    #     async for (
    #         idx,
    #         message,
    #     ) in id_messages_tuples:
    #         logger.info(idx)

    #         try:
    #             idx, response = await self.call_llm(idx, message)

    #             logger.info(f"Worker {name} has finished process {idx}")
    #         except Exception as e:  # BaseException
    #             logger.error(
    #                 f"[{name}] Exception raised on chunk {idx}\n{e}"
    #             )  # type(e)
    #             if self.n_completions == 1:
    #                 response = ""
    #             else:
    #                 response = [""] * self.n_completions

    #         async with self.lock:
    #             self.store_responses(
    #                 idx,
    #                 response,
    #             )

    async def async_worker(
        self,
        name: str,
        id_messages_tuples: AsyncIterator[Tuple[int, List[List[Dict[str, str]]]]],
    ):
        while True:
            try:
                (
                    idx,
                    message,
                ) = await anext(id_messages_tuples)  # noqa: F821
                idx, response = await self.call_llm(idx, message)

                logger.info(f"Worker {name} has finished process {idx}")
            except StopAsyncIteration:
                # Everything has been parsed!
                logger.info(
                    f"[{name}] Received StopAsyncIteration, worker will shutdown"
                )
                break
            except TimeoutError as e:
                logger.error(f"[{name}] TimeoutError on chunk {idx}\n{e}")
                logger.error(f"Timeout was set to {self.timeout} seconds")
                if self.n_completions == 1:
                    response = ""
                else:
                    response = [""] * self.n_completions
            except BaseException as e:
                logger.error(
                    f"[{name}] Exception raised on chunk {idx}\n{e}"
                )  # type(e)
                if self.n_completions == 1:
                    response = ""
                else:
                    response = [""] * self.n_completions
            async with self.lock:
                self.store_responses(
                    idx,
                    response,
                )

    def sort_responses(self):
        sorted_responses = []
        for i, output in sorted(self.responses, key=lambda x: x[0]):
            if isinstance(output, ChatCompletion):
                if self.n_completions == 1:
                    sorted_responses.append(output.choices[0].message.content)
                else:
                    sorted_responses.append(
                        [choice.message.content for choice in output.choices]
                    )
            else:
                sorted_responses.append(output)

        return sorted_responses

    def clean_storage(self):
        del self.responses
        self.responses = []

    async def __call__(self, batch_messages: List[List[Dict[str, str]]]):
        """
        Asynchronous coroutine, it should be called using the
        `edsnlp.utils.asynchronous.run_async` function.

        Parameters
        ----------
        batch_messages : List[List[Dict[str, str]]]
            List of message batches to send to the LLM, where each batch is a list
            of dictionaries with keys 'role' and 'content'.
        """
        try:
            # Shared prompt generator
            id_messages_tuples = self.async_id_message_generator(batch_messages)

            # n concurrent tasks
            tasks = {
                asyncio.create_task(
                    self.async_worker(f"Worker-{i}", id_messages_tuples)
                )
                for i in range(self.n_concurrent_tasks)
            }

            await asyncio.gather(*tasks)
            tasks.clear()
            predictions = self.sort_responses()
            self.clean_storage()

            return predictions
        except Exception:
            # Ensure cleanup even if an exception occurs
            await self.aclose()
            raise


def create_prompt_messages(
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    examples: Optional[List[Tuple[str, str]]] = None,
    final_user_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Create a list of prompt messages formatted for use with a language model (LLM) API.

    system_prompt : Optional[str], default=None
        The initial system prompt to set the behavior or context for the LLM.
    user_prompt : Optional[str], default=None
        An initial user prompt to provide context or instructions to the LLM.
    examples : Optional[List[Tuple[str, str]]], default=None
        A list of example (prompt, response) pairs to guide the LLM's behavior.
    final_user_prompt : Optional[str], default=None
        The final user prompt to be appended at the end of the message sequence.

    Returns
    -------
    List[Dict[str, str]]
        A list of message dictionaries, each containing a 'role'
        (e.g., 'system', 'user', 'assistant')
        and corresponding 'content', formatted for LLM input.

    """

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    if user_prompt:
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
    if examples:
        for prompt, response in examples:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
    if final_user_prompt:
        messages.append(
            {
                "role": "user",
                "content": final_user_prompt,
            }
        )

    return messages


def parse_json_response(
    response: str,
    response_format: Optional[Dict[str, Any]] = None,
    errors: str = "ignore",
) -> Dict[str, Any]:
    """
    Parses a response string as JSON if a JSON schema format is specified,
    otherwise returns the raw response.

    Parameters
    ----------
    response : str
        The response string to parse.
    response_format : Optional[Dict[str, Any]], optional
        A dictionary specifying the expected response format.
        If it contains {"type": "json_schema"}, the response will be parsed as JSON.
        Defaults to None.
    errors : str, optional
        Determines error handling behavior when JSON decoding fails.
        If set to "ignore", returns an empty dictionary on failure.
        Otherwise, returns the raw response. Defaults to "ignore".

    Returns
    -------
    Dict[str, Any]
        The parsed JSON object if parsing is successful and a JSON schema is specified.
        If parsing fails and errors is "ignore", returns an empty dictionary.
        If parsing fails and errors is not "ignore", returns the raw response string.
        If no response format is specified, returns the raw response string.
    """
    if response is None:
        return {}

    if (response_format is not None) and (response_format.get("type") == "json_schema"):
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            if errors == "ignore":
                return {}
            else:
                return response
    else:
        # If no response format is specified, return the raw response
        return response
