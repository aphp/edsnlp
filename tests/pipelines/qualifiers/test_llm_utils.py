from typing import Optional

import httpx
import respx
from openai.types.chat.chat_completion import ChatCompletion
from pytest import mark

from edsnlp.pipes.qualifiers.llm.llm_utils import (
    AsyncLLM,
    create_prompt_messages,
    parse_json_response,
)
from edsnlp.utils.asynchronous import run_async


@mark.parametrize("n_concurrent_tasks", [1, 2])
def test_async_llm(n_concurrent_tasks):
    api_url = "http://localhost:8000/v1/"
    suffix_url = "chat/completions"
    llm_api = AsyncLLM(n_concurrent_tasks=n_concurrent_tasks, api_url=api_url)

    with respx.mock:
        respx.post(api_url + suffix_url).mock(
            side_effect=[
                httpx.Response(
                    200, json={"choices": [{"message": {"content": "positive"}}]}
                ),
                httpx.Response(
                    200, json={"choices": [{"message": {"content": "negative"}}]}
                ),
            ]
        )

        response = run_async(
            llm_api(
                batch_messages=[
                    [{"role": "user", "content": "your prompt here"}],
                    [{"role": "user", "content": "your second prompt here"}],
                ]
            )
        )
    assert response == ["positive", "negative"]


def test_create_prompt_messages():
    messages = create_prompt_messages(
        system_prompt="Hello",
        user_prompt="Hi",
        examples=[("One", "1")],
        final_user_prompt="What is your name?",
    )
    messages_expected = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "What is your name?"},
    ]
    assert messages == messages_expected
    messages2 = create_prompt_messages(
        system_prompt="Hello",
        user_prompt=None,
        examples=[("One", "1")],
        final_user_prompt="What is your name?",
    )
    messages_expected2 = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "What is your name?"},
    ]
    assert messages2 == messages_expected2


def create_fake_chat_completion(
    choices: int = 1, content: Optional[str] = '{"biopsy":false}'
):
    fake_response_data = {
        "id": "chatcmpl-fake123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "toto",
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for i in range(choices)
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    # Create the ChatCompletion object
    fake_completion = ChatCompletion.model_validate(fake_response_data)
    return fake_completion


def test_parse_json_response():
    response = create_fake_chat_completion()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "DateModel",
            "schema": {
                "properties": {"biopsy": {"title": "Biopsy", "type": "boolean"}},
                "required": ["biopsy"],
                "title": "DateModel",
                "type": "object",
            },
        },
    }

    llm = AsyncLLM(n_concurrent_tasks=1)
    parsed_response = llm.parse_messages(response, response_format)
    assert parsed_response == {"biopsy": False}

    response = create_fake_chat_completion(content=None)
    parsed_response = llm.parse_messages(response, response_format=response_format)
    assert parsed_response == {}

    parsed_response = llm.parse_messages(response, response_format=None)
    assert parsed_response == response

    parsed_response = llm.parse_messages(None, response_format=None)
    assert parsed_response is None


@mark.parametrize("n_completions", [1, 2])
def test_exception_handling(n_completions):
    api_url = "http://localhost:8000/v1/"
    suffix_url = "chat/completions"
    llm_api = AsyncLLM(
        n_concurrent_tasks=1, api_url=api_url, n_completions=n_completions
    )

    with respx.mock:
        respx.post(api_url + suffix_url).mock(
            side_effect=[
                httpx.Response(404, json={"choices": [{}]}),
            ]
        )

        response = run_async(
            llm_api(
                batch_messages=[
                    [{"role": "user", "content": "your prompt here"}],
                ]
            )
        )
    if n_completions == 1:
        assert response == [""]
    else:
        assert response == [[""] * n_completions]


@mark.parametrize("errors", ["ignore", "raw"])
def test_json_decode_error(errors):
    raw_response = '{"biopsy";false}'
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "DateModel",
            "schema": {
                "properties": {"biopsy": {"title": "Biopsy", "type": "boolean"}},
                "required": ["biopsy"],
                "title": "DateModel",
                "type": "object",
            },
        },
    }

    response = parse_json_response(raw_response, response_format, errors=errors)
    if errors == "ignore":
        assert response == {}
    else:
        assert response == raw_response


def test_decode_no_format():
    raw_response = '{"biopsy":false}'

    response = parse_json_response(raw_response, response_format=None)

    assert response == raw_response


def test_multiple_completions(n_completions=2):
    api_url = "http://localhost:8000/v1/"
    suffix_url = "chat/completions"
    llm_api = AsyncLLM(
        n_concurrent_tasks=1, api_url=api_url, n_completions=n_completions
    )
    completion = create_fake_chat_completion(n_completions, content="false")
    with respx.mock:
        respx.post(api_url + suffix_url).mock(
            side_effect=[
                httpx.Response(200, json=completion.model_dump()),
            ]
        )

        response = run_async(
            llm_api(
                batch_messages=[
                    [{"role": "user", "content": "your prompt here"}],
                ]
            )
        )
    assert response == [["false", "false"]]
