import httpx
import respx
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


def test_parse_json_response():
    raw_response = '{"biopsy":false}'
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

    response = parse_json_response(raw_response, response_format)
    assert response == {"biopsy": False}
