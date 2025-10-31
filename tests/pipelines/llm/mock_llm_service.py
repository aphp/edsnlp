import re
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def mock_llm_service(responder, *, base_url="http://localhost:8080/v1"):
    """
    Patch the OpenAI client so calls to chat.completions.create go through
    responder instead.
    """

    import inspect
    from unittest.mock import patch

    from openai.resources.chat.completions.completions import (
        AsyncCompletions,
        Completions,
    )

    tag_pattern = re.compile(r"</?[^>]+?>")

    def extract_user_text(messages):
        if not messages:
            return None
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
        return None

    def to_openai_response(result, *, messages):
        if inspect.isawaitable(result):  # pragma: no cover
            raise TypeError("responder must return a string or response, not awaitable")

        if isinstance(result, str):
            content = result
            user_text = extract_user_text(messages)
            if user_text:
                plain_fragment = tag_pattern.sub("", content)
                if plain_fragment and plain_fragment in user_text:
                    content = user_text.replace(plain_fragment, content, 1)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )
                ]
            )

        if isinstance(result, dict) and "choices" in result:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=choice["message"]["content"])
                    )
                    for choice in result["choices"]
                ]
            )

        return result

    def check_base_url(client) -> None:
        if base_url is None:
            return
        try:
            client_url = str(client.base_url)
        except AttributeError:
            try:
                client_url = str(client._client.base_url)
            except AttributeError:  # pragma: no cover - very defensive
                return
        assert client_url.rstrip("/") == base_url.rstrip("/"), (
            f"Unexpected base_url {client_url}, expected {base_url}"
        )

    def sync_create(self, *args, **kwargs):
        if args:
            raise TypeError("mocked completions.create expects keyword arguments only")
        check_base_url(self._client)
        return to_openai_response(responder(**kwargs), messages=kwargs.get("messages"))

    async def async_create(self, *args, **kwargs):
        if args:
            raise TypeError("mocked completions.create expects keyword arguments only")
        check_base_url(self._client)
        result = responder(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return to_openai_response(result, messages=kwargs.get("messages"))

    with patch.object(Completions, "create", sync_create), patch.object(
        AsyncCompletions, "create", async_create
    ):
        yield
