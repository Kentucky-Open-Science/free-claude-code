"""Tests for the MLX-VLM (mlx_vlm.server) OpenAI-compatible provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

MLXVLM_MODEL = "Qwen3.8-27B-Uncensored-MLX-8bit"


@pytest.fixture
def provider() -> OpenAIChatProvider:
    return profiled_provider(
        "mlxvlm",
        make_provider_config(api_key="mlx-vlm", base_url="http://localhost:8080/v1"),
        admission=immediate_admission(),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:8080", "http://localhost:8080/v1"),
        ("http://localhost:8080/", "http://localhost:8080/v1"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "mlxvlm",
            make_provider_config(api_key="mlx-vlm", base_url=configured),
            admission=immediate_admission(),
        )

    assert provider._base_url == expected
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_build_request_body_uses_openai_chat_shape(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(MLXVLM_MODEL, max_tokens=None)

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == MLXVLM_MODEL
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"][0]["role"] == "system"
    assert "thinking" not in body
    # NO_REASONING: never send effort fields an arbitrary local server may reject.
    assert "reasoning_effort" not in body


def test_replay_reinjects_think_tags(provider: OpenAIChatProvider) -> None:
    request = make_messages_request(
        MLXVLM_MODEL,
        system=None,
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "s"},
                    {"type": "text", "text": "visible"},
                ],
            },
        ],
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["messages"][1]["content"] == ("<think>\nprivate\n</think>\n\nvisible")


@pytest.mark.asyncio
async def test_stream_response_uses_shared_openai_chat_provider(
    provider: OpenAIChatProvider,
) -> None:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from mlx_vlm.server",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(prompt_tokens=8, completion_tokens=4)

    async def stream():
        yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream(),
    ) as create:
        output = "".join(
            [
                event
                async for event in provider.stream_response(
                    make_messages_request(MLXVLM_MODEL)
                )
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == MLXVLM_MODEL
    assert "Hello from mlx_vlm.server" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
