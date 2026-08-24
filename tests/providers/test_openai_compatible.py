"""Tests for the user-configured OpenAI-compatible provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.inference_support import collect_anthropic
from tests.providers.request_factory import canonical_request, make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

OPENAI_COMPATIBLE_MODEL = "zai-org/GLM-5.2-FP8"
BASE_URL = "https://api-llm-factory.ai.uky.edu/v1"


@pytest.fixture
def provider() -> OpenAIChatProvider:
    return profiled_provider(
        "openai_compatible",
        make_provider_config(api_key="test-key", base_url=BASE_URL),
        admission=immediate_admission(),
    )


def test_init_uses_configured_base_url_and_key() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider(
            "openai_compatible",
            make_provider_config(api_key="test-key", base_url=BASE_URL),
            admission=immediate_admission(),
        )

    assert provider._provider_name == "OPENAI_COMPATIBLE"
    assert provider._base_url == BASE_URL
    assert provider._api_key == "test-key"
    assert openai_client.call_args.kwargs["base_url"] == BASE_URL


def test_build_request_body_uses_max_completion_tokens(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(OPENAI_COMPATIBLE_MODEL, max_tokens=512)

    body = provider._build_request_body(
        canonical_request(request),
        reasoning=reasoning_for(request),
        provider_model=request.model,
    )

    assert body["model"] == OPENAI_COMPATIBLE_MODEL
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert body["messages"][0]["role"] == "system"
    # NO_REASONING: never send effort fields an arbitrary backend may reject.
    assert "reasoning_effort" not in body


def test_extra_body_passthrough(provider: OpenAIChatProvider) -> None:
    request = make_messages_request(
        OPENAI_COMPATIBLE_MODEL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    body = provider._build_request_body(
        canonical_request(request),
        reasoning=reasoning_for(request),
        provider_model=request.model,
    )

    assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_stream_response_uses_shared_openai_chat_provider(
    provider: OpenAIChatProvider,
) -> None:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from a custom deployment",
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
            await collect_anthropic(
                provider.stream_response(
                    canonical_request(make_messages_request(OPENAI_COMPATIBLE_MODEL)),
                    provider_model=make_messages_request(OPENAI_COMPATIBLE_MODEL).model,
                )
            )
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == OPENAI_COMPATIBLE_MODEL
    assert "Hello from a custom deployment" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
