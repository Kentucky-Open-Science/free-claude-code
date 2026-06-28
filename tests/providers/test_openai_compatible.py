"""Tests for the custom OpenAI-compatible provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.openai_compatible import (
    OPENAI_COMPATIBLE_DEFAULT_BASE,
    OpenAICompatibleProvider,
)

CUSTOM_BASE_URL = "https://custom.example.com/v1"


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "gpt-4o-mini"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def openai_compatible_config():
    return ProviderConfig(
        api_key="test_openai_compatible_key",
        base_url=CUSTOM_BASE_URL,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def openai_compatible_provider(openai_compatible_config):
    return OpenAICompatibleProvider(openai_compatible_config)


def test_init_uses_configured_base_url(openai_compatible_config):
    """The provider uses the user-supplied base URL, not a fixed default."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = OpenAICompatibleProvider(openai_compatible_config)
        assert provider._api_key == "test_openai_compatible_key"
        assert provider._base_url == CUSTOM_BASE_URL
        mock_openai.assert_called_once()


def test_default_base_url_constant_is_empty():
    assert OPENAI_COMPATIBLE_DEFAULT_BASE == ""


def test_build_request_body_basic(openai_compatible_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = MockRequest()
    body = openai_compatible_provider._build_request_body(req)

    assert body["model"] == "gpt-4o-mini"
    assert body["messages"][0]["role"] == "system"
    assert "max_completion_tokens" in body


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            api_key="test_openai_compatible_key",
            base_url=CUSTOM_BASE_URL,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_prefers_existing_max_completion_tokens(
    openai_compatible_provider,
):
    with patch(
        "providers.transports.openai_chat.request_policy.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 77,
            "max_tokens": 999,
        }
        body = openai_compatible_provider._build_request_body(MockRequest())

    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body


def test_build_request_body_preserves_caller_extra_body(openai_compatible_provider):
    req = MockRequest(extra_body={"clear_thinking": False})

    body = openai_compatible_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("clear_thinking") is False


@pytest.mark.asyncio
async def test_stream_response_text(openai_compatible_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in openai_compatible_provider.stream_response(req)
        ]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(openai_compatible_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        openai_compatible_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [
            event async for event in openai_compatible_provider.stream_response(req)
        ]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(openai_compatible_provider):
    openai_compatible_provider._client = AsyncMock()

    await openai_compatible_provider.cleanup()

    openai_compatible_provider._client.close.assert_called_once()
