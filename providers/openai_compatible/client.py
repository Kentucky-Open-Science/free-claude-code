"""Custom OpenAI-compatible provider (user-supplied base URL and API key)."""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import OPENAI_COMPATIBLE_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENAI_COMPATIBLE",
    include_extra_body=True,
    max_tokens_field="max_completion_tokens",
)


class OpenAICompatibleProvider(OpenAIChatTransport):
    """User-configured OpenAI-compatible Chat Completions deployment.

    The base URL and API key are both supplied via environment settings
    (``OPENAI_COMPATIBLE_BASE_URL`` / ``OPENAI_COMPATIBLE_API_KEY``); there is
    no fixed upstream default.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENAI_COMPATIBLE",
            base_url=config.base_url or OPENAI_COMPATIBLE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
