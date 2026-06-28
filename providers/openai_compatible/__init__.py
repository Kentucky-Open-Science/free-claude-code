"""Custom OpenAI-compatible deployment adapter."""

from providers.defaults import OPENAI_COMPATIBLE_DEFAULT_BASE

from .client import OpenAICompatibleProvider

__all__ = ["OPENAI_COMPATIBLE_DEFAULT_BASE", "OpenAICompatibleProvider"]
