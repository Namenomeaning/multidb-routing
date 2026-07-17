"""OpenRouter client for routing experiments.

Project policy (CLAUDE.md §11): ALL LLM and embedding calls go through OpenRouter only.
This module is the single embedding boundary (`embed`) plus a plain-text chat helper
(`openrouter_chat`, used by the from-scratch build scripts). The main pipeline's structured
chat lives in `src.core.semantic_card` (`client`/`call_json`) and `src.agent.router`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from math import sqrt
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

T = TypeVar("T")
_openai_clients: dict[tuple[str, str], OpenAI] = {}
DEFAULT_OPENROUTER_CHAT_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONALITY = 1536
DEFAULT_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-large"


class EmptyResponseError(RuntimeError):
    """Provider returned an empty response."""


def _env(*names: str) -> str | None:
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def openrouter_config() -> tuple[str, str]:
    api_key = _env("OPENROUTER_API_KEY", "openrouter_api_key")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return api_key, OPENROUTER_BASE_URL


def embedding_model() -> str:
    return _env("EMBEDDING_MODEL", "embedding_model") or DEFAULT_EMBEDDING_MODEL


def embedding_dimensionality() -> int:
    raw = _env("EMBEDDING_DIMENSIONALITY", "embedding_dimensionality")
    if not raw:
        return DEFAULT_EMBEDDING_DIMENSIONALITY
    return int(raw)


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def _openrouter_client() -> OpenAI:
    api_key, base_url = openrouter_config()
    cache_key = (base_url, api_key[-8:])
    if cache_key not in _openai_clients:
        _openai_clients[cache_key] = OpenAI(base_url=base_url, api_key=api_key)
    return _openai_clients[cache_key]


def _retry_openrouter(fn: Callable[[], T], *, label: str, attempts: int = 10) -> T:
    """Retry with longer backoff for OpenRouter upstream 429s."""
    transient = (RateLimitError, APIConnectionError, InternalServerError, EmptyResponseError)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            body = getattr(exc, "body", None) or {}
            meta = body.get("error", {}).get("metadata", {}) if isinstance(body, dict) else {}
            retry_after = meta.get("retry_after_seconds", 10 * (attempt + 1))
            wait = min(float(retry_after) + 1, 60)
            print(f"[{label}] upstream 429, retry {attempt + 1}/{attempts} after {wait:.0f}s")
            time.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code < 500:
                raise
            last_exc = exc
            time.sleep(min(2 ** attempt, 30))
        except transient as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 30))
    raise last_exc or RuntimeError(f"{label} failed")


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed texts via OpenRouter (project policy: all LLM + embedding go through OpenRouter)."""
    selected_model = model or _env("EMBEDDING_MODEL", "embedding_model") or DEFAULT_OPENROUTER_EMBED_MODEL

    def _call():
        try:
            resp = _openrouter_client().embeddings.create(
                input=texts, model=selected_model, extra_body={"service_tier": "flex"}
            )
        except APIStatusError as exc:
            if exc.status_code != 400:
                raise
            # embeddings endpoint may not accept service_tier -- retry without it
            resp = _openrouter_client().embeddings.create(input=texts, model=selected_model)
        if not getattr(resp, "data", None):
            raise EmptyResponseError("empty embedding data")
        return resp

    resp = _retry_openrouter(_call, label=f"or-embed-{selected_model}")
    return [_normalize(list(data.embedding)) for data in resp.data]


# Project policy: prefer the low-cost "flex" service tier on OpenRouter.
# https://openrouter.ai/docs/guides/features/service-tiers
OPENROUTER_SERVICE_TIER = "flex"


def openrouter_chat(
    messages: list[dict],
    model: str = DEFAULT_OPENROUTER_CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    service_tier: str = OPENROUTER_SERVICE_TIER,
) -> str:
    """Plain-text chat completion via OpenRouter (used by the from-scratch build scripts)."""
    def _call():
        resp = _openrouter_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # flex = low cost; reasoning disabled because callers want direct
            # structured output (reasoning models otherwise spend the token
            # budget on hidden reasoning and return empty content).
            extra_body={"service_tier": service_tier, "reasoning": {"enabled": False}},
        )
        if not getattr(resp, "choices", None):
            raise EmptyResponseError("empty choices in response")
        content = resp.choices[0].message.content
        # Empty/None content (model hiccup under flex tier) must retry, not slip
        # through as a silent failure that corrupts downstream JSON parsing.
        if not content or not content.strip():
            raise EmptyResponseError("empty message content")
        return content

    return _retry_openrouter(_call, label=f"or-chat-{model}")
