"""OpenRouter chat calls that return JSON: retries, lenient parsing and a 24-hour cache.

OpenRouter speaks the OpenAI API, so this uses the openai SDK's async client. The
model has no JSON mode, so replies are parsed leniently and re-asked once with a
stricter system prompt when they aren't usable.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Hashable
from typing import Any

import openai
from openai import AsyncOpenAI

from cache import TTLCache

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Image-capable free model for the visual "Create outfit" pipeline. It must accept
# MULTIPLE images per request (several candidate thumbnails arrive in one call). Swap
# for another free vision id from https://openrouter.ai/api/v1/models whose
# architecture.input_modalities includes "image" (e.g. "google/gemma-4-31b-it:free")
# if this one is deprecated.
VISION_MODEL = "nvidia/nemotron-3.5-super-vl:free"
# Sent on every call so OpenRouter can attribute (and route) the app's free-tier traffic.
APP_HEADERS = {"HTTP-Referer": "http://localhost:8000", "X-Title": "Roblox Avatar Stylist"}

# Nemotron is a reasoning model. With reasoning on, one ranking call spent 3504 tokens
# and 163 s to answer "[3, 1, 2]"; with it off, the same call took 1.5 s and 13 tokens.
EXTRA_BODY = {"reasoning": {"enabled": False}}

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
# Wall-clock budget for one complete_json() call, retries and re-ask included. The
# SDK's own timeout only limits the wait between bytes (a 163 s call got through a
# 90 s timeout), so it can't bound how long a slow free model keeps us waiting.
DEADLINE_SECONDS = 45.0
JSON_ONLY_PROMPT = "Return JSON only, no markdown"

_RETRYABLE_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

_cache = TTLCache(ttl_seconds=24 * 60 * 60)
_client: AsyncOpenAI | None = None
_sleep = asyncio.sleep  # module-level so tests can skip the waits


class LLMError(Exception):
    """The model couldn't produce a usable reply."""


async def complete_json(
    system: str,
    user: str | list,
    *,
    label: str,
    cache_key: Hashable,
    parse: Callable[[Any], Any] = lambda value: value,
    model: str = MODEL,
    extra_body: dict | None = EXTRA_BODY,
) -> Any:
    """Ask the model for JSON and return parse(decoded reply), cached for 24 hours by cache_key.

    user may be a plain string or a list of OpenAI content parts (text + image_url
    blocks), which is how the vision model receives candidate thumbnails. model selects
    the model (MODEL for text, VISION_MODEL for images); extra_body is sent as-is.

    parse validates the decoded JSON and raises ValueError when it's unusable; that,
    or a reply that isn't JSON at all, gets one re-ask with a JSON-only system prompt.
    label names the call in logs. Raises LLMError when no usable reply comes back
    within DEADLINE_SECONDS.
    """
    cached = _cache.get(cache_key)
    if cached is not None:
        log.info("LLM %s: cache hit", label)
        return cached
    try:
        result = await asyncio.wait_for(
            _complete_json(system, user, label, parse, model, extra_body), DEADLINE_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise LLMError(f"{label}: no usable reply within {DEADLINE_SECONDS:.0f}s") from exc
    _cache.set(cache_key, result)
    return result


async def _complete_json(
    system: str,
    user: str | list,
    label: str,
    parse: Callable[[Any], Any],
    model: str,
    extra_body: dict | None,
) -> Any:
    reply = await _chat(system, user, label, model, extra_body)
    try:
        result = parse(parse_json(reply))
    except ValueError as exc:
        log.warning("LLM %s: unusable reply (%s); asking again for JSON only", label, exc)
        reply = await _chat(f"{system}\n\n{JSON_ONLY_PROMPT}", user, label, model, extra_body)
        try:
            result = parse(parse_json(reply))
        except ValueError as exc:
            raise LLMError(f"{label}: unusable reply after the JSON-only retry: {exc}") from exc
    return result


def parse_json(text: str) -> Any:
    """Decode a model reply as JSON, tolerating ```json fences, <think> blocks and chatter around it."""
    cleaned = _CODE_FENCE.sub("", _THINK_BLOCK.sub("", text).strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Models sometimes wrap the JSON in a sentence; try the outermost [...] or {...}.
    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opening), cleaned.rfind(closing)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"no JSON in reply {text[:120]!r}")


async def aclose() -> None:
    """Close the OpenRouter client; call on app shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def _chat(
    system: str, user: str | list, label: str, model: str = MODEL, extra_body: dict | None = EXTRA_BODY
) -> str:
    """One chat completion, tried up to MAX_ATTEMPTS times with exponential backoff.

    user may be a plain string or a list of OpenAI content parts (text + image_url),
    which is how the vision model is shown candidate thumbnails.
    """
    client = _get_client()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        retry_after = None
        try:
            response = await client.chat.completions.create(model=model, messages=messages, extra_body=extra_body)
        except _RETRYABLE_ERRORS as exc:
            error: Exception = exc
            if isinstance(exc, openai.APIStatusError):
                retry_after = exc.response.headers.get("retry-after")
        except openai.OpenAIError as exc:  # bad key, unknown model, bad request: retrying won't help
            raise LLMError(f"{label}: {exc}") from exc
        else:
            text = _reply_text(response)
            usage = getattr(response, "usage", None)
            log.info(
                "LLM %s -> %s in %.1fs (attempt %d/%d, tokens in/out: %s/%s)",
                label, "reply" if text else "empty reply", time.monotonic() - started, attempt, MAX_ATTEMPTS,
                getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"),
            )
            if text:
                return text
            error = LLMError("empty reply")

        if attempt == MAX_ATTEMPTS:
            raise LLMError(f"{label}: failed after {MAX_ATTEMPTS} attempts: {error}") from error
        delay = _backoff_delay(attempt, retry_after)
        log.warning("LLM %s attempt %d/%d failed (%s); retrying in %.1fs", label, attempt, MAX_ATTEMPTS, error, delay)
        await _sleep(delay)
    raise AssertionError("unreachable")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        _client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=DEADLINE_SECONDS,
            max_retries=0,  # _chat retries itself so that every attempt is logged
            default_headers=APP_HEADERS,
        )
    return _client


def _reply_text(response: Any) -> str:
    """The reply's text, or "" when the provider sent back no usable choice."""
    choices = getattr(response, "choices", None) or []
    message = choices[0].message if choices else None
    return ((message.content if message else None) or "").strip()


def _backoff_delay(attempt: int, retry_after: str | None) -> float:
    """1s, 2s, ... (+ jitter), or longer when the server says when to come back."""
    delay = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
    try:
        delay = max(delay, float(retry_after))
    except (TypeError, ValueError):
        pass
    return min(delay, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5)
