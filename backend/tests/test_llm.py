import httpx2
import openai
import pytest

import llm
from cache import TTLCache

pytestmark = pytest.mark.anyio

URL = "https://openrouter.ai/api/v1/chat/completions"


def rate_limited(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx2.Response(429, headers=headers, request=httpx2.Request("POST", URL))
    return openai.RateLimitError("rate limited", response=response, body=None)


def connection_error():
    return openai.APIConnectionError(request=httpx2.Request("POST", URL))


async def ask(**kwargs):
    return await llm.complete_json("system prompt", "user prompt", label="test", cache_key="key", **kwargs)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("[1, 2]", [1, 2]),
        ("```json\n[1, 2]\n```", [1, 2]),
        ('```\n{"ids": [3]}\n```', {"ids": [3]}),
        ("<think>they want all black</think>\n[2]", [2]),
        ("Sure! Here are my picks: [3, 1] Hope that helps.", [3, 1]),
    ],
)
def test_parse_json_tolerates_fences_and_chatter(reply, expected):
    assert llm.parse_json(reply) == expected


def test_parse_json_rejects_prose():
    with pytest.raises(ValueError):
        llm.parse_json("I would pick the first one.")


async def test_sends_one_request_and_returns_the_parsed_reply(fake_llm):
    fake_llm.replies = ['```json\n[{"slot": "Hat"}]\n```']

    assert await ask() == [{"slot": "Hat"}]

    [call] = fake_llm.calls
    assert call["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert call["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert call["extra_body"] == {"reasoning": {"enabled": False}}


async def test_a_slow_model_runs_into_the_deadline(fake_llm, monkeypatch):
    monkeypatch.setattr(llm, "DEADLINE_SECONDS", 0.05)
    fake_llm.latency = 5
    fake_llm.replies = ["[1]"]

    with pytest.raises(llm.LLMError, match="no usable reply within"):
        await ask()


async def test_reasks_once_with_a_json_only_system_prompt(fake_llm):
    fake_llm.replies = ["I'd go with the first two.", "[1, 2]"]

    assert await ask() == [1, 2]
    assert fake_llm.calls[1]["messages"][0]["content"] == "system prompt\n\nReturn JSON only, no markdown"


async def test_a_reply_the_caller_cannot_use_is_reasked_too(fake_llm):
    def parse(reply):
        if not isinstance(reply, list):
            raise ValueError("not a list")
        return reply

    fake_llm.replies = ['{"oops": true}', "[1]"]

    assert await ask(parse=parse) == [1]
    assert len(fake_llm.calls) == 2


async def test_fails_when_the_reask_is_unusable_too(fake_llm):
    fake_llm.replies = ["no idea", "still prose"]

    with pytest.raises(llm.LLMError, match="JSON-only retry"):
        await ask()


async def test_transient_errors_are_retried_with_exponential_backoff(fake_llm):
    fake_llm.replies = [rate_limited(), connection_error(), "[1]"]

    assert await ask() == [1]
    assert len(fake_llm.calls) == 3
    assert 1 <= fake_llm.sleeps[0] <= 1.5
    assert 2 <= fake_llm.sleeps[1] <= 2.5


async def test_retry_after_stretches_the_backoff(fake_llm):
    fake_llm.replies = [rate_limited(retry_after="7"), "[1]"]

    await ask()

    assert 7 <= fake_llm.sleeps[0] <= 7.5


async def test_empty_replies_are_retried(fake_llm):
    fake_llm.replies = [None, "", "[1]"]

    assert await ask() == [1]


async def test_gives_up_after_three_attempts(fake_llm):
    fake_llm.replies = [connection_error() for _ in range(3)]

    with pytest.raises(llm.LLMError, match="3 attempts"):
        await ask()
    assert len(fake_llm.calls) == 3
    assert len(fake_llm.sleeps) == 2


async def test_errors_retrying_cannot_fix_fail_fast(fake_llm):
    response = httpx2.Response(401, request=httpx2.Request("POST", URL))
    fake_llm.replies = [openai.AuthenticationError("invalid key", response=response, body=None)]

    with pytest.raises(llm.LLMError, match="invalid key"):
        await ask()
    assert len(fake_llm.calls) == 1
    assert fake_llm.sleeps == []


async def test_replies_are_cached(fake_llm):
    fake_llm.replies = ["[1]"]

    assert await ask() == [1]
    assert await ask() == [1]
    assert len(fake_llm.calls) == 1


async def test_missing_api_key_is_an_llm_error(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_cache", TTLCache(ttl_seconds=60))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(llm.LLMError, match="OPENROUTER_API_KEY"):
        await ask()


async def test_vision_call_uses_the_vision_model_and_sends_image_parts(fake_llm):
    fake_llm.replies = ['{"1": 2}']
    content = [
        {"type": "text", "text": "pick one per slot"},
        {"type": "image_url", "image_url": {"url": "https://tr.fake/12.png"}},
    ]

    result = await llm.complete_json(
        "system prompt", content, label="vision", cache_key="vkey", model=llm.VISION_MODEL
    )

    assert result == {"1": 2}
    [call] = fake_llm.calls
    assert call["model"] == llm.VISION_MODEL
    assert call["messages"][1]["content"] == content  # multimodal content passed through unchanged


async def test_client_targets_openrouter_with_the_app_headers(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    client = llm._get_client()
    try:
        assert str(client.base_url) == "https://openrouter.ai/api/v1/"
        assert client.default_headers["HTTP-Referer"] == "http://localhost:8000"
        assert client.default_headers["X-Title"] == "Roblox Avatar Stylist"
    finally:
        await llm.aclose()
