import asyncio
from types import SimpleNamespace

import pytest

import llm
from cache import TTLCache


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeLLM:
    """Stands in for the AsyncOpenAI client.

    Each call is answered by respond(messages) when set, else by the next entry in
    replies: the reply text, None for an empty reply, or an exception to raise.
    """

    def __init__(self):
        self.replies: list = []
        self.respond = None
        self.latency = 0.0  # real seconds each reply takes
        self.calls: list[dict] = []
        self.sleeps: list[float] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, *, model, messages, **options):
        self.calls.append({"model": model, "messages": messages, **options})
        await asyncio.sleep(self.latency)
        reply = self.respond(messages) if self.respond else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )

    async def sleep(self, seconds):
        self.sleeps.append(seconds)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "_sleep", fake.sleep)
    monkeypatch.setattr(llm, "_cache", TTLCache(ttl_seconds=60))
    return fake
