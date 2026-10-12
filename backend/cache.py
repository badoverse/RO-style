"""Small in-memory TTL cache. Per-process: contents vanish on restart.

Not thread-safe; fine for use from a single asyncio event loop (FastAPI async routes).
"""

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class TTLCache:
    """Maps keys to values that expire ttl_seconds after they were set.

    Holds at most max_entries; past that the oldest entries are dropped first
    (with one TTL per cache, those are also the ones closest to expiring).
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    def get(self, key: Hashable, default: Any = None) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return default
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._entries[key]
            return default
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._entries[key] = (self._clock() + self.ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
