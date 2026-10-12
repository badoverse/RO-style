from cache import TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_values_expire_after_ttl():
    clock = FakeClock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    cache.set("key", "value")

    clock.now = 9.9
    assert cache.get("key") == "value"
    clock.now = 10
    assert cache.get("key") is None
    assert len(cache) == 0


def test_oldest_entries_are_evicted_past_max_entries():
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (None, 2, 3)


def test_empty_values_are_cached_too():
    cache = TTLCache(ttl_seconds=60)
    cache.set("no results", [])

    assert cache.get("no results", "miss") == []
