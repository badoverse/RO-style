"""Offline tests for roblox_client against an in-memory fake of the Roblox endpoints."""

import json

import httpx
import pytest

import roblox_client as rc
from cache import TTLCache

pytestmark = pytest.mark.anyio


def asset(item_id, price=10, **extra):
    """A catalog details record, as v1/catalog/items/details returns it."""
    return {"id": item_id, "itemType": "Asset", "name": f"Item {item_id}", "price": price, "lowestPrice": price, **extra}


def bundle(item_id, price=60):
    return {"id": item_id, "itemType": "Bundle", "name": f"Bundle {item_id}", "price": price, "lowestPrice": price}


def economy(price, for_sale=True, resale=None):
    """An economy v2 asset details record."""
    return {
        "PriceInRobux": price,
        "IsForSale": for_sale,
        "IsPublicDomain": False,
        "CollectiblesItemDetails": {"CollectibleLowestResalePrice": resale} if resale is not None else None,
    }


class FakeRoblox:
    """Stand-in for the catalog, economy and thumbnails endpoints."""

    csrf_token = "token-1"

    def __init__(self):
        self.catalog: list[dict] = []  # search results in relevance order, as catalog details records
        self.economy: dict[int, dict] = {}  # asset id -> economy details; other ids 404
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self._queued: dict[str, list[httpx.Response]] = {}

    def queue(self, path, *responses):
        """Answer the next requests to path with these responses, then behave normally."""
        self._queued.setdefault(path, []).extend(responses)

    def requests_to(self, path):
        return [request for request in self.requests if request.url.path == path]

    async def sleep(self, seconds):
        self.sleeps.append(seconds)

    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if self._queued.get(path):
            return self._queued[path].pop(0)
        if path == "/v1/search/items":
            return self._search(request.url.params)
        if path == "/v1/catalog/items/details":
            return self._details(request)
        if path in ("/v1/assets", "/v1/bundles/thumbnails"):
            return self._thumbnails(request.url.params)
        if path.startswith("/v2/assets/") and int(path.split("/")[3]) in self.economy:
            return httpx.Response(200, json=self.economy[int(path.split("/")[3])])
        return httpx.Response(404, json={"errors": [{"code": 0, "message": "NotFound"}]})

    def _search(self, params):
        start = int(params.get("cursor", 0))
        end = start + int(params["limit"])
        page = [{"id": item["id"], "itemType": item["itemType"]} for item in self.catalog[start:end]]
        return httpx.Response(200, json={"data": page, "nextPageCursor": str(end) if end < len(self.catalog) else None})

    def _details(self, request):
        if request.headers.get("x-csrf-token") != self.csrf_token:
            return httpx.Response(403, headers={"x-csrf-token": self.csrf_token})
        wanted = {(item["itemType"], item["id"]) for item in json.loads(request.content)["items"]}
        found = [record for record in self.catalog if (record["itemType"], record["id"]) in wanted]
        return httpx.Response(200, json={"data": found[::-1]})  # reversed: order must not matter

    def _thumbnails(self, params):
        item_type, ids = ("Asset", params["assetIds"]) if "assetIds" in params else ("Bundle", params["bundleIds"])
        data = [
            {"targetId": int(item_id), "state": "Completed", "imageUrl": f"https://tr.fake/{item_type}/{item_id}.png"}
            for item_id in ids.split(",")
        ]
        return httpx.Response(200, json={"data": data})


@pytest.fixture
async def roblox(monkeypatch):
    fake = FakeRoblox()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as client:
        monkeypatch.setattr(rc, "_client", client)
        monkeypatch.setattr(rc, "_csrf_token", None)
        monkeypatch.setattr(rc, "_sleep", fake.sleep)
        for cache in ("_search_cache", "_thumbnail_cache", "_price_cache"):
            monkeypatch.setattr(rc, cache, TTLCache(ttl_seconds=60))
        yield fake


async def test_search_returns_named_priced_items_in_relevance_order(roblox):
    roblox.catalog = [asset(3, 65), bundle(7, 60), asset(1, 80)]

    items = await rc.search_slot("black hoodie", "Clothing", "ShirtAccessories")

    assert items == [
        {"id": 3, "name": "Item 3", "price": 65, "item_type": "Asset"},
        {"id": 7, "name": "Bundle 7", "price": 60, "item_type": "Bundle"},
        {"id": 1, "name": "Item 1", "price": 80, "item_type": "Asset"},
    ]
    assert dict(roblox.requests_to("/v1/search/items")[0].url.params) == {
        "category": "Clothing",
        "subcategory": "ShirtAccessories",
        "keyword": "black hoodie",
        "limit": "120",
        "sortType": "Relevance",
    }


async def test_search_paginates_until_max_results(roblox, monkeypatch):
    monkeypatch.setattr(rc, "SEARCH_PAGE_SIZE", 30)
    roblox.catalog = [asset(item_id) for item_id in range(1, 200)]

    items = await rc.search_slot("hat", "Accessories", "HeadAccessories")

    assert [item["id"] for item in items] == list(range(1, 101))
    assert len(roblox.requests_to("/v1/search/items")) == 4
    assert roblox.sleeps == [0.5, 0.5, 0.5]


async def test_search_stops_when_cursor_runs_out(roblox):
    roblox.catalog = [asset(item_id) for item_id in range(1, 41)]

    items = await rc.search_slot("hat", "Accessories", "HeadAccessories")

    assert len(items) == 40
    assert len(roblox.requests_to("/v1/search/items")) == 1
    assert roblox.sleeps == []


@pytest.mark.parametrize(
    ("kwargs", "expected_max_price"),
    [
        ({}, None),
        ({"max_price": 200}, None),  # ignored unless price_filter is on
        ({"price_filter": True, "max_price": 200}, "200"),
        ({"free_only": True}, "0"),
        ({"free_only": True, "price_filter": True, "max_price": 200}, "0"),
    ],
)
async def test_price_params_are_only_sent_when_filtering(roblox, kwargs, expected_max_price):
    roblox.catalog = [asset(1, price=0)]

    await rc.search_slot("hat", "Accessories", **kwargs)

    params = roblox.requests_to("/v1/search/items")[0].url.params
    assert params.get("maxPrice") == expected_max_price
    assert "minPrice" not in params
    assert "salesTypeFilter" not in params


async def test_price_cap_is_enforced_on_fresh_prices(roblox):
    # As if the search index still had items 2 and 3 as free.
    roblox.catalog = [asset(1, price=0), asset(2, price=50), asset(3, priceStatus="Off Sale")]

    items = await rc.search_slot("hat", "Accessories", free_only=True)

    assert [item["id"] for item in items] == [1]


async def test_search_results_are_cached(roblox):
    roblox.catalog = [asset(1)]

    first = await rc.search_slot("Black Hoodie ", "Clothing", "ShirtAccessories")
    first[0]["name"] = "changed by the caller"
    second = await rc.search_slot("black hoodie", "Clothing", "ShirtAccessories")

    assert second == [{"id": 1, "name": "Item 1", "price": 10, "item_type": "Asset"}]
    assert len(roblox.requests_to("/v1/search/items")) == 1

    await rc.search_slot("black hoodie", "Clothing", "ShirtAccessories", free_only=True)
    assert len(roblox.requests_to("/v1/search/items")) == 2  # other price range, other key


async def test_rate_limited_requests_back_off_exponentially(roblox):
    roblox.catalog = [asset(1)]
    roblox.queue("/v1/search/items", httpx.Response(429), httpx.Response(429))

    items = await rc.search_slot("hat", "Accessories")

    assert len(items) == 1
    assert len(roblox.sleeps) == 2
    assert 1 <= roblox.sleeps[0] <= 1.5
    assert 2 <= roblox.sleeps[1] <= 2.5


async def test_backoff_waits_for_the_rate_limit_window_to_reset(roblox):
    roblox.catalog = [asset(1)]
    roblox.queue("/v1/search/items", httpx.Response(429, headers={"x-ratelimit-reset": "37"}))

    await rc.search_slot("hat", "Accessories")

    assert 37 <= roblox.sleeps[0] <= 37.5


async def test_gives_up_after_four_retries_and_caches_nothing(roblox):
    roblox.queue("/v1/search/items", *[httpx.Response(429) for _ in range(5)])

    with pytest.raises(rc.RobloxAPIError) as exc_info:
        await rc.search_slot("hat", "Accessories")

    assert exc_info.value.status_code == 429
    assert len(roblox.requests_to("/v1/search/items")) == 5
    assert len(roblox.sleeps) == 4

    roblox.catalog = [asset(1)]
    assert len(await rc.search_slot("hat", "Accessories")) == 1


async def test_bad_category_fails_fast_with_robloxs_message(roblox):
    error = {"errors": [{"code": 1, "message": "Category subcategory selection not supported."}]}
    roblox.queue("/v1/search/items", httpx.Response(400, json=error))

    with pytest.raises(rc.RobloxAPIError, match="selection not supported") as exc_info:
        await rc.search_slot("black hoodie", "Clothing", "Shirts")

    assert exc_info.value.status_code == 400
    assert roblox.sleeps == []


async def test_csrf_token_is_fetched_once_and_reused(roblox):
    roblox.catalog = [asset(1)]

    await rc.search_slot("hat", "Accessories")
    await rc.search_slot("cap", "Accessories")

    tokens = [request.headers.get("x-csrf-token") for request in roblox.requests_to("/v1/catalog/items/details")]
    assert tokens == [None, "token-1", "token-1"]


async def test_enrich_adds_thumbnails_and_exact_prices(roblox):
    roblox.catalog = [bundle(9, price=60)]
    roblox.economy = {
        1: economy(price=65),
        2: economy(price=0),  # free
        3: economy(price=5, for_sale=False),  # off sale; the 5 is stale
        4: economy(price=7, for_sale=False, resale=1440),  # limited, still sold by resellers
        # 5 is unknown to economy details -> 404
    }
    items = [{"id": item_id, "name": "x", "price": 10, "item_type": "Asset"} for item_id in range(1, 6)]
    items.append({"id": 9, "name": "Sneakers", "price": 55, "item_type": "Bundle"})

    enriched = await rc.enrich_items(items)

    assert [(item["price"], item["off_sale"]) for item in enriched] == [
        (65, False),
        (0, False),
        (None, True),
        (1440, False),
        (10, None),  # lookup failed: search price kept, status unknown
        (60, False),
    ]
    assert enriched[0]["thumbnail_url"] == "https://tr.fake/Asset/1.png"
    assert enriched[5]["thumbnail_url"] == "https://tr.fake/Bundle/9.png"
    assert "thumbnail_url" not in items[0]
    # Bundle ids are a separate id space, so they must never reach the asset endpoints.
    assert not roblox.requests_to("/v2/assets/9/details")
    assert "9" not in roblox.requests_to("/v1/assets")[0].url.params["assetIds"].split(",")


async def test_enrich_reuses_bundle_prices_from_the_search(roblox):
    roblox.catalog = [bundle(9, price=60)]
    items = await rc.search_slot("zombie", "AvatarAnimations", "AnimationBundles")
    roblox.requests.clear()

    [item] = await rc.enrich_items(items)

    assert (item["price"], item["off_sale"]) == (60, False)
    assert not roblox.requests_to("/v1/catalog/items/details")


async def test_enrich_batches_and_caches_lookups(roblox):
    roblox.economy = {item_id: economy(price=5) for item_id in range(1, 151)}
    items = [{"id": item_id, "name": "x", "price": 5, "item_type": "Asset"} for item_id in range(1, 151)]

    await rc.enrich_items(items)

    batches = [request.url.params["assetIds"].split(",") for request in roblox.requests_to("/v1/assets")]
    assert [len(batch) for batch in batches] == [100, 50]

    roblox.requests.clear()
    again = await rc.enrich_items(items)

    assert roblox.requests == []
    assert all(item["thumbnail_url"] and item["price"] == 5 for item in again)


async def test_enrich_survives_a_failing_thumbnail_service(roblox):
    roblox.queue("/v1/assets", *[httpx.Response(503) for _ in range(5)])
    roblox.economy = {1: economy(price=65)}

    [item] = await rc.enrich_items([{"id": 1, "name": "x", "price": 60, "item_type": "Asset"}])

    assert item["thumbnail_url"] is None
    assert item["price"] == 65


async def test_enrich_reuses_asset_prices_from_the_search_but_falls_back_when_unknown(roblox):
    # Item 1 has a catalog price (reused); item 2 has none (falls back to economy details).
    roblox.catalog = [asset(1, price=65), asset(2, price=None)]
    roblox.economy = {1: economy(price=999), 2: economy(price=42)}  # 999 would show if 1 hit economy
    items = await rc.search_slot("hoodie", "Clothing", "ShirtAccessories")
    roblox.requests.clear()

    enriched = await rc.enrich_items(items)

    assert {item["id"]: item["price"] for item in enriched} == {1: 65, 2: 42}
    assert not roblox.requests_to("/v2/assets/1/details")  # catalog price reused, no economy call
    assert roblox.requests_to("/v2/assets/2/details")  # unknown catalog price -> economy fallback
