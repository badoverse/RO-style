"""Roblox catalog API layer: slot search, enrichment, rate-limit handling and caching.

Talks to Roblox's public, unauthenticated web APIs. Per-IP limits as reported by
the x-ratelimit-* response headers (checked Oct 2026):

    catalog search         catalog.roblox.com/v1/search/items          12 req / 60 s
    catalog batch details  catalog.roblox.com/v1/catalog/items/details 10 req / 60 s
    asset details          economy.roblox.com/v2/assets/{id}/details 1000 req / 60 s
    asset thumbnails       thumbnails.roblox.com/v1/assets             50 req / 1 s
    bundle thumbnails      thumbnails.roblox.com/v1/bundles/thumbnails 60 req / 60 s

Search is the bottleneck, so results are cached for 2 hours and fetched in the
largest pages the endpoint allows.
"""

import asyncio
import logging
import random
import time
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from cache import TTLCache

log = logging.getLogger(__name__)

SEARCH_URL = "https://catalog.roblox.com/v1/search/items"
CATALOG_DETAILS_URL = "https://catalog.roblox.com/v1/catalog/items/details"
ASSET_DETAILS_URL = "https://economy.roblox.com/v2/assets/{asset_id}/details"
# item_type -> (endpoint, id query param)
THUMBNAIL_ENDPOINTS = {
    "Asset": ("https://thumbnails.roblox.com/v1/assets", "assetIds"),
    "Bundle": ("https://thumbnails.roblox.com/v1/bundles/thumbnails", "bundleIds"),
}

# limit=120 is accepted (30 is not the max), so 100 results cost one request
# out of the 12/min search budget instead of four.
SEARCH_PAGE_SIZE = 120
CATALOG_DETAILS_BATCH_SIZE = 120
THUMBNAIL_BATCH_SIZE = 100
# Economy details has no batch form, but its 1000/min budget allows some fan-out.
ASSET_DETAILS_CONCURRENCY = 8

PAGE_DELAY_SECONDS = 0.5
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 10.0
USER_AGENT = "RobloxAvatarStylist/0.1"

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# Assets and bundles have separate id spaces, so items are keyed by both.
ItemKey = tuple[str, int]  # (item_type, id)

_search_cache = TTLCache(ttl_seconds=2 * 60 * 60, max_entries=1000)
_thumbnail_cache = TTLCache(ttl_seconds=30 * 60)  # image URLs expire; keep this short
_price_cache = TTLCache(ttl_seconds=10 * 60)

_client: httpx.AsyncClient | None = None
_csrf_token: str | None = None
_sleep = asyncio.sleep  # module-level so tests can skip the waits


class RobloxAPIError(Exception):
    """A Roblox request failed with a non-retryable error or ran out of retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


async def search_slot(
    keyword: str,
    category: str,
    subcategory: str | None = None,
    max_results: int = 100,
    price_filter: bool = False,
    max_price: int | None = None,
    free_only: bool = False,
) -> list[dict]:
    """Search the catalog for one outfit slot.

    category/subcategory are Roblox enum names, e.g. "Clothing"/"ShirtAccessories"
    or "Accessories"/"HeadAccessories"; Roblox rejects mismatched pairs with a 400.
    Returns up to max_results items in relevance order, each
    {"id", "name", "price", "item_type"}: price is in Robux (0 = free, None = not
    buyable right now or unknown) and item_type is "Asset" or "Bundle" (shoes, bodies, ...).
    """
    if max_results <= 0:
        return []
    price_cap = _price_cap(price_filter, max_price, free_only)
    keyword = keyword.strip()
    cache_key = (keyword.lower(), category, subcategory, price_cap, max_results)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        log.info("search cache hit %s", cache_key)
        return [dict(item) for item in cached]

    params: dict[str, Any] = {"category": category, "limit": SEARCH_PAGE_SIZE, "sortType": "Relevance"}
    if keyword:
        params["keyword"] = keyword
    if subcategory:
        params["subcategory"] = subcategory
    if price_cap is not None:
        # maxPrice=0 is what filters to free items: salesTypeFilter=1 means "All",
        # and minPrice=0 on its own filters nothing.
        params["maxPrice"] = price_cap

    hits = await _search_hits(params, max_results)
    # Search hits are bare ids; names and prices come from one batch details call.
    details = await _catalog_details(hits)
    items = []
    for item_type, item_id in hits:
        detail = details.get((item_type, item_id))
        if detail is None:  # moderated or deleted since it was indexed
            continue
        price, off_sale = _catalog_price(detail)
        if item_type == "Bundle":
            # Catalog details is also enrich_items' exact-price source for bundles; spare it the call.
            _price_cache.set((item_type, item_id), (price, off_sale))
        items.append({"id": item_id, "name": detail.get("name"), "price": price, "item_type": item_type})
    if price_cap is not None:
        # The search index can lag behind price changes; enforce the cap on fresh prices.
        items = [item for item in items if item["price"] is not None and item["price"] <= price_cap]

    _search_cache.set(cache_key, items)
    return [dict(item) for item in items]


async def enrich_items(items: list[dict]) -> list[dict]:
    """Return copies of search_slot items with thumbnail_url, exact price and off_sale added.

    off_sale is True when the item can't be bought right now, None when unknown.
    Failed lookups leave thumbnail_url None, or keep the search price with
    off_sale None, rather than failing the whole batch.
    """
    keys = [(item["item_type"], item["id"]) for item in items]
    thumbnails, prices = await asyncio.gather(_thumbnails(keys), _exact_prices(keys))
    enriched = []
    for item, key in zip(items, keys):
        price, off_sale = prices.get(key, (item.get("price"), None))
        enriched.append({**item, "price": price, "off_sale": off_sale, "thumbnail_url": thumbnails.get(key)})
    return enriched


async def aclose() -> None:
    """Close the shared HTTP client; call on app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _price_cap(price_filter: bool, max_price: int | None, free_only: bool) -> int | None:
    """Highest acceptable price in Robux (0 = free only), or None when unfiltered."""
    if free_only:
        return 0
    if price_filter and max_price is not None:
        return max(max_price, 0)
    return None


async def _search_hits(params: dict[str, Any], max_results: int) -> list[ItemKey]:
    """Follow nextPageCursor until there are max_results unique hits or no more pages."""
    hits: list[ItemKey] = []
    seen: set[ItemKey] = set()
    cursor = None
    while True:
        page = await _request("GET", SEARCH_URL, params={**params, "cursor": cursor} if cursor else params)
        for hit in page.get("data") or []:
            key = (hit.get("itemType"), hit.get("id"))
            if key not in seen:
                seen.add(key)
                hits.append(key)
        cursor = page.get("nextPageCursor")
        if len(hits) >= max_results or not cursor:
            return hits[:max_results]
        await _sleep(PAGE_DELAY_SECONDS)


async def _catalog_details(keys: Sequence[ItemKey]) -> dict[ItemKey, dict]:
    """Batch-fetch catalog records (name, price, priceStatus, ...) for assets and bundles."""
    details: dict[ItemKey, dict] = {}
    for chunk in _chunks(keys, CATALOG_DETAILS_BATCH_SIZE):
        body = {"items": [{"itemType": item_type, "id": item_id} for item_type, item_id in chunk]}
        payload = await _request("POST", CATALOG_DETAILS_URL, json=body)
        for detail in payload.get("data") or []:
            details[(detail.get("itemType"), detail.get("id"))] = detail
    return details


def _catalog_price(detail: dict) -> tuple[int | None, bool | None]:
    """(price, off_sale) from a catalog details record."""
    status = detail.get("priceStatus")
    if status == "Free":
        return 0, False
    if status in ("Off Sale", "No Resellers"):
        return None, True  # "price" can still hold a stale value here
    # lowestPrice also counts resale listings, so it's what the item costs right now.
    for field in ("lowestPrice", "price"):
        if isinstance(detail.get(field), int):
            return detail[field], False
    return None, None


def _economy_price(detail: dict) -> tuple[int | None, bool | None]:
    """(price, off_sale) from economy asset details, where most fields can be null."""
    offers = []
    if detail.get("IsForSale"):
        price = 0 if detail.get("IsPublicDomain") else detail.get("PriceInRobux")
        if isinstance(price, int):
            offers.append(price)
    resale = (detail.get("CollectiblesItemDetails") or {}).get("CollectibleLowestResalePrice")
    if isinstance(resale, int):
        offers.append(resale)  # limiteds stay buyable from resellers after going off sale
    if offers:
        return min(offers), False
    # Off-sale items keep a stale PriceInRobux, so IsForSale has the final say.
    return None, (True if detail.get("IsForSale") is False else None)


async def _thumbnails(keys: Sequence[ItemKey]) -> dict[ItemKey, str]:
    """420x420 PNG thumbnail URLs, fetched in batches of 100 and cached for 30 minutes."""
    urls: dict[ItemKey, str] = {}
    for key in keys:
        cached = _thumbnail_cache.get(key)
        if cached is not None:
            urls[key] = cached
    missing = [key for key in dict.fromkeys(keys) if key not in urls]
    log.info("thumbnails: %d cached, %d to fetch", len(urls), len(missing))

    for item_type, (endpoint, id_param) in THUMBNAIL_ENDPOINTS.items():
        ids = [item_id for kind, item_id in missing if kind == item_type]
        for chunk in _chunks(ids, THUMBNAIL_BATCH_SIZE):
            params = {id_param: ",".join(map(str, chunk)), "size": "420x420", "format": "Png"}
            try:
                payload = await _request("GET", endpoint, params=params)
            except RobloxAPIError as exc:
                log.warning("thumbnails for %d %s items unavailable: %s", len(chunk), item_type, exc)
                continue
            for thumb in payload.get("data") or []:
                # "Pending"/"Blocked" thumbnails have no URL yet; don't cache those.
                if thumb.get("state") == "Completed" and thumb.get("imageUrl"):
                    key = (item_type, thumb.get("targetId"))
                    urls[key] = thumb["imageUrl"]
                    _thumbnail_cache.set(key, thumb["imageUrl"])
    return urls


async def _exact_prices(keys: Sequence[ItemKey]) -> dict[ItemKey, tuple[int | None, bool | None]]:
    """(price, off_sale) per item: economy details for assets, catalog details for bundles."""
    prices: dict[ItemKey, tuple[int | None, bool | None]] = {}
    for key in keys:
        cached = _price_cache.get(key)
        if cached is not None:
            prices[key] = cached
    missing = [key for key in dict.fromkeys(keys) if key not in prices]
    log.info("exact prices: %d cached, %d to fetch", len(prices), len(missing))

    semaphore = asyncio.Semaphore(ASSET_DETAILS_CONCURRENCY)

    async def fetch_asset(asset_id: int) -> None:
        async with semaphore:
            try:
                detail = await _request("GET", ASSET_DETAILS_URL.format(asset_id=asset_id))
            except RobloxAPIError as exc:
                log.warning("exact price for asset %s unavailable: %s", asset_id, exc)
                return
        prices[("Asset", asset_id)] = _economy_price(detail)

    await asyncio.gather(*(fetch_asset(item_id) for item_type, item_id in missing if item_type == "Asset"))

    # Economy details only knows assets (a bundle id would hit an unrelated asset).
    bundle_keys = [key for key in missing if key[0] == "Bundle"]
    if bundle_keys:
        try:
            details = await _catalog_details(bundle_keys)
        except RobloxAPIError as exc:
            log.warning("exact prices for %d bundles unavailable: %s", len(bundle_keys), exc)
            details = {}
        for key, detail in details.items():
            prices[key] = _catalog_price(detail)

    for key in missing:
        if key in prices:
            _price_cache.set(key, prices[key])
    return prices


async def _request(method: str, url: str, *, params: dict | None = None, json: Any = None) -> Any:
    """Send one Roblox API request and return the decoded JSON body.

    Every attempt is logged with its remaining rate-limit budget. 429s, 5xx and
    network errors are retried up to MAX_RETRIES times with exponential backoff,
    stretched to the rate-limit reset time when Roblox reports one. POSTs pick up
    the CSRF token Roblox hands out with its first 403. Raises RobloxAPIError.
    """
    global _csrf_token
    retries = 0
    csrf_refreshed = False
    while True:
        headers = {"x-csrf-token": _csrf_token} if method == "POST" and _csrf_token else None
        started = time.monotonic()
        try:
            response = await _http().request(method, url, params=params, json=json, headers=headers)
        except httpx.TransportError as exc:
            if retries >= MAX_RETRIES:
                raise RobloxAPIError(f"{method} {url} failed after {retries} retries: {exc!r}") from exc
            retries += 1
            delay = _backoff_delay(retries)
            log.warning("%s %s failed (%r); retry %d/%d in %.1fs", method, url, exc, retries, MAX_RETRIES, delay)
            await _sleep(delay)
            continue
        _log_response(response, time.monotonic() - started)

        status = response.status_code
        if status == 403 and method == "POST" and not csrf_refreshed and response.headers.get("x-csrf-token"):
            _csrf_token = response.headers["x-csrf-token"]
            csrf_refreshed = True
            log.info("got a CSRF token from %s, resending", response.url.host)
            continue
        if status in _RETRYABLE_STATUSES and retries < MAX_RETRIES:
            retries += 1
            delay = _backoff_delay(retries, response)
            log.warning("%d from %s; backing off %.1fs (retry %d/%d)", status, response.url.host, delay, retries, MAX_RETRIES)
            await _sleep(delay)
            continue
        if response.is_error:
            raise RobloxAPIError(
                f"Roblox returned {status} for {method} {_short_url(response.url)}: {_error_message(response)}",
                status,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RobloxAPIError(f"{method} {_short_url(response.url)} returned invalid JSON", status) from exc


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
    return _client


def _backoff_delay(retry: int, response: httpx.Response | None = None) -> float:
    """1s, 2s, 4s, 8s (+ jitter), or longer when Roblox says when to come back."""
    delay = BACKOFF_BASE_SECONDS * 2 ** (retry - 1)
    if response is not None:
        hints = [response.headers.get("retry-after")]
        if response.status_code == 429:
            # Seconds until the rate-limit window resets; retrying sooner just burns a retry.
            hints.append(response.headers.get("x-ratelimit-reset"))
        for hint in hints:
            try:
                delay = max(delay, float(hint))
            except (TypeError, ValueError):
                pass
    return min(delay, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5)


def _log_response(response: httpx.Response, elapsed: float) -> None:
    budget = ""
    remaining = response.headers.get("x-ratelimit-remaining")
    if remaining is not None:
        # x-ratelimit-limit looks like "12, 12;w=60": 12 requests per 60-second window.
        policy = response.headers.get("x-ratelimit-limit", "").split(",")[-1].strip()
        quota, _, window = policy.partition(";w=")
        reset = response.headers.get("x-ratelimit-reset", "?")
        budget = f" [rate limit {remaining}/{quota} left per {window}s, resets in {reset}s]"
    log.info(
        "%s %s -> %d in %.0fms%s",
        response.request.method, _short_url(response.url), response.status_code, elapsed * 1000, budget,
    )


def _error_message(response: httpx.Response) -> str:
    """Roblox's {"errors": [{"message": ...}]} text, falling back to the raw body."""
    try:
        messages = [error["message"] for error in response.json().get("errors") or [] if error.get("message")]
    except (ValueError, AttributeError, KeyError, TypeError):
        messages = []
    return "; ".join(messages) or response.text[:200] or response.reason_phrase


def _short_url(url: httpx.URL | str, limit: int = 160) -> str:
    text = str(url)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _chunks(seq: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]
