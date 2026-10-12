import pytest
from fastapi.testclient import TestClient

import roblox_client
import stylist
from app import app

client = TestClient(app)

OUTFIT = [
    {
        "slot": "Hat",
        "keyword": "beanie",
        "items": [
            {"id": 1, "name": "Beanie", "price": 5, "thumbnail_url": None, "roblox_url": "https://www.roblox.com/catalog/1/"}
        ],
    }
]


def test_generate_passes_the_request_to_the_stylist(monkeypatch):
    calls = []

    async def generate_outfit(description, **filters):
        calls.append((description, filters))
        return OUTFIT

    monkeypatch.setattr(stylist, "generate_outfit", generate_outfit)

    response = client.post(
        "/generate",
        json={"description": "  emo  ", "price_filter_enabled": True, "max_price": 50, "free_only": False},
    )

    assert response.status_code == 200
    assert response.json() == {"slots": OUTFIT}
    assert calls == [("emo", {"price_filter": True, "max_price": 50, "free_only": False})]


@pytest.mark.parametrize("body", [{}, {"description": "   "}, {"description": "emo", "max_price": -1}])
def test_generate_rejects_bad_requests(body):
    assert client.post("/generate", json=body).status_code == 422


def test_generate_reports_a_catalog_outage_as_502(monkeypatch):
    async def generate_outfit(description, **filters):
        raise roblox_client.RobloxAPIError("the catalog search failed for every slot")

    monkeypatch.setattr(stylist, "generate_outfit", generate_outfit)

    assert client.post("/generate", json={"description": "emo"}).status_code == 502


def test_the_frontend_is_served_at_the_root():
    page = client.get("/")

    assert page.status_code == 200
    assert "Generate outfit" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_test_search_still_works(monkeypatch):
    async def search_slot(*args, **kwargs):
        return [{"id": 1, "name": "Hoodie", "price": 65, "item_type": "Asset"}]

    async def enrich_items(items):
        return [{**entry, "off_sale": False, "thumbnail_url": "https://tr.fake/1.png"} for entry in items]

    monkeypatch.setattr(roblox_client, "search_slot", search_slot)
    monkeypatch.setattr(roblox_client, "enrich_items", enrich_items)

    response = client.get("/test-search", params={"keyword": "hoodie", "category": "Clothing"})

    assert response.status_code == 200
    assert response.json()["count"] == 1
