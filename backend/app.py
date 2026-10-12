"""Roblox Avatar Stylist API: LLM outfit generation on top of the Roblox catalog layer.

Run from backend/:
    pip install -r requirements.txt
    cp .env.example .env    # then add your OPENROUTER_API_KEY
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/ for the app (served from ../frontend), or try the API:
    curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
        -d '{"description": "emo skater", "price_filter_enabled": true, "max_price": 100, "free_only": false}'
    http://127.0.0.1:8000/test-search?keyword=black%20hoodie&category=Clothing&subcategory=ShirtAccessories
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

import llm
import roblox_client
import stylist

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
# roblox_client and llm log every request themselves, with timings and budgets.
for http_logger in ("httpx", "httpx2"):
    logging.getLogger(http_logger).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await roblox_client.aclose()
    await llm.aclose()


app = FastAPI(title="Roblox Avatar Stylist", lifespan=lifespan)


class GenerateRequest(BaseModel):
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    price_filter_enabled: bool = False
    max_price: int | None = Field(default=None, ge=0)
    free_only: bool = False


class OutfitItem(BaseModel):
    id: int
    name: str
    price: int | None
    thumbnail_url: str | None
    roblox_url: str


class OutfitSlot(BaseModel):
    slot: str
    keyword: str
    items: list[OutfitItem]


class GenerateResponse(BaseModel):
    slots: list[OutfitSlot]


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> dict:
    """Turn a style description into a few picks per outfit slot.

    Picks are LLM-ranked; if the LLM is unavailable they are the catalog's top results.
    """
    try:
        slots = await stylist.generate_outfit(
            request.description,
            price_filter=request.price_filter_enabled,
            max_price=request.max_price,
            free_only=request.free_only,
        )
    except roblox_client.RobloxAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"slots": slots}


@app.get("/test-search")
async def test_search(
    keyword: str,
    category: str,
    subcategory: str | None = None,
    max_results: int = Query(100, ge=1, le=500),
    price_filter: bool = False,
    max_price: int | None = Query(None, ge=0),
    free_only: bool = False,
) -> dict:
    """Search one slot and enrich the results: raw output for checking the Roblox layer."""
    try:
        items = await roblox_client.search_slot(
            keyword,
            category,
            subcategory,
            max_results=max_results,
            price_filter=price_filter,
            max_price=max_price,
            free_only=free_only,
        )
        enriched = await roblox_client.enrich_items(items)
    except roblox_client.RobloxAPIError as exc:
        # A 400 from Roblox means bad input (e.g. a category/subcategory mismatch).
        raise HTTPException(status_code=400 if exc.status_code == 400 else 502, detail=str(exc)) from exc
    return {"count": len(enriched), "items": enriched}


# Mounted last: routes match in order, so the API routes above win over files at "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
