# RO-style

A Roblox avatar stylist. Describe a look ("emo kid with dark clothes and headphones") and get matching items from the
Roblox catalog, grouped by outfit slot, with thumbnails, prices and links to each item's page.

A FastAPI backend does the work. A plain HTML/CSS/JS frontend (no build step) is served by the
same server.

## How it works

1. **Plan:** an LLM turns the description into 4–6 outfit slots (Hat, Hair, Shirt, Pants,
   Face, Accessory, Animation), each with a short catalog search keyword.
2. **Search:** each slot is searched in the Roblox catalog, giving up to 100 candidates with
   names and prices. The optional price filter (max Robux or free only) is applied here.
3. **Rank:** the LLM picks the 3 best candidates per slot. It only sees names and prices.
4. **Enrich:** the picks get thumbnails and exact current prices, and the page shows them.

If the LLM is unavailable, misconfigured or too slow, generation still works: it falls back to
default slots and the catalog's own top results, and logs an error.

## Requirements

- Python 3.10 or newer (developed on 3.14)
- An [OpenRouter](https://openrouter.ai/keys) API key. The free tier works.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the project root (or in `backend/`):

```dotenv
OPENROUTER_API_KEY=sk-or-...
# Optional. INFO logs every Roblox request and LLM call.
LOG_LEVEL=INFO
```

`.env` is git-ignored. Without a key the app still runs, but results are unranked.

## Run

From `backend/`:

```bash
python -m uvicorn app:app --reload --port 8000
```

Open **http://127.0.0.1:8000/**.

- The app is `app:app` (`backend/app.py`). `app.main:app` fails with
  `Could not import module "app.main"`.
- Open the page from the FastAPI server, not from VS Code's Live Server or a `file://` URL.
  The page calls `/generate` on whatever server loaded it, and a static server can't answer.

## API

Interactive docs: http://127.0.0.1:8000/docs

### `POST /generate`

```json
{
  "description": "emo kid with dark clothes and headphones",
  "price_filter_enabled": true,
  "max_price": 100,
  "free_only": false
}
```

Response:

```json
{
  "slots": [
    {
      "slot": "Shirt",
      "keyword": "black hoodie",
      "items": [
        {
          "id": 11903161564,
          "name": "Black Hoodie",
          "price": 5,
          "thumbnail_url": "https://tr.rbxcdn.com/...",
          "roblox_url": "https://www.roblox.com/catalog/11903161564/"
        }
      ]
    }
  ]
}
```

- `price` is in Robux: `0` means free, `null` means the item can't be bought right now (or its
  price is unknown).
- Bundles (animation packs, shoes) link to `/bundles/{id}/`; everything else links to
  `/catalog/{id}/`.
- A slot can come back with no items, for example when nothing matches the price filter.
- The endpoint returns `502` only if the catalog search failed for every slot.

### `GET /test-search`

Runs one raw catalog search plus enrichment, for checking the Roblox layer on its own:

```
/test-search?keyword=black%20hoodie&category=Clothing&subcategory=ShirtAccessories
```

`category` and `subcategory` must be Roblox's enum names: `Clothing/ClassicShirts`,
`Body/HairAccessories`, `Accessories/HeadAccessories` and so on. Mismatched pairs return
`400` with Roblox's error message.

## Project layout

```
backend/
  app.py             FastAPI app: /generate, /test-search, serves frontend/ at /
  stylist.py         Generation pipeline, both prompts, slot -> catalog category mapping
  llm.py             OpenRouter client: retries, lenient JSON parsing, deadline, 24h cache
  roblox_client.py   Roblox catalog/thumbnail/economy APIs: rate limits, retries, caching
  cache.py           Small in-memory TTL cache
  tests/             Offline tests (fake Roblox and fake LLM, no network)
frontend/
  index.html, style.css, app.js
```

## Limits worth knowing

**OpenRouter free tier: 50 free-model requests a day.** One uncached generation uses 6–7
(one planning call plus one ranking call per slot). Repeating an identical request is free,
because LLM replies are cached for 24 hours.

**Roblox rate limits.** The catalog endpoints used here are public but unofficial. The
limits observed are 12 searches and 10 details calls per minute. One uncached generation
uses up to 6 of each. Past the limit, requests wait for the rate-limit window to reset (up to
60 s) instead of failing. Results are cached: searches for 2 h, thumbnails for 30 min, exact
prices for 10 min. All caches are in memory, so they reset on restart.

**Model settings.** The default model, `nvidia/nemotron-3-ultra-550b-a55b:free`, is a
reasoning model. Reasoning is switched off (`EXTRA_BODY` in `llm.py`): with it on, one ranking
call took 163 s, compared with about 1 s with it off. Each LLM step has a 45 s deadline
(`DEADLINE_SECONDS`), after which that step falls back to catalog results.

## Customizing

- **Model:** `MODEL` in `backend/llm.py`.
- **Classic vs layered clothing:** `SLOT_TARGETS` in `backend/stylist.py` maps each slot to a
  catalog category. Shirts and pants use classic clothing (from about 5 R$, works on every
  avatar). Switch to `ShirtAccessories`/`PantsAccessories` for layered 3D clothing, which
  costs about 60 R$ and up.
- **Picks and candidates per slot:** `PICKS_PER_SLOT` and `MAX_CANDIDATES` in
  `backend/stylist.py`.

## Tests

```bash
cd backend
python -m pytest
```

The tests use fakes for Roblox and the LLM, so they make no network calls and use no
quota. They need the pinned `openai==3.29.0` from `requirements.txt`.

## Known issues

- The planner sometimes files an item under an odd accessory location. For example,
  "headphones" became a Back accessory, so no headphones were found, and the ranker still
  returned 3 unrelated items.
- With a price limit, the Animation slot is usually empty: animation packs cost more than
  100 R$.
- The Roblox endpoints are unofficial and can change without notice. In October 2026 the
  category names, the free-item filter and the response shapes all already differed from
  older documentation.
