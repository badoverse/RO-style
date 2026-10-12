"""Outfit generation: style description -> slot plan (LLM) -> catalog search -> best picks per slot (LLM).

Both LLM steps have a non-LLM fallback, so a failing (or unconfigured) model
degrades to unranked catalog results instead of an error.
"""

import asyncio
import logging
from typing import Any

import llm
import roblox_client

log = logging.getLogger(__name__)

MAX_SLOTS = 6
MAX_CANDIDATES = 100
PICKS_PER_SLOT = 3
FALLBACK_SLOTS = ("Hat", "Hair", "Shirt", "Pants")

# Prompt 1 speaks in simple slot words, but the catalog search only accepts its own
# enum pairs (it rejects Accessories/Hair and BodyParts with a 400), so slots are
# mapped to pairs checked against the live API. Shirts and pants use classic
# clothing: it's cheap enough for budget filters and fits every avatar type.
# Use ShirtAccessories/PantsAccessories instead for layered (3D) clothing.
SLOT_TARGETS: dict[str, tuple[str, str | None]] = {
    "Hat": ("Accessories", "HeadAccessories"),
    "Hair": ("Body", "HairAccessories"),
    "Shirt": ("Clothing", "ClassicShirts"),
    "Pants": ("Clothing", "ClassicPants"),
    "Face": ("Accessories", "FaceAccessories"),
    "Accessory": ("Accessories", None),
    "Animation": ("AvatarAnimations", "AnimationBundles"),
}
# For "Accessory" slots, prompt 1's subcategory says where the accessory goes.
ACCESSORY_TARGETS: dict[str, tuple[str, str | None]] = {
    "hat": SLOT_TARGETS["Hat"],
    "hats": SLOT_TARGETS["Hat"],
    "hair": SLOT_TARGETS["Hair"],
    "face": SLOT_TARGETS["Face"],
    "neck": ("Accessories", "NeckAccessories"),
    "shoulder": ("Accessories", "ShoulderAccessories"),
    "back": ("Accessories", "BackAccessories"),
    "waist": ("Accessories", "WaistAccessories"),
}
_SLOT_NAMES = {name.lower(): name for name in SLOT_TARGETS}

PLAN_SYSTEM = (
    "You translate avatar style descriptions into Roblox catalog search queries. "
    "Return ONLY a JSON array, no markdown, no explanation."
)
PLAN_USER = """Style description: "{description}"
Break it into 4-6 outfit slots. For each slot return:
- "slot": one of Hat, Hair, Shirt, Pants, Face, Accessory, Animation
- "keyword": 1-3 concrete words for Roblox catalog search (translate vibes into items, e.g. "emo" → "black hoodie", "scene hair")
- "category": one of Clothing, Accessories, AvatarAnimations, BodyParts
- "subcategory": for Clothing use Shirt or Pants; for Accessories the closest of Hats, Hair, Face, Neck, Shoulder, Back, Waist; else null"""

RANK_SYSTEM = (
    "You are a Roblox stylist. Given a style description and candidate items, pick the {n} best matches. "
    "Return ONLY a JSON array of chosen item ids, no markdown."
)
RANK_USER = """Style: "{description}"
Slot: {slot}
Candidates (id | name | price R$):
{candidates}
Rules: match the vibe not just keywords; prefer items wearable together; if nothing fits, return fewer than {n} — never force bad picks."""


async def generate_outfit(
    description: str,
    *,
    price_filter: bool = False,
    max_price: int | None = None,
    free_only: bool = False,
) -> list[dict]:
    """Plan outfit slots for a style, search each one and keep the best few items per slot.

    Returns [{"slot", "keyword", "items": [{"id", "name", "price", "thumbnail_url", "roblox_url"}]}].
    A slot whose catalog search fails comes back empty; raises
    roblox_client.RobloxAPIError only when every slot's search failed.
    """
    description = " ".join(description.split())
    plan = await plan_slots(description)
    picks = await asyncio.gather(
        *(_pick_items(description, slot, price_filter, max_price, free_only) for slot in plan)
    )
    if all(items is None for items in picks):
        raise roblox_client.RobloxAPIError("the catalog search failed for every slot")

    picks = [items or [] for items in picks]
    enriched = await roblox_client.enrich_items([item for items in picks for item in items])
    by_key = {(item["item_type"], item["id"]): item for item in enriched}
    return [
        {
            "slot": slot["slot"],
            "keyword": slot["keyword"],
            "items": [_outfit_item(by_key[(item["item_type"], item["id"])]) for item in items],
        }
        for slot, items in zip(plan, picks)
    ]


async def plan_slots(description: str) -> list[dict]:
    """Prompt 1: the slots to search, as {"slot", "keyword", "category", "subcategory"} with
    valid catalog enums. Falls back to default slots searched with the description itself."""
    try:
        return await llm.complete_json(
            PLAN_SYSTEM,
            PLAN_USER.format(description=description),
            label="plan",
            cache_key=("plan", description.lower()),
            parse=_parse_plan,
        )
    except llm.LLMError as exc:
        log.error("LLM slot planning failed, using default slots: %s", exc)
        keyword = _clean(description, 60)
        return [_slot(name, keyword) for name in FALLBACK_SLOTS]


async def rank_candidates(description: str, slot: str, candidates: list[dict], n: int = PICKS_PER_SLOT) -> list[dict]:
    """Prompt 2: up to n of the candidates that best fit the style, best first.
    Falls back to the catalog's relevance order when the LLM fails."""
    candidates = candidates[:MAX_CANDIDATES]
    if not candidates:
        return []
    # Candidates are numbered 1..N instead of showing 14-15 digit Roblox ids:
    # fewer tokens, and no long numbers for the model to mistype.
    lines = "\n".join(
        f"{number} | {_clean(item['name'] or '', 60)} | {'?' if item['price'] is None else item['price']}"
        for number, item in enumerate(candidates, 1)
    )
    try:
        numbers = await llm.complete_json(
            RANK_SYSTEM.format(n=n),
            RANK_USER.format(description=description, slot=slot, candidates=lines, n=n),
            label=f"rank {slot}",
            # The candidate list reflects the price filter, so a different filter is a different key.
            cache_key=("rank", description.lower(), slot, tuple((c["item_type"], c["id"]) for c in candidates), n),
            parse=lambda reply: _parse_picks(reply, len(candidates), n),
        )
    except llm.LLMError as exc:
        log.error("LLM ranking for %s failed, using catalog order: %s", slot, exc)
        return candidates[:n]
    return [candidates[number - 1] for number in numbers]


async def _pick_items(
    description: str, slot: dict, price_filter: bool, max_price: int | None, free_only: bool
) -> list[dict] | None:
    """The chosen items for one slot, or None when its catalog search failed."""
    try:
        candidates = await roblox_client.search_slot(
            slot["keyword"],
            slot["category"],
            slot["subcategory"],
            max_results=MAX_CANDIDATES,
            price_filter=price_filter,
            max_price=max_price,
            free_only=free_only,
        )
    except roblox_client.RobloxAPIError as exc:
        log.warning("catalog search for %s %r failed: %s", slot["slot"], slot["keyword"], exc)
        return None
    return await rank_candidates(description, slot["slot"], candidates)


def _parse_plan(reply: Any) -> list[dict]:
    """Turn prompt 1's reply into at most MAX_SLOTS distinct, searchable slots."""
    slots: list[dict] = []
    seen = set()
    for entry in _as_list(reply):
        if not isinstance(entry, dict):
            continue
        name = _SLOT_NAMES.get(str(entry.get("slot") or "").strip().lower())
        keyword = _clean(str(entry.get("keyword") or ""), 40)
        if not name or not keyword:
            continue
        slot = _slot(name, keyword, entry.get("subcategory"))
        identity = (slot["category"], slot["subcategory"], keyword.lower())
        if identity not in seen:
            seen.add(identity)
            slots.append(slot)
    if not slots:
        raise ValueError("no usable slots")
    return slots[:MAX_SLOTS]


def _parse_picks(reply: Any, count: int, n: int) -> list[int]:
    """Turn prompt 2's reply into up to n distinct candidate numbers. An empty array is
    a valid answer (nothing fits); an array without a single valid number is not."""
    entries = _as_list(reply)
    numbers: list[int] = []
    for entry in entries:
        if isinstance(entry, dict):  # e.g. [{"id": 3}]
            entry = entry.get("id")
        try:
            number = int(entry)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= count and number not in numbers:
            numbers.append(number)
    if entries and not numbers:
        raise ValueError("none of the returned ids are candidates")
    return numbers[:n]


def _slot(name: str, keyword: str, subcategory_hint: Any = None) -> dict:
    category, subcategory = SLOT_TARGETS[name]
    if name == "Accessory":
        category, subcategory = ACCESSORY_TARGETS.get(
            str(subcategory_hint or "").strip().lower(), (category, subcategory)
        )
    return {"slot": name, "keyword": keyword, "category": category, "subcategory": subcategory}


def _as_list(reply: Any) -> list:
    """The JSON array in a reply, also when the model wrapped it in an object."""
    if isinstance(reply, list):
        return reply
    if isinstance(reply, dict):
        for value in reply.values():
            if isinstance(value, list):
                return value
    raise ValueError("reply is not a JSON array")


def _outfit_item(item: dict) -> dict:
    # Bundle ids are their own id space: /catalog/<bundle id>/ is a 404.
    page = "bundles" if item["item_type"] == "Bundle" else "catalog"
    return {
        "id": item["id"],
        "name": item["name"] or "",
        "price": item["price"],
        "thumbnail_url": item.get("thumbnail_url"),
        "roblox_url": f"https://www.roblox.com/{page}/{item['id']}/",
    }


def _clean(text: str, limit: int) -> str:
    """One line of text without the "|" column separator, cut to limit characters."""
    return " ".join(text.replace("|", "/").split())[:limit]
