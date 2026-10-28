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
PICKS_PER_SLOT = 8
FALLBACK_SLOTS = ("Hat", "Hair", "Shirt", "Pants")

# "Create outfit" (vision): how many candidate thumbnails per slot the vision model sees,
# and an overall image cap for one call so we stay within free vision models' per-request
# limits. MAX_SLOTS * VISION_CANDIDATES_PER_SLOT should fit under VISION_MAX_IMAGES.
VISION_CANDIDATES_PER_SLOT = 3
VISION_MAX_IMAGES = 18

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

# "Create outfit" prompt: the model is shown an image per candidate (see _vision_user_content)
# and assembles one cohesive look, judging by how the items actually look together.
VISION_SYSTEM = (
    "You are a Roblox fashion stylist. You are shown images of candidate items for each "
    "outfit slot and must assemble ONE cohesive look. Judge by how the items actually look "
    "and whether they go well together, not just the names. Return ONLY a JSON object "
    "mapping each slot number to the chosen option number, no markdown, no explanation."
)


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


async def create_outfit(
    description: str,
    *,
    price_filter: bool = False,
    max_price: int | None = None,
    free_only: bool = False,
) -> list[dict]:
    """Assemble one cohesive look by having a VISION model actually look at item thumbnails.

    Same return shape as generate_outfit, but one chosen item per slot. Reuses the text
    planner, searches each slot, enriches a few candidates per slot for their thumbnails,
    then asks a vision model to pick the single option per slot that fits the style and
    looks good worn with the others. Falls back to the catalog's top result per slot when
    the vision model is unavailable or unusable. Raises roblox_client.RobloxAPIError only
    when every slot's search failed.
    """
    description = " ".join(description.split())
    plan = await plan_slots(description)
    candidate_lists = await asyncio.gather(
        *(_slot_candidates(slot, price_filter, max_price, free_only) for slot in plan)
    )
    if all(candidates is None for candidates in candidate_lists):
        raise roblox_client.RobloxAPIError("the catalog search failed for every slot")

    options = _vision_options(plan, candidate_lists)
    if not options:
        return []

    # Enrich only the shown candidates: that's where the thumbnail URLs for the images come from.
    enriched = await roblox_client.enrich_items([item for opt in options for item in opt["candidates"]])
    by_key = {(item["item_type"], item["id"]): item for item in enriched}
    for opt in options:
        opt["candidates"] = [by_key[(c["item_type"], c["id"])] for c in opt["candidates"]]

    picks = await _vision_pick(description, options)
    return [
        {
            "slot": opt["slot"],
            "keyword": opt["keyword"],
            "items": [_outfit_item(opt["candidates"][index])],
        }
        for opt, index in zip(options, picks)
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


async def _slot_candidates(
    slot: dict, price_filter: bool, max_price: int | None, free_only: bool
) -> list[dict] | None:
    """Catalog candidates for one slot, or None when its catalog search failed."""
    try:
        return await roblox_client.search_slot(
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


async def _pick_items(
    description: str, slot: dict, price_filter: bool, max_price: int | None, free_only: bool
) -> list[dict] | None:
    """The chosen items for one slot, or None when its catalog search failed."""
    candidates = await _slot_candidates(slot, price_filter, max_price, free_only)
    if candidates is None:
        return None
    return await rank_candidates(description, slot["slot"], candidates)


def _vision_options(plan: list[dict], candidate_lists: list[list[dict] | None]) -> list[dict]:
    """Per-slot candidate shortlists for the vision call, bounded by VISION_MAX_IMAGES.

    Slots whose search failed or returned nothing are dropped; once the image budget runs
    out, later slots are dropped too. Each kept entry is {"slot", "keyword", "candidates"}.
    """
    options: list[dict] = []
    budget = VISION_MAX_IMAGES
    for slot, candidates in zip(plan, candidate_lists):
        if not candidates or budget <= 0:
            continue
        take = min(VISION_CANDIDATES_PER_SLOT, len(candidates), budget)
        options.append({"slot": slot["slot"], "keyword": slot["keyword"], "candidates": candidates[:take]})
        budget -= take
    return options


async def _vision_pick(description: str, options: list[dict]) -> list[int]:
    """One chosen candidate index per option, via the vision model; catalog top (0) on failure."""
    counts = [len(opt["candidates"]) for opt in options]
    try:
        return await llm.complete_json(
            VISION_SYSTEM,
            _vision_user_content(description, options),
            label="create-outfit vision",
            cache_key=(
                "create",
                description.lower(),
                tuple((opt["slot"], tuple((c["item_type"], c["id"]) for c in opt["candidates"])) for opt in options),
            ),
            parse=lambda reply: _parse_vision_pick(reply, counts),
            model=llm.VISION_MODEL,
        )
    except llm.LLMError as exc:
        log.error("vision outfit selection failed, using catalog top picks: %s", exc)
        return [0] * len(options)


def _vision_user_content(description: str, options: list[dict]) -> list[dict]:
    """Multimodal user content: a text listing plus one labelled image per candidate.

    Each image is preceded by a "Slot i option j:" text part so the model can tie the
    picture to the option number it returns.
    """
    header = (
        f'Style to match: "{description}"\n\n'
        "Below are the candidate items for each outfit slot, each with its image. Choose "
        "exactly ONE option per slot so the chosen items share a consistent style and look "
        'good worn together. Return ONLY a JSON object mapping slot number to option number, '
        'e.g. {"1": 2, "2": 1}.'
    )
    parts: list[dict] = [{"type": "text", "text": header}]
    for i, opt in enumerate(options, 1):
        listing = "\n".join(
            f"  {j}. {_clean(c.get('name') or 'Unnamed item', 60)} — {_price_label(c.get('price'))}"
            for j, c in enumerate(opt["candidates"], 1)
        )
        parts.append({"type": "text", "text": f"\nSlot {i} ({opt['slot']}):\n{listing}"})
        for j, candidate in enumerate(opt["candidates"], 1):
            parts.append({"type": "text", "text": f"Slot {i} option {j}:"})
            url = candidate.get("thumbnail_url")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": "(no image available)"})
    return parts


def _parse_vision_pick(reply: Any, option_counts: list[int]) -> list[int]:
    """Turn the vision reply into one 0-based candidate index per slot.

    Accepts {"1": 2, ...} or a bare array [2, 1, ...] (one entry per slot, in order).
    An out-of-range or missing slot defaults to the catalog's top candidate (index 0).
    Raises ValueError when a non-empty reply yields no valid pick, so llm re-asks once.
    """
    if isinstance(reply, dict):
        picks = {_as_int(key): value for key, value in reply.items()}
    elif isinstance(reply, list):
        picks = {i: value for i, value in enumerate(reply, 1)}
    else:
        raise ValueError("expected a JSON object or array")

    indices: list[int] = []
    any_valid = False
    for slot_number, count in enumerate(option_counts, 1):
        choice = picks.get(slot_number)
        if isinstance(choice, dict):  # e.g. {"option": 2}
            choice = choice.get("option") or choice.get("id")
        option = _as_int(choice)
        if option is not None and 1 <= option <= count:
            indices.append(option - 1)
            any_valid = True
        else:
            indices.append(0)  # fall back to this slot's top candidate
    if reply and not any_valid:
        raise ValueError("no valid option numbers")
    return indices


def _price_label(price: int | None) -> str:
    if price == 0:
        return "Free"
    if price is None:
        return "off sale"
    return f"R$ {price}"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
