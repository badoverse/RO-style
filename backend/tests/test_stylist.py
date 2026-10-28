import json

import pytest

import llm
import roblox_client
import stylist

pytestmark = pytest.mark.anyio


def item(item_id, name=None, price=10, item_type="Asset"):
    """A search_slot result."""
    return {"id": item_id, "name": name or f"Item {item_id}", "price": price, "item_type": item_type}


class FakeCatalog:
    """Stands in for roblox_client.search_slot and enrich_items."""

    def __init__(self):
        self.results: dict = {}  # keyword -> candidates, or an exception to raise
        self.searches: list[dict] = []

    async def search_slot(self, keyword, category, subcategory=None, **filters):
        self.searches.append({"keyword": keyword, "category": category, "subcategory": subcategory, **filters})
        result = self.results.get(keyword, [])
        if isinstance(result, Exception):
            raise result
        return [dict(entry) for entry in result]

    async def enrich_items(self, items):
        return [{**entry, "thumbnail_url": f"https://tr.fake/{entry['id']}.png"} for entry in items]


@pytest.fixture
def catalog(monkeypatch):
    fake = FakeCatalog()
    monkeypatch.setattr(roblox_client, "search_slot", fake.search_slot)
    monkeypatch.setattr(roblox_client, "enrich_items", fake.enrich_items)
    return fake


def answer_with(plan, picks_by_slot):
    """A FakeLLM responder: the plan for prompt 1, picks_by_slot[slot] for prompt 2."""

    def respond(messages):
        user = messages[1]["content"]
        if user.startswith("Style description"):
            return json.dumps(plan)
        slot = user.split("Slot: ", 1)[1].split("\n", 1)[0]
        return json.dumps(picks_by_slot[slot])

    return respond


async def test_plan_maps_slot_words_to_searchable_categories(fake_llm):
    fake_llm.replies = [
        json.dumps(
            [
                {"slot": "Hair", "keyword": "scene hair", "category": "Accessories", "subcategory": "Hair"},
                {"slot": "shirt", "keyword": "black hoodie", "category": "Clothing", "subcategory": "Shirt"},
                {"slot": "Accessory", "keyword": "chain", "category": "Accessories", "subcategory": "Neck"},
                {"slot": "Face", "keyword": "emo face", "category": "BodyParts", "subcategory": None},
                {"slot": "Animation", "keyword": "zombie", "category": "AvatarAnimations", "subcategory": None},
                {"slot": "Shoes", "keyword": "vans"},  # not a slot
                {"slot": "Hat", "keyword": ""},  # no keyword
                {"slot": "Hair", "keyword": "Scene  Hair", "subcategory": "Hair"},  # repeat
            ]
        )
    ]

    plan = await stylist.plan_slots("emo")

    assert [(slot["slot"], slot["keyword"], slot["category"], slot["subcategory"]) for slot in plan] == [
        ("Hair", "scene hair", "Body", "HairAccessories"),
        ("Shirt", "black hoodie", "Clothing", "ClassicShirts"),
        ("Accessory", "chain", "Accessories", "NeckAccessories"),
        ("Face", "emo face", "Accessories", "FaceAccessories"),
        ("Animation", "zombie", "AvatarAnimations", "AnimationBundles"),
    ]


async def test_plan_keeps_at_most_six_slots(fake_llm):
    fake_llm.replies = [json.dumps([{"slot": "Hat", "keyword": f"hat {i}"} for i in range(9)])]

    assert len(await stylist.plan_slots("hats")) == 6


async def test_plan_falls_back_to_default_slots(fake_llm):
    fake_llm.replies = ["Sure! An emo look needs...", "I'd suggest a hoodie."]

    plan = await stylist.plan_slots("cozy dark academia")

    assert [(slot["slot"], slot["keyword"]) for slot in plan] == [
        ("Hat", "cozy dark academia"),
        ("Hair", "cozy dark academia"),
        ("Shirt", "cozy dark academia"),
        ("Pants", "cozy dark academia"),
    ]


async def test_rank_prompt_holds_only_numbered_names_and_prices(fake_llm):
    candidates = [item(91519387140359, "Black | Hoodie\n(emo)", 65), item(2, "Grey Tee", None), item(3, "Free Cap", 0)]
    fake_llm.replies = ["[3, 1]"]

    picks = await stylist.rank_candidates("emo", "Shirt", candidates)

    assert [pick["id"] for pick in picks] == [3, 91519387140359]
    [system, user] = fake_llm.calls[0]["messages"]
    assert f"pick the {stylist.PICKS_PER_SLOT} best matches" in system["content"]
    assert "Slot: Shirt\nCandidates (id | name | price R$):\n" in user["content"]
    assert "1 | Black / Hoodie (emo) | 65\n2 | Grey Tee | ?\n3 | Free Cap | 0\n" in user["content"]
    assert "91519387140359" not in user["content"]


async def test_rank_drops_unknown_and_repeated_ids_and_caps_at_n(fake_llm):
    fake_llm.replies = ['[5, 99, 5, "2", {"id": 1}, 3]']

    picks = await stylist.rank_candidates("emo", "Hat", [item(i) for i in range(1, 6)], n=3)

    assert [pick["id"] for pick in picks] == [5, 2, 1]  # 99 out of range, repeat 5 dropped, capped to 3


async def test_rank_may_pick_nothing(fake_llm):
    fake_llm.replies = ["[]"]

    assert await stylist.rank_candidates("emo", "Hat", [item(1)]) == []


async def test_rank_falls_back_to_catalog_order(fake_llm):
    fake_llm.replies = ["The first one!", "Definitely the first one."]

    picks = await stylist.rank_candidates("emo", "Hat", [item(i) for i in range(1, 6)])

    assert [pick["id"] for pick in picks] == [1, 2, 3, 4, 5]  # catalog order, up to PICKS_PER_SLOT


async def test_rank_sends_at_most_100_candidates(fake_llm):
    fake_llm.replies = ["[100]"]

    [pick] = await stylist.rank_candidates("emo", "Hat", [item(i) for i in range(1, 151)])

    assert pick["id"] == 100
    prompt = fake_llm.calls[0]["messages"][1]["content"]
    assert len([line for line in prompt.splitlines() if line[:1].isdigit()]) == 100


async def test_generate_outfit(fake_llm, catalog):
    plan = [
        {"slot": "Hat", "keyword": "beanie", "category": "Accessories", "subcategory": "Hats"},
        {"slot": "Animation", "keyword": "zombie", "category": "AvatarAnimations", "subcategory": None},
    ]
    fake_llm.respond = answer_with(plan, {"Hat": [2, 1], "Animation": [1]})
    catalog.results = {
        "beanie": [item(11, "Grey Beanie", 5), item(12, "Black Beanie", 7)],
        "zombie": [item(21, "Zombie Pack", 250, item_type="Bundle")],
    }

    slots = await stylist.generate_outfit("  emo   skater ", price_filter=True, max_price=300)

    assert slots == [
        {
            "slot": "Hat",
            "keyword": "beanie",
            "items": [
                {"id": 12, "name": "Black Beanie", "price": 7, "thumbnail_url": "https://tr.fake/12.png",
                 "roblox_url": "https://www.roblox.com/catalog/12/"},
                {"id": 11, "name": "Grey Beanie", "price": 5, "thumbnail_url": "https://tr.fake/11.png",
                 "roblox_url": "https://www.roblox.com/catalog/11/"},
            ],
        },
        {
            "slot": "Animation",
            "keyword": "zombie",
            "items": [
                {"id": 21, "name": "Zombie Pack", "price": 250, "thumbnail_url": "https://tr.fake/21.png",
                 "roblox_url": "https://www.roblox.com/bundles/21/"},
            ],
        },
    ]
    assert catalog.searches[0] == {
        "keyword": "beanie", "category": "Accessories", "subcategory": "HeadAccessories",
        "max_results": 100, "price_filter": True, "max_price": 300, "free_only": False,
    }
    assert 'Style description: "emo skater"' in fake_llm.calls[0]["messages"][1]["content"]


async def test_without_an_llm_default_slots_get_top_catalog_results(fake_llm, catalog, monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    catalog.results = {"goth": [item(i) for i in range(1, 6)]}

    slots = await stylist.generate_outfit("goth")

    assert [slot["slot"] for slot in slots] == ["Hat", "Hair", "Shirt", "Pants"]
    assert all([entry["id"] for entry in slot["items"]] == [1, 2, 3, 4, 5] for slot in slots)


async def test_a_failed_search_only_empties_its_own_slot(fake_llm, catalog):
    plan = [{"slot": "Hat", "keyword": "beanie"}, {"slot": "Shirt", "keyword": "hoodie"}]
    fake_llm.respond = answer_with(plan, {"Shirt": [1]})
    catalog.results = {"beanie": roblox_client.RobloxAPIError("rate limited", 429), "hoodie": [item(1)]}

    slots = await stylist.generate_outfit("emo")

    assert [(slot["slot"], len(slot["items"])) for slot in slots] == [("Hat", 0), ("Shirt", 1)]


async def test_fails_when_every_search_fails(fake_llm, catalog):
    fake_llm.replies = [json.dumps([{"slot": "Hat", "keyword": "beanie"}])]
    catalog.results = {"beanie": roblox_client.RobloxAPIError("catalog down", 503)}

    with pytest.raises(roblox_client.RobloxAPIError):
        await stylist.generate_outfit("emo")


def answer_vision(plan, picks):
    """A FakeLLM responder: the plan for prompt 1, picks (a dict/list) for the vision call.

    The vision call's user content is a list of parts, not a string.
    """

    def respond(messages):
        user = messages[1]["content"]
        if isinstance(user, str) and user.startswith("Style description"):
            return json.dumps(plan)
        return json.dumps(picks) if not isinstance(picks, str) else picks

    return respond


async def test_create_outfit_picks_one_image_per_slot(fake_llm, catalog):
    plan = [
        {"slot": "Hat", "keyword": "beanie", "category": "Accessories", "subcategory": "Hats"},
        {"slot": "Shirt", "keyword": "hoodie", "category": "Clothing", "subcategory": "Shirt"},
    ]
    fake_llm.respond = answer_vision(plan, {"1": 2, "2": 1})
    catalog.results = {
        "beanie": [item(11, "Grey Beanie", 5), item(12, "Black Beanie", 7), item(13, "Red Beanie", 9)],
        "hoodie": [item(21, "Black Hoodie", 65), item(22, "Grey Hoodie", 60)],
    }

    slots = await stylist.create_outfit("  emo   skater ")

    assert slots == [
        {
            "slot": "Hat",
            "keyword": "beanie",
            "items": [
                {"id": 12, "name": "Black Beanie", "price": 7, "thumbnail_url": "https://tr.fake/12.png",
                 "roblox_url": "https://www.roblox.com/catalog/12/"},
            ],
        },
        {
            "slot": "Shirt",
            "keyword": "hoodie",
            "items": [
                {"id": 21, "name": "Black Hoodie", "price": 65, "thumbnail_url": "https://tr.fake/21.png",
                 "roblox_url": "https://www.roblox.com/catalog/21/"},
            ],
        },
    ]
    # The last call is the vision call: it targets the vision model and actually sends images.
    vision_call = fake_llm.calls[-1]
    assert vision_call["model"] == llm.VISION_MODEL
    content = vision_call["messages"][1]["content"]
    image_urls = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
    assert image_urls == [
        "https://tr.fake/11.png", "https://tr.fake/12.png", "https://tr.fake/13.png",
        "https://tr.fake/21.png", "https://tr.fake/22.png",
    ]


async def test_create_outfit_falls_back_to_top_candidate_when_vision_fails(fake_llm, catalog):
    plan = [{"slot": "Hat", "keyword": "beanie"}, {"slot": "Shirt", "keyword": "hoodie"}]
    fake_llm.respond = answer_vision(plan, "the black one looks best!")  # never JSON -> LLMError
    catalog.results = {"beanie": [item(11), item(12)], "hoodie": [item(21), item(22)]}

    slots = await stylist.create_outfit("emo")

    assert [(slot["slot"], [entry["id"] for entry in slot["items"]]) for slot in slots] == [
        ("Hat", [11]),
        ("Shirt", [21]),
    ]


async def test_create_outfit_caps_total_images(fake_llm, catalog, monkeypatch):
    monkeypatch.setattr(stylist, "VISION_MAX_IMAGES", 4)
    plan = [{"slot": "Hat", "keyword": "beanie"}, {"slot": "Shirt", "keyword": "hoodie"}]
    fake_llm.respond = answer_vision(plan, {"1": 1, "2": 1})
    catalog.results = {
        "beanie": [item(i) for i in range(11, 15)],  # 4 candidates
        "hoodie": [item(i) for i in range(21, 25)],  # 4 candidates
    }

    slots = await stylist.create_outfit("emo")

    content = fake_llm.calls[-1]["messages"][1]["content"]
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 4  # Hat takes 3, Shirt takes the 1 of the budget that's left
    assert [slot["slot"] for slot in slots] == ["Hat", "Shirt"]
