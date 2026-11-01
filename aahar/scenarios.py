"""Responsive allergy-safety practice scenarios grounded in the training menu and scripts."""
import json, re
from pathlib import Path
from .allergens import detect
from .config import DATA
from .pipeline import retriever
from .retrieval import tokenize

MAN = json.loads((DATA / "scenarios" / "scenario_manifest.json").read_text())["scenarios"]
IMG = {"party_intermediate": "birthday_group.png", "dinner_with_friends": "dinner_with_friends.png"}
SCRIPT = {"party_intermediate": "birthday_party_dialogue.json", "dinner_with_friends": "dinner_friends_dialogue.json"}


def cards():
    out = []
    for sid, m in MAN.items():
        if m.get("enabled", True):
            out.append({"id": sid, "title": m["name"], "description": m["description"], "type": m["type"],
                        "level": m["level"], "required": m.get("requiredScore", 0),
                        "image": "/assets/images/backgrounds/" + IMG.get(sid, "restaurant_dining_card_image.jpeg")})
    if "dinner_with_friends" not in MAN:
        out.append({"id": "dinner_with_friends", "title": "Dinner with Friends", "type": "party", "level": "group",
                    "description": "Handle group ordering situation when friends want to share unsafe food",
                    "required": 0, "image": "/assets/images/backgrounds/dinner_with_friends.png"})
    return out


def _scenario(sid):
    f = MAN.get(sid, {}).get("file")
    return json.loads((DATA / "scenarios" / Path(f).name).read_text()) if f else {}


def _friends(sid):
    return [l["line"] for l in json.loads((DATA / "knowledge" / SCRIPT[sid]).read_text()) if l["speaker"] != "user"]


def _menu_items(sid=None):
    chunks = [c for c in retriever().chunks if c.kind == "menu"]
    if sid and sid.startswith("restaurant_"):
        level = sid.rsplit("_", 1)[-1]
        return [c for c in chunks if f"({level})" in c.title]
    return chunks


def _named_menu_item(text, chunks):
    """Match a dish by its title tokens; never trust a loose semantic match for menu facts."""
    query = set(tokenize(text))
    matches = []
    for chunk in chunks:
        title_tokens = set(tokenize(chunk.title.split("›")[-1]))
        overlap = query & title_tokens
        if len(overlap) >= 2:
            ratio = len(overlap) / max(1, len(title_tokens))
            if ratio >= 0.6:
                matches.append((ratio, len(overlap), chunk))
    return max(matches, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def _is_menu_question(message):
    return bool(re.search(
        r"ingredient|contain|have in|what is in|what's in|allergen|safe|cross.?contact|cross.?contamin|"
        r"prepare|prepar|sauce|suitable|gluten|peanut|nut|milk|dairy|egg|prawn|shrimp|fish|sesame|"
        r"celery|mustard|soya|soy|lupin|sulphite|sulfite|menu|dish|food", message, re.I
    ))


def _is_allergy_disclosure(message):
    return bool(re.search(
        r"\b(allergic to|allergy to|have (?:a |an )?(?:[a-z-]+ )?allerg(?:y|ies)|"
        r"intoleran\w* to|can't eat|cannot eat|can't have|cannot have)\b",
        message, re.I
    ))


def _menu_reply(item):
    details = item.text.split("Good questions to ask:")[0].strip()
    questions = item.text.partition("Good questions to ask:")[2].strip()
    answer = "Let me check that dish. " + details
    if questions:
        answer += " Good questions to confirm with the kitchen: " + questions
    return answer + " Please confirm ingredients and cross-contact with the restaurant; this training menu cannot guarantee safety."


def _restaurant_reply(sid, msg, history, allergens):
    menu = _menu_items(sid)
    item = _named_menu_item(msg, menu)
    if item is None:
        for earlier in reversed((history or [])[-6:]):
            item = _named_menu_item(earlier, menu)
            if item:
                break
    if item and _is_menu_question(msg):
        return _menu_reply(item)

    other_level_item = _named_menu_item(msg, _menu_items())
    if other_level_item is None:
        for earlier in reversed((history or [])[-6:]):
            other_level_item = _named_menu_item(earlier, _menu_items())
            if other_level_item:
                break
    if other_level_item and _is_menu_question(msg):
        name = other_level_item.title.split("›")[-1].strip()
        return (f"I don't see {name} on this level's training menu, so I can't verify its ingredients here. "
                "Please ask about a listed dish or check directly with restaurant staff.")

    disclosure = _is_allergy_disclosure(msg)
    asks_for_options = bool(re.search(r"which|what|any|option|dish|menu|contain|ingredient|safe|order", msg, re.I))
    if disclosure and not asks_for_options:
        return ("Thanks for telling me. I'll flag that for the kitchen. Which dish are you thinking about "
                "so I can check its ingredients and preparation?")

    if re.search(r"cross.?contact|cross.?contamin|shared (?:fryer|oil|equipment|utensil)|separate(?:ly)? prepared", msg, re.I):
        return ("I'll check with the kitchen whether they use clean, separate equipment and avoid shared cooking oil. "
                "Which dish are you considering? Please confirm the preparation details with staff before ordering.")

    if _is_menu_question(msg):
        mentioned = detect(msg)
        if mentioned:
            relevant = [c for c in menu if c.allergens & mentioned][:3]
            if relevant:
                names = ", ".join(c.title.split("›")[-1].strip() for c in relevant)
                return (f"The training menu lists {', '.join(sorted(mentioned))} for: {names}. "
                        "That list is not a safety guarantee; ask staff about every ingredient and cross-contact. "
                        "Which dish would you like me to check?")
        return "Which dish would you like me to check? Tell me its name and the ingredient or allergen you want me to ask about."

    if disclosure:
        prefix = "Thanks for telling me. I'll flag that for the kitchen. "
    else:
        prefix = ""

    prompts = _scenario(sid).get("gentlePrompts", [])
    prompt = prompts[0] if prompts else "What would you like to order? Let me know if you have any allergies I should tell the kitchen about."
    return prefix + prompt


def _group_reply(sid, msg, turn, history):
    if _is_menu_question(msg):
        item = _named_menu_item(msg, _menu_items())
        if item is None:
            for earlier in reversed((history or [])[-6:]):
                item = _named_menu_item(earlier, _menu_items())
                if item:
                    break
        if item:
            return "Let's check the training menu instead of guessing. " + _menu_reply(item)
        return ("I'm not sure what's in that food from this scenario. Let's check the label or ask the host/restaurant "
                "before anyone eats it, especially about ingredients and cross-contact.")
    friends = _friends(sid)
    return friends[min(turn + 1, len(friends) - 1)]


def start(sid):
    sc = _scenario(sid)
    group = sid in SCRIPT
    return {"speaker": "Friend" if group else "Waiter",
            "text": _friends(sid)[0] if group else sc.get("initialDialogue", "Hi there! Welcome in."),
            "context": sc.get("scenarioContext") or sc.get("description", ""),
            "background": "/assets/images/backgrounds/" + (IMG.get(sid) or "restaurant_interior.png")}


def reply(sid, msg, allergens, turn, history=None):
    sc = _scenario(sid)
    disclosed = _is_allergy_disclosure(msg)
    if sid in SCRIPT:
        text = _group_reply(sid, msg, turn, history)
        who, maxt = "Friend", 6
    else:
        text = _restaurant_reply(sid, msg, history, allergens)
        who, maxt = "Waiter", sc.get("maxTurns", 8)
    if disclosed:
        coach = "Nice - you clearly disclosed your allergy. Next, ask about the ingredients and cross-contact for a specific dish."
    elif _is_menu_question(msg):
        coach = "Good safety question. Also clearly tell the waiter or host which allergy you need them to flag."
    else:
        coach = "Tip: say clearly that you have an allergy and which food(s), before ordering or eating."
    return {"speaker": who, "text": text, "coach": coach, "points": 10 if disclosed else 0, "done": turn + 1 >= maxt}
