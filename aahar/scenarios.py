"""Practice scenarios. Waiter answers come from retrieved menu items (RAG over the menu);
each user turn also gets a coaching tip retrieved from the behaviour/education modules."""
import json, random, re
from pathlib import Path
from .allergens import detect
from .config import DATA
from .pipeline import retriever

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


def start(sid):
    sc = _scenario(sid)
    group = sid in SCRIPT
    return {"speaker": "Friend" if group else "Waiter",
            "text": _friends(sid)[0] if group else sc.get("initialDialogue", "Hi there! Welcome in."),
            "context": sc.get("scenarioContext") or sc.get("description", ""),
            "background": "/assets/images/backgrounds/" + (IMG.get(sid) or "restaurant_interior.png")}


def reply(sid, msg, allergens, turn):
    sc = _scenario(sid)
    said = detect(msg)
    disclosed = bool(re.search(r"allerg|intoleran|can'?t (eat|have)|cannot (eat|have)", msg, re.I)) or bool(said)
    if sid in SCRIPT:
        friends = _friends(sid)
        text = friends[min(turn + 1, len(friends) - 1)]
        who, maxt = "Friend", 6
    else:
        lvl = sid.split("_")[-1]
        hits = retriever().search(msg, k=8, user_allergens=set(allergens), kinds={"menu"})
        hits = [h for h in hits if f"({lvl})" in h.chunk.title]
        dish = hits[0].chunk if hits and hits[0].coverage > 0.3 else None
        if dish and re.search(r"contain|have|in it|safe|allerg|gluten|nut|milk|egg|sauce|ingredient", msg, re.I):
            text = "Let me check that for you. " + dish.text.split("Good questions")[0].strip()
        elif not disclosed:
            text = random.choice(sc.get("gentlePrompts", ["Any allergies I should know about?"]))
        else:
            text = random.choice(sc.get("positiveReinforcements", ["Thanks for letting me know - I'll take care of it."]))
        who, maxt = "Waiter", sc.get("maxTurns", 8)
    if disclosed:
        tip = retriever().search(msg + " allergy", k=1, kinds={"behaviour", "education"})
        coach = "Nice - you spoke up about your allergy." + (" " + tip[0].chunk.text.split(". ")[0].strip() + "." if tip else "")
    else:
        coach = "Tip: say clearly that you have an allergy and which food(s), before ordering or eating."
    return {"speaker": who, "text": text, "coach": coach, "points": 10 if disclosed else 0, "done": turn + 1 >= maxt}
