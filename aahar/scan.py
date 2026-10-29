"""Barcode -> allergen verdict using Open Food Facts, matched against the user's profile."""
import httpx
from .allergens import canon_tag, detect

OFF = ("https://world.openfoodfacts.org/api/v2/product/{}.json?fields=product_name,brands,"
       "allergens_tags,traces_tags,ingredients_text,image_front_small_url")


def lookup(barcode: str, user_allergens: list[str]) -> dict:
    barcode = "".join(ch for ch in barcode if ch.isdigit())
    if not 6 <= len(barcode) <= 14:
        return {"verdict": "unknown", "message": "That doesn't look like a valid barcode.", "barcode": barcode}
    try:
        data = httpx.get(OFF.format(barcode), timeout=10, headers={"User-Agent": "Aahar.AI/1.0"}).json()
    except Exception:
        return {"verdict": "unknown", "barcode": barcode,
                "message": "Couldn't reach the product database. Check your connection."}
    if data.get("status") != 1:
        return {"verdict": "not_found", "barcode": barcode, "message": "Product not found."}
    return judge(data["product"], user_allergens, barcode)


def judge(p: dict, user_allergens: list[str], barcode: str = "") -> dict:
    contains = {a for t in p.get("allergens_tags", []) if (a := canon_tag(t))}
    traces = {a for t in p.get("traces_tags", []) if (a := canon_tag(t))}
    ing = p.get("ingredients_text") or ""
    contains |= detect(ing)
    mine = set(user_allergens)
    hit, trace_hit = sorted(contains & mine), sorted((traces & mine) - contains)
    if hit:
        verdict, msg = "risky", f"Contains {', '.join(hit)}. Do not eat."
    elif trace_hit:
        verdict, msg = "risky", f"May contain traces of {', '.join(trace_hit)}. Check with your allergy plan."
    elif not ing and not p.get("allergens_tags"):
        verdict, msg = "unknown", "No ingredient data available. Don't eat it unless you can read the label."
    else:
        verdict, msg = "safe", "No match with your allergens in the listed data. Always double-check the label."
    return {"verdict": verdict, "message": msg, "barcode": barcode,
            "name": p.get("product_name") or "Unknown Product", "brand": p.get("brands", ""),
            "image": p.get("image_front_small_url"), "contains": sorted(contains),
            "traces": sorted(traces), "matched": hit + trace_hit}
