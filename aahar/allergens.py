"""Allergen ontology: the 14 major allergens used by the original app, with synonyms.

Used for (1) query expansion in RAG, (2) tagging chunks, (3) matching scanned products.
"""
import re
from typing import Optional

MAJOR_ALLERGENS = [
    "Peanuts", "Tree nuts", "Milk", "Eggs", "Soya", "Wheat", "Sesame",
    "Celery", "Mustard", "Fish", "Crustaceans", "Molluscs", "Lupin", "Sulphites",
]

SYNONYMS = {
    "Peanuts": ["peanut", "peanuts", "groundnut", "groundnuts", "arachis", "satay", "monkey nut"],
    "Tree nuts": ["tree nut", "tree nuts", "almond", "walnut", "hazelnut", "cashew", "pistachio",
                  "pecan", "brazil nut", "macadamia", "nut", "nuts", "marzipan", "praline", "pesto"],
    "Milk": ["milk", "dairy", "lactose", "casein", "whey", "cream", "butter", "cheese", "yoghurt",
             "yogurt", "ghee", "custard"],
    "Eggs": ["egg", "eggs", "mayonnaise", "mayo", "albumin", "meringue", "ovalbumin"],
    "Soya": ["soya", "soy", "soybean", "soybeans", "tofu", "edamame", "miso", "tempeh", "soy sauce"],
    "Wheat": ["wheat", "gluten", "flour", "bread", "pasta", "couscous", "semolina", "spelt",
              "breadcrumb", "breadcrumbs", "batter", "barley", "rye", "tempura", "croutons"],
    "Sesame": ["sesame", "tahini", "hummus", "houmous", "halva"],
    "Celery": ["celery", "celeriac"],
    "Mustard": ["mustard", "dijon"],
    "Fish": ["fish", "salmon", "cod", "tuna", "anchovy", "anchovies", "worcestershire", "fish sauce"],
    "Crustaceans": ["crustacean", "crustaceans", "prawn", "prawns", "shrimp", "crab", "lobster",
                    "crayfish", "langoustine", "shellfish"],
    "Molluscs": ["mollusc", "molluscs", "mussel", "mussels", "oyster", "oysters", "squid", "calamari",
                 "clam", "clams", "scallop", "snail", "whelk"],
    "Lupin": ["lupin", "lupine"],
    "Sulphites": ["sulphite", "sulphites", "sulfite", "sulfites", "sulphur dioxide", "e220", "wine"],
}

# OpenFoodFacts / menu tag -> canonical
_TAG_ALIASES = {
    "peanuts": "Peanuts", "nuts": "Tree nuts", "tree nuts": "Tree nuts", "milk": "Milk",
    "eggs": "Eggs", "egg": "Eggs", "soybeans": "Soya", "soya": "Soya", "soy": "Soya",
    "gluten": "Wheat", "wheat": "Wheat", "sesame-seeds": "Sesame", "sesame": "Sesame",
    "celery": "Celery", "mustard": "Mustard", "fish": "Fish", "crustaceans": "Crustaceans",
    "molluscs": "Molluscs", "lupin": "Lupin", "sulphur-dioxide-and-sulphites": "Sulphites",
    "sulphites": "Sulphites",
}


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", re.I)


_PATTERNS = {a: [_pattern(t) for t in terms] for a, terms in SYNONYMS.items()}


def detect(text: str) -> set[str]:
    """Return canonical allergens mentioned (directly or via synonym) in text."""
    found = set()
    for allergen, pats in _PATTERNS.items():
        if any(p.search(text) for p in pats):
            found.add(allergen)
    return found


def canon_tag(tag: str) -> Optional[str]:
    """Normalise OpenFoodFacts-style tags ('en:milk') or menu words to canonical names."""
    t = tag.lower().replace("en:", "").replace("_", " ").strip()
    if t in _TAG_ALIASES:
        return _TAG_ALIASES[t]
    for a in MAJOR_ALLERGENS:
        if t == a.lower():
            return a
    d = detect(t)
    return next(iter(d)) if len(d) == 1 else None


def expand_terms(allergens: set[str]) -> list[str]:
    out = []
    for a in allergens:
        out.extend(SYNONYMS.get(a, [a.lower()])[:6])
    return out
