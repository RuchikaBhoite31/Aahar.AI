"""End-to-end RAG pipeline: safety -> retrieve -> confidence gate -> generate -> cite."""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Optional

from .allergens import detect
from .config import settings
from .generation import extractive_answer, generate
from .kb import load_chunks
from .retrieval import HybridRetriever, tokenize
from .safety import DISCLAIMER, emergency_banner, wellbeing_message

ABSTAIN = ("I couldn't find that in my allergy sources, so I don't want to guess. "
           "Try rephrasing, or ask your allergy team / pharmacist. I'm best at questions about allergens, "
           "reactions, adrenaline pens, eating out, travel, alcohol and handling peer pressure.")


@lru_cache(maxsize=1)
def retriever() -> HybridRetriever:
    return HybridRetriever(load_chunks())


def _contextualise(question: str, history) -> str:
    """Resolve short follow-ups ('what about tree nuts?') using the previous user turn."""
    if history and len(question.split()) <= 5:
        prev = [t for r, t in history if r == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question


def _is_general_restaurant_question(question: str, hits) -> bool:
    """Avoid guessing a specific menu item when the user asks general waiter advice."""
    if not re.search(r"\b(waiter|waitress|server|restaurant|chef|staff)\b", question, re.I):
        return False
    if not re.search(r"\b(ask|tell|say|check|confirm|request|questions?)\b", question, re.I):
        return False
    query_tokens = set(tokenize(question))
    generic = set(tokenize("waiter waitress server restaurant chef staff ask tell say check confirm request question dish food menu allergy allergen"))
    specific_tokens = query_tokens - generic
    return not any(
        hit.chunk.kind == "menu"
        and specific_tokens & set(tokenize(hit.chunk.title.split("›")[-1]))
        for hit in hits
    )


def ask(question: str, user_allergens: Optional[list[str]] = None, history=None, k: Optional[int] = None) -> dict:
    question = question.strip()
    allergens = set(user_allergens or [])
    banner = emergency_banner(question)
    care = wellbeing_message(question)
    if care:
        return {"answer": care, "citations": [], "confidence": 1.0, "mode": "safety", "emergency": False}
    if banner:
        # Urgent instructions are fixed and independent of retrieval/model quality.
        return {"answer": banner, "citations": [], "confidence": 1.0, "mode": "safety",
                "emergency": True, "disclaimer": DISCLAIMER}

    search_q = _contextualise(question, history)
    r = retriever()
    kinds = None
    hits = r.search(search_q, k=k or settings.top_k, user_allergens=allergens, kinds=kinds)
    general_restaurant_question = not kinds and _is_general_restaurant_question(question, hits)
    if general_restaurant_question:
        # Do not infer a dish from a generic question; retrieve communication and cross-contact guidance.
        search_q = ("What should I ask restaurant staff about ingredients, sauces, preparation, shared equipment, "
                    "and cross-contact? " + search_q)
        hits = r.search(search_q, k=k or settings.top_k, user_allergens=allergens,
                        kinds={"behaviour", "education"})
    social_pressure_question = bool(re.search(r"\b(friend|friends|peer|party|birthday)\b", question, re.I)
                                    and re.search(r"\b(pressure|pushing|offer|say|handle|respond)\b", question, re.I))
    if social_pressure_question:
        hits = r.search(search_q, k=k or settings.top_k, user_allergens=allergens,
                        kinds={"dialogue", "scenario"})
    conf = r.confidence(search_q, hits)

    if conf < settings.min_confidence:
        answer, mode, hits = ABSTAIN, "abstain", []
    else:
        answer_question = search_q if general_restaurant_question else question
        answer, mode = generate(answer_question, hits, allergens, history, weight=r.weight)
        if conf < 0.6 and mode == "extractive":
            answer = "I'm not fully certain, but this is the closest information I found:\n\n" + answer

    # personalised allergen heads-up (deterministic, never LLM-generated)
    mentioned = detect(question)
    if mentioned & allergens:
        answer += f"\n\nHeads-up: **{', '.join(sorted(mentioned & allergens))}** is on your allergen list."
    cites = []
    for i, h in enumerate(hits, 1):
        if f"[{i}]" in answer or mode == "llm":
            cites.append({"n": i, "title": h.chunk.title, "source": h.chunk.source, "kind": h.chunk.kind,
                          "snippet": h.chunk.text[:260] + ("…" if len(h.chunk.text) > 260 else "")})
    return {"answer": answer, "citations": cites, "confidence": conf, "mode": mode,
            "emergency": False, "disclaimer": DISCLAIMER}
