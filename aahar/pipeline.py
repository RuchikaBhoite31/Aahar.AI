"""End-to-end RAG pipeline: safety -> retrieve -> confidence gate -> generate -> cite."""
from __future__ import annotations
from functools import lru_cache
from typing import Optional

from .allergens import detect
from .config import settings
from .generation import extractive_answer, generate
from .kb import load_chunks
from .retrieval import HybridRetriever
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


def ask(question: str, user_allergens: Optional[list[str]] = None, history=None, k: Optional[int] = None) -> dict:
    question = question.strip()
    allergens = set(user_allergens or [])
    banner = emergency_banner(question)
    care = wellbeing_message(question)
    if care:
        return {"answer": care, "citations": [], "confidence": 1.0, "mode": "safety", "emergency": False}

    search_q = _contextualise(question, history)
    r = retriever()
    kinds = None
    if banner:   # live reaction: ground the answer in the anaphylaxis / adrenaline guidance, not menus or dialogue
        search_q = "What to do in anaphylaxis? use adrenaline pen then call ambulance. " + search_q
        kinds = {"behaviour", "education"}
    hits = r.search(search_q, k=k or settings.top_k, user_allergens=allergens, kinds=kinds)
    conf = r.confidence(search_q, hits)

    if conf < settings.min_confidence:
        answer, mode, hits = ABSTAIN, "abstain", []
    else:
        answer, mode = generate(question, hits, allergens, history, weight=r.weight)
        if conf < 0.6 and mode == "extractive":
            answer = "I'm not fully certain, but this is the closest information I found:\n\n" + answer

    # personalised allergen heads-up (deterministic, never LLM-generated)
    mentioned = detect(question)
    if mentioned & allergens:
        answer += f"\n\nHeads-up: **{', '.join(sorted(mentioned & allergens))}** is on your allergen list."
    if banner:
        answer = banner + "\n\n" + answer

    cites = []
    for i, h in enumerate(hits, 1):
        if f"[{i}]" in answer or mode == "llm":
            cites.append({"n": i, "title": h.chunk.title, "source": h.chunk.source, "kind": h.chunk.kind,
                          "snippet": h.chunk.text[:260] + ("…" if len(h.chunk.text) > 260 else "")})
    return {"answer": answer, "citations": cites, "confidence": conf, "mode": mode,
            "emergency": bool(banner), "disclaimer": DISCLAIMER}
