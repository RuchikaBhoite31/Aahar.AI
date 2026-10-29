"""Answer generation: grounded LLM (OpenAI / Anthropic) with an offline extractive fallback."""
from __future__ import annotations
import re
from typing import Optional

import httpx

from .config import settings
from .kb import _split_sentences
from .retrieval import Hit, tokenize

SYSTEM = """You are Aahar.AI, a warm, clear assistant helping teenagers manage food allergies safely.
RULES:
1. Answer ONLY from the numbered CONTEXT passages. If they do not contain the answer, say you don't have that in your sources and suggest asking their allergy team. Never invent medical facts, doses, or product/menu details.
2. Cite every factual sentence with its passage number like [1] or [2][3]. Never cite a number that isn't in CONTEXT.
3. The user's allergens are given. When relevant, tailor the answer to them, but do not claim any food is safe for them unless the context says so explicitly.
4. Be concise (max ~140 words), friendly, plain language for a 13-17 year old. Use short paragraphs or a short list.
5. Never tell the user to skip, delay or replace their adrenaline pen or emergency services."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def _user_prompt(question, hits, allergens, history):
    hist = "\n".join(f"{r.upper()}: {t}" for r, t in (history or [])[-4:])
    history_section = f"RECENT CHAT:\n{hist}\n" if hist else ""
    return (f"USER ALLERGENS: {', '.join(sorted(allergens)) or 'not specified'}\n"
            f"{history_section}"
            f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}")


def _call_openai(system: str, user: str) -> str:
    r = httpx.post(f"{settings.openai_base_url}/chat/completions", timeout=45,
                   headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                   json={"model": settings.openai_model, "temperature": 0.1, "max_tokens": 450,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_anthropic(system: str, user: str) -> str:
    r = httpx.post("https://api.anthropic.com/v1/messages", timeout=45,
                   headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"},
                   json={"model": settings.anthropic_model, "max_tokens": 450, "temperature": 0.1,
                         "system": system, "messages": [{"role": "user", "content": user}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()


def llm_complete(system: str, user: str) -> Optional[str]:
    try:
        if settings.provider == "openai":
            return _call_openai(system, user)
        if settings.provider == "anthropic":
            return _call_anthropic(system, user)
    except Exception as e:
        print(f"[llm] {e}")
    return None


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3, weight=None) -> str:
    """Offline fallback: choose the most question-relevant sentences verbatim from the top sources.
    Sentences are scored by IDF-weighted query-term coverage, so rare, specific words
    (e.g. 'adrenaline', 'lupin') matter more than generic ones."""
    if hits and hits[0].chunk.kind == "menu" and hits[0].coverage >= 0.5:
        # structured menu record: the whole record *is* the answer (name, allergens, hidden allergens, safe-or-not)
        return "\n".join(f"• {x.strip()} [1]" for x in _split_sentences(hits[0].chunk.text) if x.strip())
    q = set(tokenize(question))
    w = weight or (lambda t: 1.0)
    denom = sum(w(t) for t in q) or 1.0

    def sentences(chunk):
        out = []
        for sent in _split_sentences(chunk.text):            # break very long ';' lists into readable pieces
            out += [x.strip() for x in sent.split("; ")] if len(sent.split()) > 40 else [sent.strip()]
        return [x if x.endswith((".", "!", "?")) else x + "." for x in out if x]

    scored = []                                              # (chunk_score, rank, sentences, best_idx)
    for n, h in enumerate(hits[:3], 1):
        sents = sentences(h.chunk)
        title_toks = set(tokenize(h.chunk.title))
        ctx = 0.5 * sum(w(t) for t in q if t in title_toks) / denom   # heading match = context credit
        best_i, best = 0, -1.0
        for i, sent in enumerate(sents):
            toks = set(tokenize(sent))
            sc = sum(w(t) for t in q if t in toks) / denom - 0.01 * i
            if sc > best:
                best_i, best = i, sc
        scored.append((best + ctx + 0.3 / n, n, sents, best_i))
    if not scored:
        return ""
    scored.sort(reverse=True)
    lines, words = [], 0
    _, n, sents, i = scored[0]
    start = 0 if i < 5 else i - 1                            # lead with the passage opening unless the match is deep inside a long chunk
    for sent in sents[start:]:
        if words >= 85 or len(lines) >= 4:
            break
        lines.append(f"• {sent} [{n}]"); words += len(sent.split())
    if len(scored) > 1 and scored[1][0] >= 0.75 * scored[0][0] and scored[1][2][0:1]:
        _, n2, sents2, i2 = scored[1]
        if hits[n2 - 1].chunk.title != hits[n - 1].chunk.title:
            lines.append(f"• {sents2[i2]} [{n2}]")
    return "\n".join(lines)


def validate_citations(text: str, n_sources: int) -> tuple[str, bool]:
    """Strip invalid citation numbers. Returns (text, has_valid_citation)."""
    ok = {"v": False}

    def fix(m):
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            ok["v"] = True
            return m.group(0)
        return ""
    return re.sub(r"\[(\d+)\]", fix, text), ok["v"]


def generate(question: str, hits: list[Hit], allergens: set[str], history=None, weight=None) -> tuple[str, str]:
    """Returns (answer, mode) where mode is 'llm' or 'extractive'."""
    if settings.provider != "none":
        out = llm_complete(SYSTEM, _user_prompt(question, hits, allergens, history))
        if out:
            out, cited = validate_citations(out, len(hits))
            if cited:                       # ungrounded (uncited) LLM output is rejected
                return out, "llm"
    return extractive_answer(question, hits, weight=weight), "extractive"
