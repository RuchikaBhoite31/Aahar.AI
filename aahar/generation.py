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
    """Return concise, verbatim evidence instead of dumping the start of a passage.

    This deterministic path is used when no LLM key is configured. It ranks individual
    source sentences against the question and retains their source citations.
    """
    if hits and hits[0].chunk.kind == "menu" and hits[0].coverage >= 0.5:
        # structured menu record: the whole record *is* the answer (name, allergens, hidden allergens, safe-or-not)
        return "\n".join(f"• {x.strip()} [1]" for x in _split_sentences(hits[0].chunk.text) if x.strip())

    core = set(tokenize(question))
    w = weight or (lambda t: 1.0)
    term_weights = {t: w(t) for t in core}
    if re.search(r"\b(signs|symptoms)\b", question, re.I) and re.search(r"\banaphylaxis\b", question, re.I):
        for term in tokenize("swelling tongue throat breathing wheeze cough blood pressure dizzy faint collapse confused drowsy"):
            if term not in term_weights:
                term_weights[term] = 0.65 * w(term)
    if re.search(r"\b(waiter|waitress|server|restaurant|chef|staff)\b", question, re.I):
        for term in tokenize("restaurant ingredients sauces preparation cross contact equipment separately"):
            if term not in term_weights:
                term_weights[term] = 0.2 * w(term)
    denom = sum(term_weights.values()) or 1.0
    asks_definition = bool(re.search(r"\b(what is|define|meaning of|what does .* mean)\b", question, re.I))
    asks_action = bool(re.search(r"\b(what should i do|what do i do|how do i|what to do)\b", question, re.I))

    def sentences(chunk):
        if chunk.kind == "dialogue":
            quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", chunk.text)
            if quoted:
                return [f"“{quote.strip()}”" for quote in quoted if 6 <= len(quote.split()) <= 55]
        out = []
        for sent in _split_sentences(chunk.text):
            parts = sent.split("; ") if len(sent.split()) > 35 else [sent]
            for part in parts:
                part = part.strip().strip("•- ")
                if part and not part.endswith((".", "!", "?")):
                    part += "."
                if 1 <= len(part.split()) <= 55 and not re.fullmatch(r"[\W\d_]+", part):
                    out.append(part)
        return out

    candidates = []  # (score, source number, sentence, normalized tokens)
    for n, hit in enumerate(hits[:5], 1):
        title_tokens = set(tokenize(hit.chunk.title))
        source_bonus = 0.01 / n
        for sent in sentences(hit.chunk):
            tokens = set(tokenize(sent))
            matched = sum(value for term, value in term_weights.items() if term in tokens)
            coverage = matched / denom
            if coverage <= 0:
                continue
            score = coverage + source_bonus
            definition_sentence = bool(re.search(r"\b(is|means|refers to)\s+(a|an|the)\b", sent, re.I))
            if asks_definition and definition_sentence:
                score += 1.0
            if asks_action and re.search(r"\b(should|use|ask|call|tell|check|make sure|lie down)\b", sent, re.I):
                score += 0.15
            if core & title_tokens:
                score += 0.03
            candidates.append((score, n, sent, tokens))

    if not candidates:
        return ""
    if asks_definition:
        definitions = [candidate for candidate in candidates
                       if re.search(r"\b(is|means|refers to)\s+(a|an|the)\b", candidate[2], re.I)]
        if definitions:
            candidates = definitions
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    selected, selected_tokens, title_counts = [], [], {}
    for score, n, sent, tokens in candidates:
        title = hits[n - 1].chunk.title
        if score < max(0.16, best_score * 0.42) or title_counts.get(title, 0) >= 2:
            continue
        if any(len(tokens & prior) / max(1, len(tokens | prior)) > 0.72 for prior in selected_tokens):
            continue
        selected.append(f"• {sent} [{n}]")
        selected_tokens.append(tokens)
        title_counts[title] = title_counts.get(title, 0) + 1
        if len(selected) >= max_sentences:
            break
    return "\n\n".join(selected)


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
