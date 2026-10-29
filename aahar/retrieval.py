"""Hybrid retriever.

Signals fused with Reciprocal Rank Fusion (RRF):
  1. BM25 over stemmed word tokens          (exact / keyword precision)
  2. Char n-gram TF-IDF cosine              (typos, plurals, "anaphylactic" ~ "anaphylaxis")
  3. Optional dense embeddings              (semantic recall; sentence-transformers)
Then: allergen-aware boost, per-document diversity cap, and a calibrated confidence
score used by the pipeline to abstain instead of hallucinating.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .allergens import detect, expand_terms
from .config import settings
from .kb import Chunk

STOP = set("""a an the and or of to in on at for with without is are was were be been am do does did can could
should would will may might i me my we you your it its this that these those what which who how why when where
about as by from into if so than then there they them their not no yes please tell us any some more most""".split())

# Light query-intent expansion (teen phrasing -> corpus vocabulary)
INTENT = {
    "epipen": "adrenaline pen auto-injector", "epi": "adrenaline", "anaphylactic": "anaphylaxis",
    "allergic reaction": "anaphylaxis symptoms adrenaline antihistamine",
    "reaction": "anaphylaxis symptoms", "jab": "adrenaline pen injection",
    "kissing": "kiss kissed", "kissed": "kiss kissing",
    "drinking": "alcohol", "pub": "alcohol", "holiday": "travel", "flight": "travel airline",
    "plane": "travel airline", "restaurant": "restaurant waiter ask ingredients",
    "friends": "peer pressure friends", "peer pressure": "friends pressure assert",
    "inhaler": "asthma", "wheeze": "asthma", "outgrow": "tolerance introduce challenge",
    "test": "tests skin prick blood challenge", "tests": "skin prick blood challenge",
    "nasal": "neffy nasal spray adrenaline", "spray": "neffy nasal spray",
}


# Domain vocabulary: a question must touch allergy / medicine / eating-out / social-safety topics
# (or name an allergen) to be answerable. Everything else is refused rather than force-matched.
ANCHOR = re.compile(
    r"allerg|anaphyla|adrenalin|epinephrin|epi-?pen|\bepi\b|jext|anapen|emerade|neffy|auto-?injector|antihistamin|histamine|"
    r"asthma|inhaler|intoleran|immune|\bige\b|react(ion|s|ed)?\b|symptom|hives|swell|wheez|itch|rash|"
    r"cross-?(contact|contam)|contaminat|label|ingredient|may contain|traces?\b|gluten|coeliac|celiac|lactose|"
    r"restaurant|waiter|waitress|menu|takeaway|take-away|eat(ing)? out|canteen|school|teacher|"
    r"airline|flight|\bfly(ing)?\b|plane|airport|travel|holiday|alcohol|\bdrink|\bpub\b|beer|wine|"
    r"kiss|friend|peer|pressure|party|birthday|date\b|boyfriend|girlfriend|"
    r"medication|medicine|action plan|first aid|ambulance|\ba&e\b|doctor|allergist|dietitian|skin prick|blood test|food challenge|"
    r"co-?factor|tolerance|outgrow|top ?14|food (allerg|safety)|scan|barcode|nut\b|nuts\b|shellfish|vegan", re.I)


W_TITLE, W_CHAR = 0.35, 0.8   # fusion weights (tuned on tests/gold.py)


def _stem(w: str) -> str:
    """Small suffix stripper: intolerances/intolerance -> intolerance, allergies -> allergy,
    reactions -> reaction; leaves -is/-us/-ss words (anaphylaxis, plus) intact."""
    if len(w) > 4:
        if w.endswith("ies"):
            w = w[:-3] + "y"
        elif w.endswith(("sses", "ches", "shes", "xes", "zes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _norm(text: str) -> str:
    """Lowercase, strip punctuation: 'anaphylaxis?' and 'Anaphylaxis' must index/query identically."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def expand_query(q: str, allergens: Optional[set[str]] = None) -> str:
    ql = q.lower()
    extra = [v for k, v in INTENT.items() if re.search(r"(?<![a-z])" + re.escape(k), ql)]
    mentioned = detect(q)
    extra += expand_terms(mentioned)[:6]
    return q + " " + " ".join(extra)


class BM25:
    def __init__(self, docs: list[list[str]], k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(q):
            idf = self.idf.get(t)
            if not idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    coverage: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.toks = [tokenize(c.index_text) for c in chunks]
        self.bm25 = BM25(self.toks)
        self.tbm25 = BM25([tokenize(c.title.split('›', 1)[-1]) for c in chunks], k1=1.2, b=0.4)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.mat = self.vec.fit_transform([_norm(c.index_text) for c in chunks])
        self.name_toks = [set(tokenize(c.title.split("›")[-1])) if c.kind == "menu" else set() for c in chunks]
        self.df = Counter(t for d in self.toks for t in set(d))
        self.dense = None
        if settings.dense_model:
            try:  # optional dependency
                from sentence_transformers import SentenceTransformer
                self.dense_model = SentenceTransformer(settings.dense_model)
                self.dense = self.dense_model.encode([c.index_text for c in chunks], normalize_embeddings=True)
            except Exception as e:  # pragma: no cover
                print(f"[retriever] dense disabled: {e}")

    def weight(self, token: str) -> float:
        """IDF weight of a (stemmed) token; unseen tokens get a neutral weight."""
        return self.bm25.idf.get(token, 0.5 * max(self.bm25.idf.values()))

    def _coverage(self, core: set[str], toks: set[str]) -> float:
        """IDF-weighted share of query terms present in the chunk. Terms absent from the
        whole corpus get the maximum IDF, so out-of-domain questions score near zero."""
        if not core:
            return 0.0
        mx = max(self.bm25.idf.values())
        w = {t: (self.bm25.idf.get(t, 0.25 * mx) * (2.0 if ANCHOR.search(t) or detect(t) else 1.0)) for t in core}
        return sum(v for t, v in w.items() if t in toks) / sum(w.values())

    @staticmethod
    def _ranks(scores: np.ndarray) -> dict[int, int]:
        order = np.argsort(-scores)
        return {int(i): r for r, i in enumerate(order) if scores[i] > 0}

    def search(self, query: str, k: int = 5, user_allergens: Optional[set[str]] = None,
               kinds: Optional[set[str]] = None) -> list[Hit]:
        q_exp = expand_query(query, user_allergens)
        q_tok = tokenize(q_exp)
        core = set(tokenize(query))
        core_tok = tokenize(query)
        extra_tok = [t for t in q_tok if t not in set(core_tok)]
        # literal query terms dominate; synonym/intent expansion only nudges
        bm = self.bm25.scores(core_tok) + 0.4 * self.bm25.scores(extra_tok)
        ch = (self.mat @ self.vec.transform([_norm(q_exp)]).T).toarray().ravel()
        tt = self.tbm25.scores(core_tok)
        signals = [(bm, 1.0), (tt, W_TITLE), (ch, W_CHAR)]
        if self.dense is not None:
            dn = self.dense @ self.dense_model.encode([query], normalize_embeddings=True)[0]
            signals.append((dn, 0.8))
        # score-normalised fusion (keeps the *margin* between strong and weak matches, which RRF discards)
        # plus a small RRF term for robustness when one signal is flat.
        fused = np.zeros(len(self.chunks))
        for sc, w in signals:
            mx = float(sc.max())
            if mx <= 0:
                continue
            fused += w * np.clip(sc, 0, None) / mx
            for i, r in self._ranks(sc).items():
                fused[i] += 0.15 * w / (60 + r) * 60
        q_all = detect(query)
        for i, c in enumerate(self.chunks):
            if kinds and c.kind not in kinds:
                fused[i] = 0
                continue
            if q_all and c.allergens & q_all:
                fused[i] *= 1.1            # asked about peanuts -> favour peanut chunks
            if c.kind in ("dialogue", "scenario") and not re.search(r"\bfriends?\b|pressure|party|practi[cs]e|scenario|\bsay\b|respond|reply", query.lower()):
                fused[i] *= 0.5
            if q_all and c.kind == "menu" and not (c.allergens & q_all) and "menu" not in query.lower():
                fused[i] *= 0.7
            if c.kind == "menu" and not (self.name_toks[i] & core) and not re.search(r"menu|dish|order|starter|main course|dessert|ingredient", query.lower()):
                fused[i] *= 0.5            # keep menu items out of general-knowledge answers
        order = np.argsort(-fused)
        hits, per_src = [], Counter()
        for i in order:
            if fused[i] <= 0 or len(hits) >= k:
                break
            c = self.chunks[i]
            if per_src[c.id.split(":")[0] + c.title.split("›")[-1]] >= 2:
                continue
            per_src[c.id.split(":")[0] + c.title.split("›")[-1]] += 1
            toks = set(self.toks[i])
            cov = self._coverage(core, toks)
            hits.append(Hit(c, float(fused[i]), float(bm[i]), cov))
        return hits


    def in_domain(self, query: str) -> bool:
        """Domain gate: query must mention an allergen or a term that recurs in the corpus."""
        return bool(detect(query) or ANCHOR.search(query))

    def confidence(self, query: str, hits: list[Hit]) -> float:
        """0..1 estimate that the retrieved context can answer the question."""
        if not hits or not self.in_domain(query):
            return 0.0
        cov = max(h.coverage for h in hits[:3])
        bm = 1 - math.exp(-hits[0].bm25 / 8)
        return round(0.65 * cov + 0.35 * bm, 3)
