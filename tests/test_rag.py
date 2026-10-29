import re
import pytest
from aahar import generation
from aahar.allergens import canon_tag, detect
from aahar.kb import load_chunks
from aahar.pipeline import ask, retriever
from aahar.safety import emergency_banner, wellbeing_message
from aahar.scan import judge
from tests.gold import GOLD, IN_SCOPE, OOS


def test_allergen_detection_and_tags():
    assert detect("is there any tahini or prawn in it?") == {"Sesame", "Crustaceans"}
    assert "Peanuts" in detect("peanut butter") and detect("capital of france") == set()
    assert canon_tag("en:milk") == "Milk" and canon_tag("en:sulphur-dioxide-and-sulphites") == "Sulphites"


def test_chunks_are_structured_and_unique():
    cs = load_chunks()
    assert len(cs) > 150 and len({c.id for c in cs}) == len(cs)
    titles = {c.title for c in cs}
    assert any("How does adrenaline work?" in t for t in titles)          # sub-headings preserved
    assert all(len(c.text.split()) <= 200 for c in cs)                     # no giant chunks
    bullets = [c for c in cs if "Peanut" in c.title and c.kind == "education"]
    assert bullets and "•" not in bullets[0].text                          # bullets folded into sentences


def test_retrieval_quality_thresholds():
    r = retriever()
    h1 = h3 = 0
    for q, gold in GOLD:
        hits = r.search(q, k=5)
        ranks = [i for i, h in enumerate(hits, 1) if any(g.lower() in h.chunk.title.lower() for g in gold)]
        h1 += bool(ranks and ranks[0] == 1); h3 += bool(ranks and ranks[0] <= 3)
    assert h3 / len(GOLD) >= 0.95 and h1 / len(GOLD) >= 0.85


@pytest.mark.parametrize("q", OOS)
def test_out_of_scope_is_refused(q):
    a = ask(q)
    assert a["mode"] == "abstain" and a["citations"] == []


@pytest.mark.parametrize("q", IN_SCOPE)
def test_in_scope_is_answered_with_citations(q):
    a = ask(q)
    assert a["mode"] != "abstain" and a["citations"]


def test_punctuation_and_case_do_not_change_top_result():
    r = retriever()
    a = r.search("Can I drink alcohol with a food allergy?", k=1)[0].chunk.title
    b = r.search("can i drink alcohol with a food allergy", k=1)[0].chunk.title
    assert a == b


def test_every_extractive_sentence_comes_from_a_cited_source():
    a = ask("Is coconut a tree nut?", ["Tree nuts"])
    sources = {c["n"]: c for c in a["citations"]}
    body = [l for l in a["answer"].split("\n") if l.startswith("•")]
    assert body
    r = retriever()
    hits = r.search("Is coconut a tree nut?", k=5, user_allergens={"Tree nuts"})
    for line in body:
        m = re.search(r"\[(\d+)\]$", line); sent = line[2:line.rfind("[")].strip().rstrip(".")
        assert m and sent[:40].lower() in hits[int(m.group(1)) - 1].chunk.text.lower()


def test_personalised_headsup_is_deterministic():
    a = ask("what foods have milk in them", ["Milk"])
    assert "Heads-up" in a["answer"] and "Milk" in a["answer"]


def test_emergency_banner_comes_first_and_uses_anaphylaxis_guidance():
    a = ask("my lips are swelling after eating a nut", ["Tree nuts"])
    assert a["emergency"] and a["answer"].startswith("**If you are having a reaction right now")
    assert any("anaphylaxis" in c["title"].lower() for c in a["citations"])
    assert emergency_banner("I can't breathe") and not emergency_banner("what is a peanut")


def test_wellbeing_message_overrides_retrieval():
    a = ask("sometimes I want to die because of my allergies")
    assert a["mode"] == "safety" and "112" in a["answer"] and a["citations"] == []
    assert wellbeing_message("tell me about milk") is None


def test_llm_output_without_valid_citations_is_rejected(monkeypatch):
    monkeypatch.setattr(generation.settings.__class__, "provider", property(lambda s: "openai"))
    hits = retriever().search("what is anaphylaxis", k=3)
    monkeypatch.setattr(generation, "llm_complete", lambda s, u: "Anaphylaxis is totally harmless.")          # uncited
    _, mode = generation.generate("what is anaphylaxis", hits, set())
    assert mode == "extractive"
    monkeypatch.setattr(generation, "llm_complete", lambda s, u: "It is a severe reaction [1]. Fake claim [9].")
    text, mode = generation.generate("what is anaphylaxis", hits, set())
    assert mode == "llm" and "[9]" not in text and "[1]" in text


def test_scan_verdicts():
    risky = judge({"product_name": "Choc", "allergens_tags": ["en:milk", "en:nuts"], "ingredients_text": "milk, hazelnut"}, ["Milk"])
    trace = judge({"product_name": "Bar", "allergens_tags": [], "traces_tags": ["en:peanuts"], "ingredients_text": "oats"}, ["Peanuts"])
    safe = judge({"product_name": "Rice", "allergens_tags": [], "ingredients_text": "rice"}, ["Peanuts"])
    unknown = judge({"product_name": "?"}, ["Peanuts"])
    assert (risky["verdict"], trace["verdict"], safe["verdict"], unknown["verdict"]) == ("risky", "risky", "safe", "unknown")
    assert judge({"ingredients_text": "contains wheat flour"}, ["Wheat"])["verdict"] == "risky"   # ingredient text beats missing tags
