# Aahar.AI

A RAG-powered allergy companion for teenagers, rebuilt in Python from the **AllerTeens** Flutter app.
Same screens, colours, spacing, copy and flow. The AI is now grounded in the app's own allergy-education content,
shows its sources, and refuses to guess.

```
pip install -r requirements.txt
python run.py            # http://localhost:8000   (open on a phone-sized window)
python -m pytest -q      # 28 tests
python -m tests.eval_rag # retrieval quality report
```

## What is RAG here?

`data/knowledge/*` (Allergen Education + Behaviour & Safety modules, training menus, scenario tips, peer-pressure dialogue)
-> **structure-aware chunks** (170) that keep their section + sub-heading (e.g. *Behaviour & Safety › Adrenaline › How does adrenaline work?*)
-> **hybrid retrieval**: BM25 on text + BM25 on titles + character n-gram TF-IDF (typos, plurals), optional dense embeddings,
fused by normalised score, with allergen-aware boosting and an allergen synonym ontology (tahini -> Sesame, prawn -> Crustaceans, ...)
-> **confidence gate**: out-of-scope questions are refused, never force-matched
-> **grounded answer**: LLM (OpenAI/Anthropic, optional) must cite `[n]` sources and uncited output is discarded;
without an API key the answer is quoted passages from the sources (offline, deterministic)
-> **safety layer** that never depends on retrieval or an LLM: live-reaction wording always shows
"use your adrenaline pen, call 112/999" first; self-harm wording gets a support message.

Your allergen list personalises answers ("Heads-up: Peanuts is on your list") deterministically, not via the LLM.

## Measured quality (tests/gold.py, 35 questions + 8 off-topic + 7 in-scope phrasings)

| metric | result |
|---|---|
| Hit@1 / Hit@3 / MRR | 91.4% / 100% / 0.957 |
| Hit@1 on lowercased, unpunctuated variants | 88.6% |
| Off-topic questions refused | 8 / 8 |
| In-scope questions answered | 7 / 7 |

Caveat: the gold set was written by the developer alongside the system and is small, so treat it as a regression guard,
not a benchmark. Fusion weights were checked on it and results sit on a flat plateau (differences of 1-2 questions).
Add your own questions to `tests/gold.py` (real teen questions are the best test).

## Screens (ported 1:1 from the Flutter app)

Onboarding · Sign up · Log in · Allergen picker (14 allergens) · Medical info / emergency contact · Home (greeting, Accidental Exposure,
Scan Food, emergency call FAB) · AI Training Scenarios (Groups/Beginner/Intermediate/Advanced, same unlock rules) with scored conversations
and feedback · Log (Monday-first calendar, exposure form with the original 20 symptoms, severity, medication) · Learn (Allergen Education /
Behaviour & Safety, search, expandable cards) · Profile. **New:** *Ask Aahar.AI* (home card + Learn "no results" fallback).

Colours, radii, paddings and icons come straight from `AppColors` / `AppConstants` / `assets/`.

## Not ported / known gaps

* Google sign-in (needs your OAuth client ID; the button is present and says so).
* Adrenaline-pen reminder push notifications and the Flame waiter sprite animation / OpenAI text-to-speech.
  (Scenario replies are text; the mic button uses the browser's speech recognition where available.)
* Firebase is replaced by SQLite (`aahar.db`) with PBKDF2-hashed passwords and bearer sessions. Email verification is not implemented.
* Barcode camera scanning uses the browser `BarcodeDetector` API (Chrome/Edge/Android); other browsers fall back to typing the number.
  Product data comes from Open Food Facts, which can be incomplete: "safe" only means *no match in the listed data*.
* Scenario waiter/friend replies are retrieval-based (menu records, scripted friend lines, coaching tip from the modules), not free-form LLM role-play.
* The LLM paths (OpenAI/Anthropic) are implemented and unit-tested with a stub, but were not exercised against the live APIs.
  Dense retrieval (`AAHAR_DENSE_MODEL`) is likewise untested here.

## Security note

The original `app_config.dart` contains a hard-coded OpenAI API key. **Revoke it** in your OpenAI dashboard and remove it from git history.
Aahar.AI reads keys only from environment variables / `.env`.

This tool shares general education from the source material. It is not medical advice; follow your allergy action plan.
