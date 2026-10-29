"""Knowledge-base ingestion: turns the app's learning modules, menus and scenario
files into retrievable, source-attributed chunks."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .allergens import detect
from .config import DATA

KB_DIR = DATA / "knowledge"
SC_DIR = DATA / "scenarios"


@dataclass
class Chunk:
    id: str
    text: str            # text shown to the LLM / user
    title: str           # human-readable citation title
    source: str          # file name
    kind: str            # education | behaviour | menu | scenario | dialogue
    allergens: set = field(default_factory=set)

    @property
    def index_text(self) -> str:
        # Title is repeated so headings influence ranking (e.g. "Peanut:")
        return f"{self.title}. {self.title}. {self.text}"


def _clean_heading(h: str) -> str:
    return h.strip().rstrip(":").strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9'\"(])", text.strip())
    return [p for p in parts if p]


def _pack(paragraphs: list[str], max_words: int = 140) -> list[str]:
    """Pack paragraphs into ~max_words windows without cutting sentences."""
    out, cur, n = [], [], 0
    for para in paragraphs:
        for sent in _split_sentences(para) or [para]:
            w = len(sent.split())
            if cur and n + w > max_words:
                out.append(" ".join(cur))
                # 1-sentence overlap keeps context across boundaries
                cur, n = [cur[-1]], len(cur[-1].split())
            cur.append(sent)
            n += w
    if cur:
        out.append(" ".join(cur))
    return out


MODULE_NAMES = {"educational_module": "Allergen Education", "behavioural_module": "Behaviour & Safety"}
_BULLET = re.compile(r"^\s*[•\u2022\-\*]\s*")


def _units(content) -> list[tuple[str, str]]:
    """Turn raw paragraphs into ('sub'|'text', string) units.

    * short paragraphs ending in '?' or ':' are sub-headings
    * '•' bullets are folded into the list intro line that precedes them
    """
    if isinstance(content, str):
        content = [content]
    paras = []
    for c in content:
        if isinstance(c, str):
            paras += [x.strip() for x in c.replace("\xa0", " ").split("\n") if x.strip()]
    out: list[tuple[str, str]] = []
    for para in paras:
        words = len(para.split())
        if _BULLET.match(para):
            item = _BULLET.sub("", para).strip()
            if out and out[-1][0] == "list":
                out[-1] = ("list", out[-1][1] + "; " + item.rstrip(".;"))
            elif out and out[-1][0] == "intro":
                out[-1] = ("list", out[-1][1].rstrip(":;") + ": " + item.rstrip(".;"))
            else:
                out.append(("list", item.rstrip(".;")))
        elif words <= 14 and para.endswith((":", ";")) and not para.endswith("."):
            # "Peanuts often found in;" is a list intro when bullets follow, otherwise a sub-heading
            out.append(("intro", para))
        elif words <= 22 and para.endswith("?"):
            out.append(("sub", para.rstrip("?").strip() + "?"))
        else:
            if not para.endswith((".", "!", "?", ":", ";", '"', "”", ")")):
                para += "."
            out.append(("text", para))
    final = []
    for i, (k, t) in enumerate(out):
        if k == "intro":
            nxt = out[i + 1][0] if i + 1 < len(out) else None
            k = "sub" if nxt in (None, "text", "sub") and t.endswith(":") and len(t.split()) <= 6 else "text"
            t = t.rstrip(":;") if k == "sub" else t.rstrip(";") + "."
        elif k == "list":
            t = t.rstrip(".") + "."
        final.append((k, t))
    return final


def _load_module(path: Path, kind: str) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mod = MODULE_NAMES.get(path.stem, data.get("title", path.stem))
    chunks = []
    for si, sec in enumerate(data.get("sections", [])):
        heading = _clean_heading(sec.get("heading", ""))
        groups: list[tuple[str, list[str]]] = [("", [])]
        for k, t in _units(sec.get("content", [])):
            if k == "sub":
                groups.append((t, []))
            else:
                groups[-1][1].append(t)
        for gi, (sub, paras) in enumerate(groups):
            if not paras:
                continue
            title = f"{mod} › {heading}" + (f" › {sub}" if sub else "")
            for ci, piece in enumerate(_pack(paras)):
                chunks.append(Chunk(
                    id=f"{path.stem}:{si}:{gi}:{ci}", text=piece, title=title, source=path.name,
                    kind=kind, allergens=detect(heading + " " + sub + " " + piece[:300]),
                ))
    return chunks


def _load_menu(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    level = path.stem.replace("restaurant_", "").replace("menu", "beginner")
    chunks = []
    for sec in data.get("menu_sections", []):
        for it in sec.get("items", []):
            al = it.get("allergens", [])
            hid = it.get("hidden_allergens", [])
            safe = it.get("modifiable_to_safe")
            lines = [
                f"{it['name']} ({sec['section']}, £{it.get('price', '?')}): {it.get('description', '')}.",
                f"Declared allergens: {', '.join(al) if al else 'none declared'}.",
            ]
            if hid:
                lines.append(f"Hidden allergens to ask about: {'; '.join(hid)}.")
            if safe is not None:
                lines.append("Can be modified to be safe." if safe else "Cannot be reliably made safe for these allergens.")
            if it.get("suggested_questions"):
                lines.append("Good questions to ask: " + " ".join(it["suggested_questions"]))
            text = " ".join(lines)
            chunks.append(Chunk(
                id=f"{path.stem}:{it.get('id', it['name'])}", text=text,
                title=f"Training menu ({level}) › {it['name']}", source=path.name, kind="menu",
                allergens=detect(" ".join(al + hid)) | {a for x in al for a in detect(x)},
            ))
    return chunks


def _load_scenario(path: Path) -> list[Chunk]:
    d = json.loads(path.read_text(encoding="utf-8"))
    name = d.get("name") or d.get("title") or path.stem
    chunks = []
    tips = d.get("allergyTips") or d.get("learning_objectives") or []
    if tips:
        chunks.append(Chunk(f"{path.stem}:tips", f"Tips for '{name}': " + " ".join(t.rstrip(".") + "." for t in tips),
                            f"Practice scenario › {name}", path.name, "scenario", detect(" ".join(tips))))
    crit = d.get("success_criteria")
    if crit:
        chunks.append(Chunk(f"{path.stem}:success", f"Success looks like: " + "; ".join(crit) + ".",
                            f"Practice scenario › {name} › success", path.name, "scenario"))
    return chunks


def _load_dialogue(path: Path) -> list[Chunk]:
    lines = json.loads(path.read_text(encoding="utf-8"))
    user_lines = [l["line"] for l in lines if l.get("speaker") == "user"]
    if not user_lines:
        return []
    text = "Example assertive replies a teen can use with friends: " + " ".join(f'"{u}"' for u in user_lines)
    return [Chunk(f"{path.stem}:examples", text, f"Peer-pressure examples › {path.stem.replace('_', ' ')}",
                  path.name, "dialogue", detect(text))]


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(KB_DIR.glob("*.json")):
        if p.name == "educational_module.json":
            chunks += _load_module(p, "education")
        elif p.name == "behavioural_module.json":
            chunks += _load_module(p, "behaviour")
        elif p.name.startswith("restaurant_"):
            chunks += _load_menu(p)
        elif p.name.endswith("_dialogue.json"):
            chunks += _load_dialogue(p)
    for p in sorted(SC_DIR.glob("*.json")):
        if p.name != "scenario_manifest.json":
            chunks += _load_scenario(p)
    # Drop the duplicate advanced/intermediate scenario copies' empty chunks, ensure unique ids
    seen, uniq = set(), []
    for c in chunks:
        if c.id not in seen and c.text.strip():
            seen.add(c.id)
            uniq.append(c)
    return uniq
