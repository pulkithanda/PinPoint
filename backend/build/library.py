"""The hand-checked project library: load it, list it, and match a spoken request to it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from backend.build.board import BuildPlan

LIBRARY_PATH = Path(__file__).with_name("library.yaml")


@dataclass(frozen=True)
class LibraryEntry:
    plan: BuildPlan
    keywords: tuple[str, ...]


@lru_cache(maxsize=2)
def load_library(path: str | Path = LIBRARY_PATH) -> tuple[LibraryEntry, ...]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(LibraryEntry(BuildPlan.from_dict(p, source="library"), tuple(k.lower() for k in p.get("keywords", [])))
                 for p in d["projects"])


def get(project_id: str) -> BuildPlan | None:
    return next((e.plan for e in load_library() if e.plan.id == project_id), None)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _vocab(e: LibraryEntry) -> set[str]:
    return {k for k in e.keywords if " " not in k} | _words(e.plan.title)


def match(request: str, min_score: float = 0.5) -> BuildPlan | None:
    """Best library project for a spoken request, by keyword overlap (no Claude needed).
    Rare words count more: "joystick" says more than "light", which half the projects share."""
    req = request.lower()
    words = _words(request) | {w[:-1] for w in _words(request) if w.endswith("s")}
    entries = load_library()
    df: dict[str, int] = {}
    for e in entries:
        for w in _vocab(e):
            df[w] = df.get(w, 0) + 1
    best, best_score = None, 0.0
    for e in entries:
        score = sum(1.0 / df[w] for w in words & _vocab(e))
        score += sum(1.0 for k in e.keywords if " " in k and k in req)
        if score > best_score:
            best, best_score = e.plan, score
    return best if best_score >= min_score else None


_FILLER = {"yeah", "yes", "ok", "okay", "so", "um", "uh", "i", "i'd", "i'm", "id", "im", "want", "wanna", "would",
           "like", "to", "build", "make", "create", "do", "a", "an", "the", "that", "which", "me", "please", "can",
           "could", "you", "we", "let's", "lets", "some", "something", "thing", "just", "simple", "basic", "one",
           "how", "about", "maybe", "it", "is", "and"}


def confident_match(request: str, max_content_words: int = 4) -> BuildPlan | None:
    """A short request that clearly names one library project ("a blinking light", "yeah, I want to build a
    traffic light"). Used to skip the LLM entirely. Richer or ambiguous requests go to the planner."""
    words = _words(request)
    content = [w for w in re.findall(r"[a-z']+", request.lower()) if w not in _FILLER]
    if not content or len(content) > max_content_words:
        return None
    entries = load_library()
    df: dict[str, int] = {}
    for e in entries:
        for w in _vocab(e):
            df[w] = df.get(w, 0) + 1
    req = request.lower()
    scored = sorted(((sum(1.0 / df[w] for w in (words | {w[:-1] for w in words if w.endswith("s")}) & _vocab(e))
                      + sum(1.0 for k in e.keywords if " " in k and k in req), e.plan) for e in entries),
                    key=lambda x: x[0], reverse=True)
    (best, plan), (second, _) = scored[0], scored[1]
    entry = next(e for e in entries if e.plan is plan)
    vocab = _vocab(entry) | {k for e in [entry] for k in e.keywords}
    covered = sum(1 for w in content if w in vocab or w.rstrip("s") in vocab or w + "ing" in vocab)
    if covered < 0.6 * len(content):
        return None                        # "control my kettle": one keyword isn't enough
    return plan if best >= 1.0 and best >= 2 * max(second, 0.01) else None


def menu(difficulty: str | None = None) -> list[str]:
    return [e.plan.title for e in load_library() if difficulty is None or e.plan.difficulty == difficulty]


def menu_text() -> str:
    """Spoken list of ideas."""
    basic = ", ".join(t.lower() for t in menu("basic"))
    adv = ", ".join(t.lower() for t in menu("advanced"))
    return f"For example, beginner builds like a {basic}; or bigger ones like a {adv}."
