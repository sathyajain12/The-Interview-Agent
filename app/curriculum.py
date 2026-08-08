"""In-process retrieval over the 31-day curriculum.

The corpus is ~31 short documents, so a vector database would be pure ceremony
here: a TF-IDF-weighted bag of words over title + objectives + tools gives
better precision at zero operational cost and stays deterministic, which
matters because the same index feeds both the LLM path and the offline path.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import settings

_WORD = re.compile(r"[a-z0-9+#.]+")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "this", "to", "up", "use", "using", "was", "were",
    "what", "when", "where", "which", "while", "will", "with", "your", "you",
}

# Words that mean nothing on their own inside this corpus - every day mentions
# them - so they carry no retrieval signal.
_DOMAIN_STOP = {"chatbot", "healthcare", "project", "build", "create", "python", "chatbots"}


def _stem(token: str) -> str:
    """Crude plural folding so 'agents' and 'agent' match. Good enough here."""
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall(text.lower()):
        if len(raw) < 2 or raw in _STOP or raw in _DOMAIN_STOP:
            continue
        stemmed = _stem(raw)
        if stemmed not in _DOMAIN_STOP:
            out.append(stemmed)
    return out


@dataclass(frozen=True)
class Day:
    day: int
    title: str
    type: str
    tools: tuple[str, ...]
    objectives: tuple[str, ...]
    module: str
    module_n: int

    @property
    def label(self) -> str:
        return f"Day {self.day} - {self.title}"

    def brief(self) -> str:
        """Compact context block handed to the LLM."""
        lines = [f"{self.label} [{self.type}] (Module {self.module_n}: {self.module})"]
        if self.tools:
            lines.append(f"  Tools: {', '.join(self.tools)}")
        for obj in self.objectives:
            lines.append(f"  - {obj}")
        return "\n".join(lines)

    def searchable(self) -> str:
        return " ".join([self.title, self.type, " ".join(self.tools), " ".join(self.objectives)])


@dataclass
class Curriculum:
    cohort: str
    days: dict[int, Day]
    modules: dict[int, str]
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _vectors: dict[int, dict[str, float]] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Path) -> "Curriculum":
        raw = json.loads(path.read_text(encoding="utf-8"))

        modules: dict[int, str] = {}
        ranges: list[tuple[int, int, int, str]] = []
        for mod in raw.get("modules", []):
            span = mod.get("days") or []
            start = span[0] if span else 0
            end = span[-1] if span else 0
            modules[mod["n"]] = mod["title"]
            ranges.append((start, end, mod["n"], mod["title"]))

        days: dict[int, Day] = {}
        for entry in raw.get("days", []):
            n = int(entry["day"])
            module_n, module_title = 0, "Uncategorised"
            for start, end, mn, mt in ranges:
                if start <= n <= end:
                    module_n, module_title = mn, mt
                    break
            days[n] = Day(
                day=n,
                title=str(entry.get("title", f"Day {n}")),
                type=str(entry.get("type", "BUILD")),
                tools=tuple(entry.get("tools", []) or []),
                objectives=tuple(entry.get("objectives", []) or []),
                module=module_title,
                module_n=module_n,
            )

        inst = cls(cohort=str(raw.get("cohort", "AI Cohort")), days=days, modules=modules)
        inst._build_index()
        return inst

    def _build_index(self) -> None:
        docs = {n: _tokens(d.searchable()) for n, d in self.days.items()}
        total = max(len(docs), 1)
        df: Counter[str] = Counter()
        for toks in docs.values():
            df.update(set(toks))
        self._idf = {term: math.log(1 + total / (1 + count)) for term, count in df.items()}

        for n, toks in docs.items():
            tf = Counter(toks)
            longest = max(tf.values()) if tf else 1
            vec = {t: (0.5 + 0.5 * c / longest) * self._idf.get(t, 0.0) for t, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._vectors[n] = {t: v / norm for t, v in vec.items()}

    # -------------------------------------------------------------- retrieval

    def get(self, day: int) -> Day | None:
        return self.days.get(day)

    def search(self, query: str, *, limit: int = 5, exclude: set[int] | None = None) -> list[tuple[Day, float]]:
        """Rank curriculum days by cosine similarity to a free-text query."""
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        tf = Counter(q_tokens)
        longest = max(tf.values())
        q = {t: (0.5 + 0.5 * c / longest) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        q = {t: v / norm for t, v in q.items()}

        exclude = exclude or set()
        scored: list[tuple[Day, float]] = []
        for n, vec in self._vectors.items():
            if n in exclude:
                continue
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((self.days[n], score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].day))
        return scored[:limit]

    def neighbours(self, day: int, *, limit: int = 3, exclude: set[int] | None = None) -> list[Day]:
        """Curriculum days most similar to a given day - used to bridge gaps."""
        target = self.days.get(day)
        if not target:
            return []
        skip = {day} | (exclude or set())
        return [d for d, _ in self.search(target.searchable(), limit=limit + len(skip), exclude=skip)][:limit]

    def module_days(self, module_n: int) -> list[Day]:
        return sorted((d for d in self.days.values() if d.module_n == module_n), key=lambda d: d.day)

    def context_pack(self, days: list[int]) -> str:
        """Curriculum extract for the days an interview will actually touch."""
        seen: set[int] = set()
        blocks: list[str] = []
        for n in days:
            if n in seen:
                continue
            seen.add(n)
            day = self.days.get(n)
            if day:
                blocks.append(day.brief())
        return "\n".join(blocks)


@lru_cache(maxsize=1)
def get_curriculum() -> Curriculum:
    return Curriculum.load(settings.curriculum_path)
