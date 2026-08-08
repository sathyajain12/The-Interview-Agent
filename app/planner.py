"""Builds the interview blueprint before a single question is asked.

Letting a model free-associate its way through an interview is how you end up
with eight questions about embeddings and none about deployment. So coverage
is decided deterministically up front - which topics, in what order, at what
depth, probing for what - and the model is left to do the part it is actually
good at: phrasing, listening, and following up.

The blueprint guarantees the spec's floor (>= 8 questions, >= 4 curriculum
days) by construction rather than by hoping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import question_bank as qb
from .config import settings
from .curriculum import Day
from .profile import CandidateProfile, TopicSignal

SlotKind = Literal[
    "warmup",
    "strength_depth",
    "shaky_probe",
    "gap_bridge",
    "integration",
    "scenario",
    "closing",
]

Difficulty = Literal["core", "deep"]


@dataclass
class Slot:
    """One planned topic. May consume more than one question via follow-ups."""

    id: str
    kind: SlotKind
    days: list[int]
    label: str
    intent: str
    difficulty: Difficulty
    rubric: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    evidence: str = ""

    @property
    def primary_day(self) -> int:
        return self.days[0] if self.days else 0

    def brief(self) -> str:
        lines = [
            f"Topic: {self.label}",
            f"Curriculum days: {', '.join(str(d) for d in self.days) or 'n/a'}",
            f"Why this topic for this candidate: {self.evidence}",
            f"What to find out: {self.intent}",
            f"Depth: {self.difficulty}",
        ]
        if self.rubric:
            lines.append("A strong answer touches:")
            lines.extend(f"  - {r}" for r in self.rubric)
        if self.seeds:
            lines.append("Reference angles (rephrase in your own voice, do not read verbatim):")
            lines.extend(f"  - {s}" for s in self.seeds)
        return "\n".join(lines)


def _rubric_from(day: Day | None, limit: int = 3) -> list[str]:
    if not day:
        return []
    points = list(day.objectives[:limit])
    if day.tools:
        points.append(f"concrete reference to the tooling used ({', '.join(day.tools[:4])})")
    return points


def _first_unused(pool: list[TopicSignal], used: set[int]) -> TopicSignal | None:
    return next((s for s in pool if s.day not in used), None)


class Planner:
    def __init__(self, profile: CandidateProfile) -> None:
        self.profile = profile
        self.curriculum = profile.curriculum
        self._used: set[int] = set()

    # ------------------------------------------------------------------ slots

    def _slot(
        self,
        kind: SlotKind,
        signal: TopicSignal | None,
        *,
        intent: str,
        difficulty: Difficulty,
        seeds: list[str] | None = None,
        extra_days: list[int] | None = None,
        label: str | None = None,
        evidence: str | None = None,
    ) -> Slot:
        days = ([signal.day] if signal else []) + (extra_days or [])
        day_obj = signal.curriculum if signal else None
        self._used.update(days)
        return Slot(
            id=f"{kind}-{'-'.join(str(d) for d in days) or 'x'}",
            kind=kind,
            days=days,
            label=label or (signal.title if signal else kind.replace("_", " ").title()),
            intent=intent,
            difficulty=difficulty,
            rubric=_rubric_from(day_obj),
            seeds=seeds or (qb.probes_for(signal.day, difficulty) if signal else []),
            evidence=evidence or (signal.evidence() if signal else "general coverage"),
        )

    # --------------------------------------------------------------- sections

    def _warmup(self) -> Slot:
        capstone = self.profile.by_day(31)
        if capstone and capstone.completed:
            return self._slot(
                "warmup",
                capstone,
                intent=(
                    "Get them talking and establish what they personally built. Listen for whether "
                    "they describe a system they own or a tutorial they followed."
                ),
                difficulty="core",
            )
        anchor = self.profile.strengths[0] if self.profile.strengths else self.profile.signals[0]
        return self._slot(
            "warmup",
            anchor,
            intent="Open on their strongest ground so the interview starts with a real answer, not a stall.",
            difficulty="core",
        )

    def _strength(self, n: int) -> list[Slot]:
        depth: Difficulty = "deep" if self.profile.depth in {"advanced", "standard"} else "core"
        pool = [s for s in self.profile.strengths if s.day not in {1, 2, 31}]
        slots: list[Slot] = []
        for _ in range(n):
            sig = _first_unused(pool, self._used)
            if not sig:
                break
            slots.append(
                self._slot(
                    "strength_depth",
                    sig,
                    intent=(
                        f"They cleared this cleanly ({sig.evidence()}), so definitions prove nothing. "
                        "Push to the trade-off underneath and see whether the understanding is load-bearing."
                    ),
                    difficulty=depth,
                )
            )
        return slots

    def _shaky(self, n: int) -> list[Slot]:
        pool = self.profile.shaky + [s for s in self.profile.gaps if s.status == "failed"]
        slots: list[Slot] = []
        for _ in range(n):
            sig = _first_unused(pool, self._used)
            if not sig:
                break
            slots.append(
                self._slot(
                    "shaky_probe",
                    sig,
                    intent=(
                        f"This one cost them ({sig.evidence()}). Find out whether the struggle produced real "
                        "understanding or whether they got through it and moved on. Do not lead the witness."
                    ),
                    difficulty="core",
                )
            )
        return slots

    def _gaps(self, n: int) -> list[Slot]:
        slots: list[Slot] = []
        candidates = list(self.profile.gaps)

        for sig in candidates:
            if len(slots) >= n:
                break
            if sig.day in self._used:
                continue
            bridge = self.curriculum.neighbours(sig.day, limit=2, exclude=set())
            done = [d for d in bridge if (m := self.profile.by_day(d.day)) and m.completed]
            seed = qb.gap_probe_for(sig.day)
            bridge_note = (
                f" They did complete {done[0].label}, so anchor the question there."
                if done else ""
            )
            slots.append(
                self._slot(
                    "gap_bridge",
                    sig,
                    intent=(
                        f"This topic is missing from their record ({sig.evidence()}). Do not quiz them on "
                        "material they never covered - ask how they would approach it, and judge the reasoning."
                        + bridge_note
                    ),
                    difficulty="core",
                    seeds=[seed] if seed else qb.probes_for(sig.day, "core"),
                    extra_days=[done[0].day] if done else None,
                )
            )

        # No declared gaps? The silent ones still matter - a day that never
        # appears in the record at all is as much a gap as one marked skipped.
        if len(slots) < n:
            for day in self.profile.uncovered_days():
                if len(slots) >= n or day.day in self._used or day.day in {1, 2, 3}:
                    continue
                seed = qb.gap_probe_for(day.day) or (qb.probes_for(day.day, "core") or [None])[0]
                ghost = TopicSignal(
                    day=day.day,
                    title=day.title,
                    status="skipped",
                    confidence=0.0,
                    attempts=None,
                    module=day.module,
                    module_n=day.module_n,
                    curriculum=day,
                )
                slots.append(
                    self._slot(
                        "gap_bridge",
                        ghost,
                        intent=(
                            "This day never appears in their mission record at all. Treat it as an unknown, "
                            "not a failure: ask how they would reason about it and calibrate from the answer."
                        ),
                        difficulty="core",
                        seeds=[seed] if seed else [],
                        evidence="absent from the mission record",
                    )
                )
        return slots

    def _scenario(self) -> Slot | None:
        # Prefer an operational topic they actually shipped - a scenario is only
        # fair if they have ground to stand on.
        preference = [28, 29, 30, 27, 22, 16, 18, 11, 10, 8, 7]
        completed = {s.day for s in self.profile.signals if s.completed}
        for day in preference:
            if day in completed and qb.scenario_for(day):
                sig = self.profile.by_day(day)
                if not sig:
                    continue
                return self._slot(
                    "scenario",
                    sig,
                    intent=(
                        "Hand them a broken system, not a definition. Watch how they narrow the search space: "
                        "a good answer forms a hypothesis and says what would confirm it."
                    ),
                    difficulty="deep" if self.profile.depth != "foundational" else "core",
                    seeds=[qb.scenario_for(day) or ""],
                )
        return None

    def _integration(self) -> Slot | None:
        """Force a connection between two modules they both completed."""
        completed = [s for s in self.profile.signals if s.completed and s.curriculum]
        pairs: list[tuple[TopicSignal, TopicSignal]] = []
        for i, a in enumerate(completed):
            for b in completed[i + 1:]:
                if a.module_n != b.module_n and abs(a.day - b.day) >= 4 and 31 not in (a.day, b.day):
                    pairs.append((a, b))
        if not pairs:
            return None

        # Prefer a pair where at least one side is still unused, so integration
        # widens coverage rather than repeating it.
        pairs.sort(key=lambda p: (p[0].day in self._used) + (p[1].day in self._used))
        a, b = pairs[0]
        slot = self._slot(
            "integration",
            a,
            intent=(
                "Nobody teaches the seam between two modules; they either see it or they don't. "
                "Make them reason across both topics in one answer."
            ),
            difficulty="deep" if self.profile.depth == "advanced" else "core",
            extra_days=[b.day],
            label=f"{a.title} x {b.title}",
            seeds=[
                f"How do the decisions you made in '{a.title}' constrain what's possible in '{b.title}'?"
            ],
            evidence=f"both completed ({a.evidence()}; {b.evidence()})",
        )
        slot.rubric = _rubric_from(a.curriculum, 2) + _rubric_from(b.curriculum, 2)
        return slot

    def _closing(self) -> Slot:
        anchor = self.profile.by_day(31) or (self.profile.signals[0] if self.profile.signals else None)
        return self._slot(
            "closing",
            anchor,
            intent=(
                "Last question. Ask for honest self-assessment - what they'd rebuild, or what they know "
                "they still don't understand. Self-awareness is signal; a polished non-answer is also signal."
            ),
            difficulty="core",
            seeds=[
                "Looking at everything you built across the cohort, what's the piece you'd least want to defend in a design review, and why?"
            ],
            label="Reflection",
            evidence="closing question",
        )

    # ------------------------------------------------------------------ build

    def build(self) -> list[Slot]:
        blueprint: list[Slot] = [self._warmup()]

        strengths = self._strength(2)
        shaky = self._shaky(2)
        gaps = self._gaps(2)
        scenario = self._scenario()
        integration = self._integration()

        # Interleave so the interview alternates between comfortable and hard
        # ground rather than front-loading either.
        ordered: list[Slot | None] = [
            strengths[0] if strengths else None,
            shaky[0] if shaky else None,
            gaps[0] if gaps else None,
            strengths[1] if len(strengths) > 1 else None,
            scenario,
            shaky[1] if len(shaky) > 1 else None,
            integration,
            gaps[1] if len(gaps) > 1 else None,
        ]
        blueprint.extend(s for s in ordered if s is not None)
        blueprint.append(self._closing())

        return _ensure_floor(blueprint, self.profile)


def _ensure_floor(blueprint: list[Slot], profile: CandidateProfile) -> list[Slot]:
    """Backstop for thin records: top up until >= 8 slots across >= 4 days."""
    used = {d for slot in blueprint for d in slot.days}
    pool = sorted(profile.signals, key=lambda s: (-s.confidence, s.day))
    closing = blueprint.pop() if blueprint and blueprint[-1].kind == "closing" else None

    for sig in pool:
        if len(blueprint) + 1 >= settings.min_questions and len(used) >= settings.min_days:
            break
        if sig.day in used:
            continue
        used.add(sig.day)
        blueprint.append(
            Slot(
                id=f"coverage-{sig.day}",
                kind="strength_depth" if sig.completed else "gap_bridge",
                days=[sig.day],
                label=sig.title,
                intent="Additional coverage to meet the interview's breadth requirement.",
                difficulty="core",
                rubric=_rubric_from(sig.curriculum),
                seeds=qb.probes_for(sig.day, "core"),
                evidence=sig.evidence(),
            )
        )

    # Still short (very sparse record): widen into untouched curriculum days.
    for day in profile.uncovered_days():
        if len(blueprint) + 1 >= settings.min_questions and len(used) >= settings.min_days:
            break
        if day.day in used or day.day in {1, 2}:
            continue
        used.add(day.day)
        blueprint.append(
            Slot(
                id=f"coverage-{day.day}",
                kind="gap_bridge",
                days=[day.day],
                label=day.title,
                intent="Not in their record - ask how they would approach it and judge the reasoning.",
                difficulty="core",
                rubric=_rubric_from(day),
                seeds=qb.probes_for(day.day, "core"),
                evidence="absent from the mission record",
            )
        )

    if closing:
        blueprint.append(closing)
    return blueprint


def build_blueprint(profile: CandidateProfile) -> list[Slot]:
    return Planner(profile).build()
