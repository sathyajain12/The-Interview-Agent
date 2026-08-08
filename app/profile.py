"""Turn a raw candidate record into interview-relevant signal.

Attempt counts are the richest thing in the profile: a topic passed on the
first try and a topic passed on the fifth try are both "passed", but they
warrant completely different questions. This module makes that difference
explicit so the planner can act on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .curriculum import Curriculum, Day, get_curriculum
from .models import Candidate

Status = Literal["mastered", "solid", "shaky", "struggled", "failed", "skipped"]

# Confidence the candidate actually owns a topic, given how it was completed.
_ATTEMPT_CONFIDENCE = {1: 0.95, 2: 0.78, 3: 0.58, 4: 0.42, 5: 0.30}

_TECHNICAL_ROLE = re.compile(
    r"engineer|developer|architect|scientist|programmer|devops|sre|data|technical|cto|ai\b",
    re.IGNORECASE,
)


def _status_for(attempts: int | None, passed: bool | None, skipped: bool | None) -> tuple[Status, float]:
    if skipped:
        return "skipped", 0.0
    if passed is False:
        return "failed", 0.12
    a = attempts or 1
    conf = _ATTEMPT_CONFIDENCE.get(min(a, 5), 0.25)
    if a <= 1:
        return "mastered", conf
    if a == 2:
        return "solid", conf
    if a == 3:
        return "shaky", conf
    return "struggled", conf


@dataclass
class TopicSignal:
    day: int
    title: str
    status: Status
    confidence: float
    attempts: int | None
    module: str
    module_n: int
    curriculum: Day | None = None

    @property
    def completed(self) -> bool:
        return self.status not in {"skipped", "failed"}

    def evidence(self) -> str:
        if self.status == "skipped":
            return "skipped entirely"
        if self.status == "failed":
            return f"never passed after {self.attempts or '?'} attempts"
        if self.attempts == 1:
            return "passed first try"
        return f"passed on attempt {self.attempts}"


@dataclass
class CandidateProfile:
    name: str
    candidate_id: str
    job_role: str
    years_experience: int
    education: str
    signals: list[TopicSignal]
    commit_days: int
    missions_completed: int
    missions_first_try: int
    curriculum: Curriculum = field(repr=False)

    # ------------------------------------------------------------ aggregates

    @property
    def first_try_rate(self) -> float:
        if self.missions_completed <= 0:
            return 0.0
        return min(self.missions_first_try / self.missions_completed, 1.0)

    @property
    def average_attempts(self) -> float:
        vals = [s.attempts for s in self.signals if s.completed and s.attempts]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def consistency(self) -> float:
        return min(self.commit_days / 31.0, 1.0)

    @property
    def technical_background(self) -> bool:
        return bool(_TECHNICAL_ROLE.search(self.job_role or ""))

    @property
    def depth(self) -> Literal["foundational", "standard", "advanced"]:
        """How hard the interview should push, from demonstrated performance."""
        avg = self.average_attempts or 5.0
        rate = self.first_try_rate
        if rate >= 0.65 and avg <= 1.8:
            return "advanced"
        if rate <= 0.25 or avg >= 3.6:
            return "foundational"
        return "standard"

    # -------------------------------------------------------------- lookups

    def by_day(self, day: int) -> TopicSignal | None:
        return next((s for s in self.signals if s.day == day), None)

    @property
    def strengths(self) -> list[TopicSignal]:
        return sorted(
            (s for s in self.signals if s.completed and s.confidence >= 0.7),
            key=lambda s: (-s.confidence, s.day),
        )

    @property
    def shaky(self) -> list[TopicSignal]:
        return sorted(
            (s for s in self.signals if s.completed and s.confidence < 0.6),
            key=lambda s: (s.confidence, s.day),
        )

    @property
    def gaps(self) -> list[TopicSignal]:
        """Skipped or failed topics - probe for reasoning, not recall."""
        return sorted(
            (s for s in self.signals if not s.completed),
            key=lambda s: (0 if s.status == "failed" else 1, s.day),
        )

    @property
    def covered_days(self) -> set[int]:
        return {s.day for s in self.signals}

    def uncovered_days(self) -> list[Day]:
        """Curriculum days absent from the record entirely (silent gaps)."""
        return [d for n, d in sorted(self.curriculum.days.items()) if n not in self.covered_days]

    # ------------------------------------------------------------- rendering

    def dossier(self) -> str:
        """Compact candidate briefing for the interviewer's system prompt."""
        head = (
            f"{self.name} ({self.candidate_id}) - {self.job_role}, "
            f"{self.years_experience} yrs experience, {self.education}."
        )
        stats = (
            f"Cohort signals: {self.missions_completed} missions completed, "
            f"{self.missions_first_try} passed first try "
            f"({self.first_try_rate:.0%} first-try rate), "
            f"{self.commit_days}/31 active days, "
            f"average {self.average_attempts:.1f} attempts per passed mission."
        )
        calibration = (
            f"Calibration: {self.depth} depth; "
            f"{'technical' if self.technical_background else 'non-engineering'} day-job background."
        )
        rows = [
            f"  Day {s.day:>2} {s.title[:46]:<46} {s.status:<10} ({s.evidence()})"
            for s in sorted(self.signals, key=lambda s: s.day)
        ]
        return "\n".join([head, stats, calibration, "Mission record:", *rows])

    # -------------------------------------------------------------- factory

    @classmethod
    def build(cls, candidate: Candidate, curriculum: Curriculum | None = None) -> "CandidateProfile":
        cur = curriculum or get_curriculum()
        member = candidate.member

        signals: list[TopicSignal] = []
        for mission in candidate.missions:
            day = cur.get(mission.day)
            status, conf = _status_for(mission.attempts, mission.passed, mission.skipped)
            signals.append(
                TopicSignal(
                    day=mission.day,
                    title=mission.title or (day.title if day else f"Day {mission.day}"),
                    status=status,
                    confidence=conf,
                    attempts=mission.attempts,
                    module=day.module if day else "Unknown module",
                    module_n=day.module_n if day else 0,
                    curriculum=day,
                )
            )

        sig = candidate.signals
        completed = sig.missionsCompleted if sig.missionsCompleted is not None else len(
            [s for s in signals if s.completed]
        )
        return cls(
            name=member.name or "the candidate",
            candidate_id=member.id or "unknown",
            job_role=member.jobRole or "engineer",
            years_experience=member.yearsExperience if member.yearsExperience is not None else 0,
            education=member.education or "not provided",
            signals=signals,
            commit_days=sig.commitDays or 0,
            missions_completed=completed or 0,
            missions_first_try=sig.missionsFirstTry or 0,
            curriculum=cur,
        )
