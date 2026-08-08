"""The interview engine: state machine over a planned blueprint.

Division of responsibility:

* The blueprint (planner.py) owns *coverage* - which topics, in what order.
  It is fixed at session start, which is what makes the >= 8 questions across
  >= 4 curriculum days guarantee hold regardless of what the model does.
* This module owns *routing* - follow up on this topic or move to the next,
  push harder or ease off, keep going or wrap up.
* The model owns *voice* - phrasing, listening, and judging the answer.

Each turn is a single model call that returns the evaluation plus both possible
replies (follow-up and advance); this engine decides which one to send. That
keeps turn latency to one round trip without handing coverage control to the
model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import heuristics, prompts
from .config import settings
from .llm import CLOSING_SCHEMA, OPENING_SCHEMA, TURN_SCHEMA, get_brain
from .models import Candidate
from .planner import Slot, build_blueprint
from .profile import CandidateProfile

MAX_CLARIFICATIONS_PER_SLOT = 2


@dataclass
class TopicRecord:
    slot_id: str
    kind: str
    label: str
    days: list[int]
    question: str
    answer: str | None = None
    evaluation: dict[str, Any] | None = None

    def digest(self) -> str:
        if not self.evaluation:
            return f"- {self.label}: (unanswered)"
        ev = self.evaluation
        flags = ", ".join(ev.get("flags") or []) or "none"
        return (
            f"- {self.label} (days {', '.join(map(str, self.days))}): score {ev['score']}/5, "
            f"flags: {flags}. {ev.get('signal', '')}"
        )


@dataclass
class Interview:
    session_id: str
    profile: CandidateProfile
    blueprint: list[Slot]
    system: str = field(repr=False, default="")

    slot_index: int = 0
    followups_used: int = 0
    clarifications_used: int = 0
    asked: int = 0
    days_covered: set[int] = field(default_factory=set)
    transcript: list[dict[str, str]] = field(default_factory=list)
    records: list[TopicRecord] = field(default_factory=list)

    done: bool = False
    feedback: dict[str, Any] | None = None
    last_reply: str = ""
    engine: str = "heuristic"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------- factory

    @classmethod
    def start(cls, session_id: str, candidate: Candidate) -> "Interview":
        profile = CandidateProfile.build(candidate)
        blueprint = build_blueprint(profile)
        interview = cls(
            session_id=session_id,
            profile=profile,
            blueprint=blueprint,
            system=prompts.system_prompt(profile, blueprint),
        )
        interview._open()
        return interview

    # ---------------------------------------------------------------- state

    @property
    def current(self) -> Slot:
        return self.blueprint[min(self.slot_index, len(self.blueprint) - 1)]

    @property
    def upcoming(self) -> Slot | None:
        nxt = self.slot_index + 1
        return self.blueprint[nxt] if nxt < len(self.blueprint) else None

    @property
    def followups_allowed(self) -> int:
        kind = self.current.kind
        if kind == "closing":
            # The closing reflection gets one shot; drilling into it is badgering.
            return 0
        if kind == "warmup":
            # The opener is for getting them talking, not for interrogation.
            return min(1, settings.max_followups_per_topic)
        return settings.max_followups_per_topic

    def _floor_met(self) -> bool:
        return self.asked >= settings.min_questions and len(self.days_covered) >= settings.min_days

    def _will_close(self) -> bool:
        """Decided before the model call so a turn is never two round trips."""
        if not self._floor_met():
            return False
        if self.asked >= settings.max_questions:
            return True
        at_last = self.slot_index >= len(self.blueprint) - 1
        return at_last and self.followups_used >= self.followups_allowed

    def _momentum(self) -> str:
        scores = [r.evaluation["score"] for r in self.records if r.evaluation][-3:]
        if not scores:
            return "no answers yet"
        avg = sum(scores) / len(scores)
        if avg >= 4:
            return "handling this comfortably - raise the difficulty and stop accepting definitions"
        if avg <= 1.8:
            return "struggling - simplify the next question, ask for one concrete thing, do not stack conditions"
        return "steady - keep the pressure where it is"

    def _adapt(self, slot: Slot | None) -> None:
        """Nudge the next topic's depth from how the last answers went."""
        if slot is None or slot.kind == "closing":
            return
        scores = [r.evaluation["score"] for r in self.records if r.evaluation][-2:]
        if len(scores) < 2:
            return
        avg = sum(scores) / len(scores)
        if avg >= 4 and slot.difficulty == "core":
            slot.difficulty = "deep"
            slot.seeds = slot.seeds or []
        elif avg <= 1.5 and slot.difficulty == "deep":
            slot.difficulty = "core"

    # ------------------------------------------------------------ mechanics

    def _ask(self, text: str, slot: Slot) -> None:
        self.asked += 1
        self.days_covered.update(slot.days)
        self.transcript.append({"role": "interviewer", "text": text})
        self.records.append(
            TopicRecord(
                slot_id=slot.id,
                kind=slot.kind,
                label=slot.label,
                days=list(slot.days),
                question=text,
            )
        )
        self.last_reply = text
        self.updated_at = time.time()

    def _messages(self, directive: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for turn in self.transcript:
            role = "assistant" if turn["role"] == "interviewer" else "user"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n\n" + turn["text"]
            else:
                msgs.append({"role": role, "content": turn["text"]})
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": directive})
        else:
            msgs[-1]["content"] += "\n\n" + directive
        return msgs

    def _digest(self) -> str:
        return "\n".join(r.digest() for r in self.records) or "(no answers recorded)"

    # ---------------------------------------------------------------- turns

    def _open(self) -> None:
        slot = self.current
        brain = get_brain()
        result = brain.structured(
            system=self.system,
            messages=[{"role": "user", "content": prompts.opening_directive(self.profile, slot)}],
            schema=OPENING_SCHEMA,
            max_tokens=settings.turn_max_tokens,
            effort=settings.turn_effort,
        )
        if result and isinstance(result.get("reply"), str) and result["reply"].strip():
            self.engine = "llm"
            self._ask(result["reply"].strip(), slot)
        else:
            self.engine = "heuristic"
            self._ask(heuristics.opening(self.profile, slot), slot)

    def respond(self, message: str) -> None:
        """Consume one candidate answer and produce the next interviewer turn."""
        if self.done:
            return

        answer = (message or "").strip()
        self.transcript.append({"role": "candidate", "text": answer or "(no answer)"})
        slot = self.current

        if self._will_close():
            self._close(answer, slot)
            return

        brain = get_brain()
        directive = prompts.turn_directive(
            current=slot,
            upcoming=self.upcoming,
            followups_used=self.followups_used,
            followups_allowed=self.followups_allowed,
            asked=self.asked,
            remaining=max(len(self.blueprint) - self.slot_index - 1, 1),
            momentum=self._momentum(),
        )
        result = brain.structured(
            system=self.system,
            messages=self._messages(directive),
            schema=TURN_SCHEMA,
            max_tokens=settings.turn_max_tokens,
            effort=settings.turn_effort,
        )

        if result and _valid_turn(result):
            self.engine = "llm"
            evaluation = _clean_eval(result["evaluation"], slot)
            recommendation = result["recommendation"]
            follow_text = result["follow_up"].strip()
            advance_text = result["advance"].strip()
            clarify_text = follow_text
        else:
            self.engine = "heuristic"
            evaluation = heuristics.evaluate(answer, slot)
            recommendation = _heuristic_recommendation(evaluation)
            follow_text = heuristics.follow_up(slot, evaluation)
            advance_text = heuristics.advance(self.upcoming, evaluation, deep=self._prefers_deep())
            clarify_text = heuristics.clarify(slot)

        self._record_answer(answer, evaluation)

        # A request to rephrase is not an answer - re-ask instead of scoring it,
        # and do not let it consume one of the interview's questions.
        if (
            "asked_for_clarification" in evaluation.get("flags", [])
            and self.clarifications_used < MAX_CLARIFICATIONS_PER_SLOT
        ):
            self.clarifications_used += 1
            self.records.pop()
            self.asked = max(self.asked - 1, 0)
            self._ask(clarify_text or heuristics.clarify(slot), slot)
            return

        wants_followup = (
            recommendation == "follow_up"
            and self.followups_used < self.followups_allowed
            and self.asked < settings.max_questions
            and "dont_know" not in evaluation.get("flags", [])
        )

        if wants_followup:
            self.followups_used += 1
            self._ask(follow_text or heuristics.follow_up(slot, evaluation), slot)
            return

        # Advance.
        self.slot_index += 1
        self.followups_used = 0
        self.clarifications_used = 0
        nxt = self.current if self.slot_index < len(self.blueprint) else None
        self._adapt(nxt)
        if nxt is None:
            self._close(answer, slot, already_recorded=True)
            return
        self._ask(advance_text or heuristics.advance(nxt, evaluation), nxt)

    def _close(self, answer: str, slot: Slot, *, already_recorded: bool = False) -> None:
        brain = get_brain()
        result = brain.structured(
            system=self.system,
            messages=self._messages(
                prompts.closing_directive(
                    current=slot, evaluations_digest=self._digest(), profile=self.profile
                )
            ),
            schema=CLOSING_SCHEMA,
            max_tokens=settings.report_max_tokens,
            effort=settings.report_effort,
        )

        if result and _valid_closing(result):
            self.engine = "llm"
            if not already_recorded:
                self._record_answer(answer, _clean_eval(result["evaluation"], slot))
            reply = result["reply"].strip()
            feedback = _clean_feedback(result["feedback"])
        else:
            self.engine = "heuristic"
            if not already_recorded:
                self._record_answer(answer, heuristics.evaluate(answer, slot))
            reply = heuristics.closing_reply(self.profile)
            feedback = heuristics.feedback(self.profile, [r.__dict__ for r in self.records])

        self.done = True
        self.feedback = feedback
        self.last_reply = reply
        self.transcript.append({"role": "interviewer", "text": reply})
        self.updated_at = time.time()

    # --------------------------------------------------------------- helpers

    def _record_answer(self, answer: str, evaluation: dict[str, Any]) -> None:
        if self.records:
            self.records[-1].answer = answer
            self.records[-1].evaluation = evaluation

    def _prefers_deep(self) -> bool:
        scores = [r.evaluation["score"] for r in self.records if r.evaluation][-2:]
        return bool(scores) and sum(scores) / len(scores) >= 4

    # ---------------------------------------------------------------- output

    def scorecard(self) -> dict[str, Any]:
        answered = [r for r in self.records if r.evaluation]
        scores = [r.evaluation["score"] for r in answered]
        return {
            "candidate": self.profile.name,
            "candidateId": self.profile.candidate_id,
            "calibration": self.profile.depth,
            "engine": self.engine,
            "questionsAsked": self.asked,
            "questionsAnswered": len(answered),
            "daysCovered": sorted(self.days_covered),
            "modulesCovered": sorted(
                {
                    self.profile.curriculum.days[d].module
                    for d in self.days_covered
                    if d in self.profile.curriculum.days
                }
            ),
            "averageScore": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "topics": [
                {
                    "topic": r.label,
                    "kind": r.kind,
                    "days": r.days,
                    "score": r.evaluation["score"] if r.evaluation else None,
                    "flags": (r.evaluation or {}).get("flags", []),
                    "note": (r.evaluation or {}).get("signal", ""),
                }
                for r in self.records
            ],
        }

    def progress(self) -> dict[str, Any]:
        return {
            "questionsAsked": self.asked,
            "minQuestions": settings.min_questions,
            "daysCovered": sorted(self.days_covered),
            "minDays": settings.min_days,
            "currentTopic": None if self.done else self.current.label,
            "phase": "complete" if self.done else ("intro" if self.asked <= 1 else "in_progress"),
            "engine": "llm" if self.engine == "llm" else "heuristic",
        }


# ------------------------------------------------------------- validation


def _valid_turn(result: dict[str, Any]) -> bool:
    return (
        isinstance(result.get("evaluation"), dict)
        and result.get("recommendation") in {"follow_up", "advance"}
        and isinstance(result.get("follow_up"), str)
        and isinstance(result.get("advance"), str)
        and bool(result["follow_up"].strip() or result["advance"].strip())
    )


def _valid_closing(result: dict[str, Any]) -> bool:
    fb = result.get("feedback")
    return (
        isinstance(result.get("evaluation"), dict)
        and isinstance(result.get("reply"), str)
        and bool(result["reply"].strip())
        and isinstance(fb, dict)
        and isinstance(fb.get("summary"), str)
        and bool(fb["summary"].strip())
        and all(isinstance(fb.get(k), list) for k in ("strengths", "gaps", "next"))
    )


def _clean_eval(raw: dict[str, Any], slot: Slot) -> dict[str, Any]:
    try:
        score = int(raw.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    return {
        "score": max(0, min(5, score)),
        "covered": [str(x) for x in (raw.get("covered") or [])][:6],
        "missing": [str(x) for x in (raw.get("missing") or [])][:6] or list(slot.rubric[:2]),
        "flags": [str(x) for x in (raw.get("flags") or [])][:6],
        "signal": str(raw.get("signal", ""))[:400],
    }


def _clean_feedback(raw: dict[str, Any]) -> dict[str, Any]:
    def items(key: str, limit: int) -> list[str]:
        return [str(x).strip() for x in (raw.get(key) or []) if str(x).strip()][:limit]

    return {
        "summary": str(raw.get("summary", "")).strip(),
        "strengths": items("strengths", 5),
        "gaps": items("gaps", 5),
        "next": items("next", 6),
    }


def _heuristic_recommendation(evaluation: dict[str, Any]) -> str:
    flags = evaluation.get("flags", [])
    if "dont_know" in flags or "asked_for_clarification" in flags:
        return "advance"
    score = evaluation.get("score", 0)
    # Push on strong answers (is it real?) and on thin-but-recoverable ones.
    return "follow_up" if score >= 4 or score == 2 else "advance"
