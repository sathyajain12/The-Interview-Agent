"""Planner, profile, scoring, and the model-backed code path."""

from __future__ import annotations

import uuid

import pytest

from app.config import settings
from app.curriculum import get_curriculum
from app.heuristics import evaluate
from app.interviewer import Interview
from app.llm import CLOSING_SCHEMA, OPENING_SCHEMA, TURN_SCHEMA
from app.models import Candidate
from app.planner import build_blueprint
from app.profile import CandidateProfile

# ------------------------------------------------------------------ profile


def test_attempts_drive_confidence(candidates):
    sarah = CandidateProfile.build(Candidate(**candidates[0]))
    first_try = sarah.by_day(7)
    hard_won = sarah.by_day(12)
    assert first_try.status == "mastered"
    assert hard_won.status == "struggled"
    assert first_try.confidence > hard_won.confidence


def test_depth_calibration_separates_profiles(candidates):
    by_name = {c["member"]["name"]: c for c in candidates}
    assert CandidateProfile.build(Candidate(**by_name["Diane Foster"])).depth == "advanced"
    assert CandidateProfile.build(Candidate(**by_name["Tyler Brooks"])).depth == "foundational"


def test_skips_and_failures_surface_as_gaps(candidates):
    by_name = {c["member"]["name"]: c for c in candidates}
    gerald = CandidateProfile.build(Candidate(**by_name["Gerald Combs"]))
    statuses = {s.status for s in gerald.gaps}
    assert "failed" in statuses and "skipped" in statuses


# ------------------------------------------------------------------ planner


@pytest.mark.parametrize("index", range(20))
def test_blueprint_meets_the_floor(candidates, index):
    profile = CandidateProfile.build(Candidate(**candidates[index]))
    blueprint = build_blueprint(profile)
    days = {d for slot in blueprint for d in slot.days}
    assert len(blueprint) >= settings.min_questions
    assert len(days) >= settings.min_days
    assert blueprint[0].kind == "warmup"
    assert blueprint[-1].kind == "closing"


def test_blueprint_targets_the_candidate(candidates):
    by_name = {c["member"]["name"]: c for c in candidates}
    profile = CandidateProfile.build(Candidate(**by_name["Sarah Johnson"]))
    blueprint = build_blueprint(profile)
    kinds = {s.kind for s in blueprint}
    assert {"warmup", "closing"} <= kinds
    # Day 29 was skipped, so it must be probed as a gap rather than ignored.
    gap_days = {d for s in blueprint if s.kind == "gap_bridge" for d in s.days}
    assert 29 in gap_days


def test_advanced_candidates_get_deeper_questions(candidates):
    by_name = {c["member"]["name"]: c for c in candidates}
    advanced = build_blueprint(CandidateProfile.build(Candidate(**by_name["Diane Foster"])))
    foundational = build_blueprint(CandidateProfile.build(Candidate(**by_name["Tyler Brooks"])))
    deep = lambda plan: sum(1 for s in plan if s.difficulty == "deep")  # noqa: E731
    assert deep(advanced) > deep(foundational)


# --------------------------------------------------------------- retrieval


def test_curriculum_retrieval_finds_the_right_day():
    cur = get_curriculum()
    top = cur.search("router that chooses between sql and semantic search")[0][0]
    assert top.day == 10
    assert 23 in {d.day for d, _ in cur.search("model context protocol server tools")}


# ------------------------------------------------------------------ scoring


def test_scoring_discriminates(candidates):
    profile = CandidateProfile.build(Candidate(**candidates[0]))
    slot = next(s for s in build_blueprint(profile) if s.rubric)

    strong = evaluate(
        " ".join(slot.rubric) + " We ran this in production with ChromaDB and measured p95 latency at 400ms "
        "across 12000 chunks, and I rewrote the merge step twice before the ranking was honest.",
        slot,
    )
    weak = evaluate("Basically it just kind of works, we send stuff and get things back.", slot)
    blank = evaluate("I don't know, I never got to that one.", slot)

    assert strong["score"] >= 4
    assert weak["score"] <= 2
    assert blank["score"] <= 1
    assert "dont_know" in blank["flags"]
    assert "vague" in weak["flags"]


def test_clarification_is_not_scored_as_an_answer(candidates):
    interview = Interview.start("clarify-test", Candidate(**candidates[0]))
    before = interview.asked
    interview.respond("Sorry, what do you mean exactly?")
    assert interview.asked == before, "a rephrase request must not consume a question"
    assert interview.clarifications_used == 1


# ------------------------------------------------------- model-backed path


class StubBrain:
    """Stands in for the Messages API so the LLM branch is exercised offline."""

    available = True
    status = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def structured(self, *, system, messages, schema, max_tokens, effort):
        assert "<candidate>" in system and "<curriculum>" in system
        assert messages and messages[-1]["role"] == "user"
        if schema is OPENING_SCHEMA:
            self.calls.append("opening")
            return {"reply": "Thanks for joining. Tell me about your capstone."}
        if schema is TURN_SCHEMA:
            self.calls.append("turn")
            return {
                "evaluation": {
                    "score": 4,
                    "covered": ["retrieval design"],
                    "missing": ["evaluation"],
                    "flags": ["concrete_example"],
                    "signal": "Concrete and specific.",
                },
                "recommendation": "advance",
                "follow_up": "Push further on that.",
                "advance": "Different topic. What did you measure?",
            }
        self.calls.append("closing")
        return {
            "evaluation": {"score": 3, "covered": [], "missing": [], "flags": [], "signal": "ok"},
            "reply": "That's everything - thanks.",
            "feedback": {
                "summary": "Strong on retrieval, thinner on evaluation.",
                "strengths": ["Explained the router with real trade-offs."],
                "gaps": ["No benchmark set for retrieval quality."],
                "next": ["Build a 30-question evaluation set for the capstone."],
            },
        }


def test_llm_path_produces_a_complete_interview(monkeypatch, candidates):
    import app.interviewer as engine

    stub = StubBrain()
    monkeypatch.setattr(engine, "get_brain", lambda: stub)

    interview = Interview.start(str(uuid.uuid4()), Candidate(**candidates[2]))
    assert interview.engine == "llm"

    guard = 0
    while not interview.done and guard < 40:
        guard += 1
        interview.respond("We used Chroma with metadata filters and measured recall at k=5.")

    assert interview.done
    assert interview.asked >= settings.min_questions
    assert len(interview.days_covered) >= settings.min_days
    assert interview.feedback["summary"].startswith("Strong on retrieval")
    assert stub.calls[0] == "opening" and stub.calls[-1] == "closing"


def test_engine_falls_back_when_the_model_fails(monkeypatch, candidates):
    import app.interviewer as engine

    class BrokenBrain:
        available = True
        status = "broken"

        def structured(self, **_kwargs):
            return None

    monkeypatch.setattr(engine, "get_brain", lambda: BrokenBrain())

    interview = Interview.start(str(uuid.uuid4()), Candidate(**candidates[3]))
    assert interview.engine == "heuristic"
    assert interview.last_reply.strip()

    guard = 0
    while not interview.done and guard < 40:
        guard += 1
        interview.respond("We used FastAPI with session-scoped conversation history in SQLite.")

    assert interview.done
    assert interview.feedback["summary"]
    assert interview.asked >= settings.min_questions
