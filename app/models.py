"""Request/response schemas for POST /api/interview.

The wire contract is fixed by the technical spec: a request carries a
`sessionId` plus either a `candidate` (first call) or a `message` (every call
after). A response always carries `reply` and `done`, plus `feedback` on the
final turn.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Mission(BaseModel):
    model_config = ConfigDict(extra="allow")

    day: int
    title: str | None = None
    passed: bool | None = None
    attempts: int | None = None
    skipped: bool | None = None


class Member(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    jobRole: str | None = None
    yearsExperience: int | None = None
    education: str | None = None
    status: str | None = None


class Signals(BaseModel):
    model_config = ConfigDict(extra="allow")

    commitDays: int | None = None
    missionsCompleted: int | None = None
    missionsFirstTry: int | None = None


class Candidate(BaseModel):
    """Mirrors an entry in candidates.json. Tolerant of extra keys."""

    model_config = ConfigDict(extra="allow")

    member: Member = Field(default_factory=Member)
    missions: list[Mission] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)


class InterviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(min_length=1, max_length=200)
    candidate: Candidate | None = None
    message: str | None = Field(default=None, max_length=20_000)


class Feedback(BaseModel):
    summary: str
    strengths: list[str]
    gaps: list[str]
    next: list[str]


class Progress(BaseModel):
    """Non-contractual telemetry, useful for the UI and for graders."""

    questionsAsked: int
    minQuestions: int
    daysCovered: list[int]
    minDays: int
    currentTopic: str | None = None
    phase: Literal["intro", "in_progress", "complete"] = "in_progress"
    engine: Literal["llm", "heuristic"] = "heuristic"


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Feedback | None = None
    progress: Progress | None = None
    report: dict[str, Any] | None = None
