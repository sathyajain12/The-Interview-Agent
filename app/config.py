"""Runtime configuration, read once at import."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _has_credentials() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    )


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("INTERVIEW_MODEL", "claude-opus-5"))

    # Interview shape. The spec's floor is 8 questions across 4 curriculum days;
    # the planner builds a blueprint above that floor and the engine enforces it.
    min_questions: int = field(default_factory=lambda: _int("INTERVIEW_MIN_QUESTIONS", 8))
    max_questions: int = field(default_factory=lambda: _int("INTERVIEW_MAX_QUESTIONS", 14))
    min_days: int = field(default_factory=lambda: _int("INTERVIEW_MIN_DAYS", 4))
    max_followups_per_topic: int = field(default_factory=lambda: _int("INTERVIEW_MAX_FOLLOWUPS", 2))

    # Session retention. Long-term history is explicitly out of scope, so
    # sessions live in-process and expire.
    session_ttl_seconds: int = field(default_factory=lambda: _int("INTERVIEW_SESSION_TTL", 6 * 3600))
    max_sessions: int = field(default_factory=lambda: _int("INTERVIEW_MAX_SESSIONS", 500))

    # LLM budget. Turn generation is latency-sensitive, the closing report is not.
    turn_max_tokens: int = field(default_factory=lambda: _int("INTERVIEW_TURN_MAX_TOKENS", 2000))
    report_max_tokens: int = field(default_factory=lambda: _int("INTERVIEW_REPORT_MAX_TOKENS", 4000))
    turn_effort: str = field(default_factory=lambda: os.environ.get("INTERVIEW_TURN_EFFORT", "low"))
    report_effort: str = field(default_factory=lambda: os.environ.get("INTERVIEW_REPORT_EFFORT", "medium"))

    llm_enabled: bool = field(default_factory=lambda: os.environ.get("INTERVIEW_FORCE_OFFLINE", "").strip().lower()
                              not in {"1", "true", "yes"} and _has_credentials())

    curriculum_path: Path = DATA_DIR / "curriculum.json"
    candidates_path: Path = DATA_DIR / "candidates.json"


settings = Settings()
