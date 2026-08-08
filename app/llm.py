"""Anthropic transport layer.

Everything the interviewer asks the model for is a fixed JSON shape, so every
call goes through structured outputs (`output_config.format`) rather than
prompt-and-pray parsing. If the API is unreachable, unauthenticated, or
refuses, `structured()` returns None and the caller drops to the offline
engine - the endpoint never fails because of the model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import settings

log = logging.getLogger("interview.llm")

# ---------------------------------------------------------------- schemas

_EVALUATION = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "0-5. 0 = no answer, 3 = correct but shallow, 5 = expert."},
        "covered": {"type": "array", "items": {"type": "string"}, "description": "Rubric points the answer actually hit."},
        "missing": {"type": "array", "items": {"type": "string"}, "description": "Rubric points the answer did not reach."},
        "flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "concrete_example",
                    "vague",
                    "memorised",
                    "dont_know",
                    "off_topic",
                    "contradiction",
                    "overclaiming",
                    "asked_for_clarification",
                ],
            },
        },
        "signal": {"type": "string", "description": "One sentence of interviewer's-notebook assessment."},
    },
    "required": ["score", "covered", "missing", "flags", "signal"],
    "additionalProperties": False,
}

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation": _EVALUATION,
        "recommendation": {
            "type": "string",
            "enum": ["follow_up", "advance"],
            "description": "Whether this topic deserves one more question or is exhausted.",
        },
        "follow_up": {
            "type": "string",
            "description": "Reply that stays on the current topic and digs into what the answer left open.",
        },
        "advance": {
            "type": "string",
            "description": "Reply that closes out this topic and opens the next one.",
        },
    },
    "required": ["evaluation", "recommendation", "follow_up", "advance"],
    "additionalProperties": False,
}

OPENING_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
    "additionalProperties": False,
}

CLOSING_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation": _EVALUATION,
        "reply": {"type": "string", "description": "Short spoken close of the interview. No feedback content here."},
        "feedback": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "gaps": {"type": "array", "items": {"type": "string"}},
                "next": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "strengths", "gaps", "next"],
            "additionalProperties": False,
        },
    },
    "required": ["evaluation", "reply", "feedback"],
    "additionalProperties": False,
}


class Brain:
    """Thin wrapper around the Messages API with a hard no-raise contract."""

    def __init__(self) -> None:
        self._client: Any = None
        self._disabled_reason: str | None = None
        self._use_fallbacks = True

        if not settings.llm_enabled:
            self._disabled_reason = "no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the environment"
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=2, timeout=90.0)
        except Exception as exc:  # pragma: no cover - import/credential edge
            self._disabled_reason = f"anthropic client unavailable: {exc}"

    # ------------------------------------------------------------- status

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def status(self) -> str:
        return f"llm:{settings.model}" if self.available else f"heuristic ({self._disabled_reason})"

    # -------------------------------------------------------------- calls

    def structured(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int,
        effort: str,
    ) -> dict[str, Any] | None:
        if not self._client:
            return None

        # Stable prefix first so the candidate dossier and curriculum pack are
        # cached across every turn of the interview.
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        payload: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        }

        for attempt in (1, 2):
            try:
                if self._use_fallbacks:
                    response = self._client.beta.messages.create(
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                        **payload,
                    )
                else:
                    response = self._client.messages.create(**payload)
                break
            except Exception as exc:
                # A rejected beta flag should cost us one retry, not the feature.
                if attempt == 1 and self._use_fallbacks and _looks_like_beta_rejection(exc):
                    log.info("server-side fallbacks unavailable, continuing without them: %s", exc)
                    self._use_fallbacks = False
                    continue
                log.warning("model call failed, falling back to offline engine: %s", exc)
                return None

        if getattr(response, "stop_reason", None) == "refusal":
            log.warning("model declined the request; falling back to offline engine")
            return None

        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("structured output was not valid JSON")
            return None
        return parsed if isinstance(parsed, dict) else None


def _looks_like_beta_rejection(exc: Exception) -> bool:
    blob = f"{type(exc).__name__}: {exc}".lower()
    return "400" in blob and ("beta" in blob or "fallback" in blob)


_brain: Brain | None = None


def get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain
