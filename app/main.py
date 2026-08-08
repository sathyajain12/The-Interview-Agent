"""HTTP surface.

`POST /api/interview` is the contract from the technical spec. Everything else
is scaffolding for running and inspecting it: the demo UI, the candidate and
curriculum fixtures it loads, and a per-session scorecard.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR, settings
from .curriculum import get_curriculum
from .interviewer import Interview
from .llm import get_brain
from .models import Feedback, InterviewRequest, InterviewResponse, Progress
from .store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("interview.api")

app = FastAPI(
    title="AI Cohort Interview Agent",
    version="1.0.0",
    description=(
        "Conducts a personalised, multi-turn technical interview against a candidate's "
        "31-day AI cohort record and returns structured feedback."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _candidates() -> list[dict[str, Any]]:
    raw = json.loads(settings.candidates_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    return raw.get("candidates", [])


def _payload(interview: Interview) -> InterviewResponse:
    return InterviewResponse(
        reply=interview.last_reply,
        done=interview.done,
        feedback=Feedback(**interview.feedback) if interview.done and interview.feedback else None,
        progress=Progress(**interview.progress()),
        report=interview.scorecard() if interview.done else None,
    )


# ------------------------------------------------------------------ contract


@app.post("/api/interview", response_model=InterviewResponse, response_model_exclude_none=True)
def interview_endpoint(request: InterviewRequest) -> InterviewResponse:
    session_id = request.sessionId.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="sessionId must be a non-empty string")

    with store.lock_for(session_id):
        interview = store.get(session_id)

        if interview is None:
            if request.candidate is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No interview in progress for sessionId '{session_id}'. "
                        "Start one by posting {sessionId, candidate}."
                    ),
                )
            try:
                interview = Interview.start(session_id, request.candidate)
            except Exception:
                log.exception("failed to start interview for %s", session_id)
                raise HTTPException(status_code=500, detail="Could not start the interview.")
            store.put(session_id, interview)
            # Tolerate a client that sends the candidate and a first answer together.
            if request.message and request.message.strip():
                interview.respond(request.message)
            return _payload(interview)

        if request.message is None:
            # Idempotent re-poll: hand back the current turn unchanged.
            return _payload(interview)

        try:
            interview.respond(request.message)
        except Exception:
            log.exception("turn failed for %s", session_id)
            raise HTTPException(status_code=500, detail="Could not process that answer.")
        return _payload(interview)


# --------------------------------------------------------------- inspection


@app.get("/api/interview/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    interview = store.get(session_id)
    if interview is None:
        raise HTTPException(status_code=404, detail=f"Unknown sessionId '{session_id}'.")
    return JSONResponse(
        {
            "sessionId": session_id,
            "done": interview.done,
            "progress": interview.progress(),
            "plan": [
                {"kind": s.kind, "label": s.label, "days": s.days, "difficulty": s.difficulty}
                for s in interview.blueprint
            ],
            "transcript": interview.transcript,
            "report": interview.scorecard(),
            "feedback": interview.feedback,
        }
    )


@app.delete("/api/interview/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    return {"deleted": store.drop(session_id)}


@app.get("/api/candidates")
def list_candidates() -> dict[str, Any]:
    return {"candidates": _candidates()}


@app.get("/api/curriculum")
def curriculum() -> dict[str, Any]:
    cur = get_curriculum()
    return {
        "cohort": cur.cohort,
        "modules": cur.modules,
        "days": [
            {
                "day": d.day,
                "title": d.title,
                "type": d.type,
                "module": d.module,
                "tools": list(d.tools),
                "objectives": list(d.objectives),
            }
            for d in sorted(cur.days.values(), key=lambda day: day.day)
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    brain = get_brain()
    return {
        "status": "ok",
        "engine": "llm" if brain.available else "heuristic",
        "detail": brain.status,
        "model": settings.model if brain.available else None,
        "curriculumDays": len(get_curriculum().days),
        "activeSessions": len(store),
        "interview": {
            "minQuestions": settings.min_questions,
            "maxQuestions": settings.max_questions,
            "minDays": settings.min_days,
        },
    }


# ------------------------------------------------------------------ demo UI

_INDEX = STATIC_DIR / "index.html"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> Any:
    if _INDEX.exists():
        return FileResponse(_INDEX)
    return JSONResponse({"service": "AI Cohort Interview Agent", "endpoint": "POST /api/interview"})
