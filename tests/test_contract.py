"""The API contract from technical-spec.md, verified end to end."""

from __future__ import annotations

import uuid

import pytest

from app.config import settings

ANSWER = (
    "We used ChromaDB for semantic retrieval over chunked plan documents, with a router "
    "in front that sends numeric questions to SQL and everything else to vector search. "
    "The tricky part was merging both result sets onto one comparable score."
)


def _run(client, candidate, *, answer=ANSWER, limit=40):
    session_id = str(uuid.uuid4())
    first = client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})
    assert first.status_code == 200
    body = first.json()
    assert body["reply"].strip()
    assert body["done"] is False
    assert "feedback" not in body

    turns = 0
    while not body["done"] and turns < limit:
        turns += 1
        response = client.post("/api/interview", json={"sessionId": session_id, "message": answer})
        assert response.status_code == 200
        body = response.json()
    return session_id, body, turns


def test_start_shape(client, candidates):
    response = client.post(
        "/api/interview", json={"sessionId": "shape-test", "candidate": candidates[0]}
    )
    body = response.json()
    assert set(body) >= {"reply", "done"}
    assert isinstance(body["reply"], str) and body["reply"]
    assert body["done"] is False


def test_full_interview_terminates_with_structured_feedback(client, candidates):
    _, body, turns = _run(client, candidates[0])

    assert body["done"] is True, "interview never terminated"
    assert turns < 40

    feedback = body["feedback"]
    assert isinstance(feedback["summary"], str) and feedback["summary"].strip()
    for key in ("strengths", "gaps", "next"):
        assert isinstance(feedback[key], list)
        assert feedback[key], f"{key} must not be empty"
        assert all(isinstance(item, str) and item.strip() for item in feedback[key])


@pytest.mark.parametrize("index", range(20))
def test_minimum_coverage_for_every_candidate(client, candidates, index):
    """>= 8 questions across >= 4 curriculum days, for all 20 profiles."""
    _, body, _ = _run(client, candidates[index])

    progress = body["progress"]
    assert body["done"] is True
    assert progress["questionsAsked"] >= settings.min_questions
    assert len(progress["daysCovered"]) >= settings.min_days
    assert body["report"]["questionsAnswered"] >= settings.min_questions - 1


def test_unknown_session_without_candidate_is_404(client):
    response = client.post("/api/interview", json={"sessionId": "nope-" + str(uuid.uuid4()), "message": "hi"})
    assert response.status_code == 404


def test_sessions_are_isolated(client, candidates):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    client.post("/api/interview", json={"sessionId": a, "candidate": candidates[0]})
    client.post("/api/interview", json={"sessionId": b, "candidate": candidates[2]})

    client.post("/api/interview", json={"sessionId": a, "message": ANSWER})
    state_a = client.get(f"/api/interview/{a}").json()
    state_b = client.get(f"/api/interview/{b}").json()

    assert state_a["report"]["candidateId"] != state_b["report"]["candidateId"]
    assert state_a["progress"]["questionsAsked"] > state_b["progress"]["questionsAsked"]


def test_repoll_without_message_is_idempotent(client, candidates):
    session_id = str(uuid.uuid4())
    first = client.post("/api/interview", json={"sessionId": session_id, "candidate": candidates[1]}).json()
    again = client.post("/api/interview", json={"sessionId": session_id}).json()
    assert again["reply"] == first["reply"]
    assert again["progress"]["questionsAsked"] == first["progress"]["questionsAsked"]


def test_answers_after_completion_do_not_break(client, candidates):
    session_id, body, _ = _run(client, candidates[4])
    extra = client.post("/api/interview", json={"sessionId": session_id, "message": "one more thing"})
    assert extra.status_code == 200
    assert extra.json()["done"] is True
    assert extra.json()["feedback"]["summary"] == body["feedback"]["summary"]


def test_health_and_fixtures(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["curriculumDays"] == 31

    assert len(client.get("/api/candidates").json()["candidates"]) == 20
    assert len(client.get("/api/curriculum").json()["days"]) == 31
