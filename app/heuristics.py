"""Offline interview engine.

This runs whenever the model is unavailable - no API key, network failure, a
refusal, a malformed response. It is deliberately not a stub: it scores answers
against the same rubric the model uses and asks questions from the curated bank,
so the endpoint keeps its contract (>= 8 questions, >= 4 days, structured
feedback) with nothing but the curriculum and the candidate record.
"""

from __future__ import annotations

import random
import re
from typing import Any

from . import question_bank as qb
from .curriculum import _tokens
from .planner import Slot
from .profile import CandidateProfile

_DONT_KNOW = re.compile(
    r"\b(i (don'?t|do not) know|no idea|not sure|never (did|got to|used|touched)|"
    r"i skipped|didn'?t (do|cover|get to)|can'?t remember|drawing a blank|pass\b)",
    re.IGNORECASE,
)
_CLARIFY = re.compile(
    r"\b(what do you mean|can you (repeat|rephrase|clarify)|could you clarify|"
    r"sorry,? (what|come again)|not sure what you'?re asking)",
    re.IGNORECASE,
)
_VAGUE = re.compile(
    r"\b(basically|kind of|sort of|stuff|things|whatever|somehow|magic|just works|etc\.?)\b",
    re.IGNORECASE,
)
_CONCRETE = re.compile(
    r"(\d+\s*(ms|s|k|m|gb|mb|tokens?|chunks?|dimensions?|%)|"
    r"\b(chromadb|chroma|pinecone|fastapi|langchain|langgraph|crewai|pydantic|docker|kubernetes|"
    r"sqlite|streamlit|ollama|prometheus|grafana|lora|qlora|mcp|react|sse|cosine|pca|tf-idf|"
    r"embedding|rerank|top-?k|chunk)\b)",
    re.IGNORECASE,
)

_ACK_HIGH = [
    "That tracks.",
    "Good - that's the answer I was hoping for.",
    "Right, and you've clearly done that yourself.",
]
_ACK_MID = ["Okay.", "Fair enough.", "Got it."]
_ACK_LOW = ["Alright.", "Okay, noted.", "Understood."]
_ACK_UNKNOWN = [
    "That's a fine answer - better than guessing.",
    "No problem, that's honest.",
]
_TRANSITIONS = [
    "Let me ask about something else.",
    "Different topic.",
    "Switching tracks.",
    "Moving on.",
]

# Term overlap saturates fast, so normalise against a capped vocabulary rather
# than the full rubric - long objectives would otherwise make every answer look
# incomplete.
_TERM_CAP = 24
_SATURATION = 0.28


_PREFIX = 5


def _rubric_terms(slot: Slot) -> set[str]:
    blob = " ".join(slot.rubric) + " " + slot.label
    return set(_tokens(blob))


def _matches(terms: set[str], answer_terms: set[str]) -> set[str]:
    """Overlap with prefix matching, so 'similarity' counts against 'similar'.

    Stemming alone is not enough here: curriculum objectives are written in
    noun form and candidates answer in verb form.
    """
    hits: set[str] = set()
    short = {a for a in answer_terms if len(a) < _PREFIX}
    prefixes = {a[:_PREFIX] for a in answer_terms if len(a) >= _PREFIX}
    for term in terms:
        if term in answer_terms:
            hits.add(term)
        elif len(term) >= _PREFIX and term[:_PREFIX] in prefixes:
            hits.add(term)
        elif term in short:
            hits.add(term)
    return hits


def evaluate(answer: str, slot: Slot) -> dict[str, Any]:
    """Score an answer against the slot rubric without a model.

    Keyword overlap can only ever approximate comprehension, so this is
    calibrated to be generous about substance and strict about evasion: a
    detailed on-topic answer lands 4-5, a hand-wave lands 1-2, and a bluff is
    caught by the vagueness markers rather than by the term count.
    """
    text = (answer or "").strip()
    words = text.split()
    flags: list[str] = []

    if _CLARIFY.search(text) and len(words) < 30:
        return {
            "score": 0,
            "covered": [],
            "missing": list(slot.rubric),
            "flags": ["asked_for_clarification"],
            "signal": "Asked for the question to be rephrased rather than answering.",
        }

    if not text:
        return {
            "score": 0,
            "covered": [],
            "missing": list(slot.rubric),
            "flags": ["dont_know"],
            "signal": "No answer given.",
        }

    if _DONT_KNOW.search(text) and len(words) < 45:
        return {
            "score": 1 if len(words) > 12 else 0,
            "covered": [],
            "missing": list(slot.rubric),
            "flags": ["dont_know"],
            "signal": "Said they did not know rather than bluffing.",
        }

    terms = _rubric_terms(slot)
    answer_terms = set(_tokens(text))
    hit = _matches(terms, answer_terms)
    relevance = len(hit) / max(min(len(terms), _TERM_CAP), 1)

    covered = [point for point in slot.rubric if len(_matches(set(_tokens(point)), answer_terms)) >= 2]
    missing = [point for point in slot.rubric if point not in covered]

    # 1 point for engaging at all, up to 2.5 for on-topic substance.
    score = 1.0 if len(words) >= 18 else 0.0
    score += 2.5 * min(relevance / _SATURATION, 1.0)

    if _CONCRETE.search(text):
        score += 1.0
        flags.append("concrete_example")
    if len(words) >= 55:
        score += 0.5
    if len(covered) >= 2:
        score += 0.5

    if _VAGUE.search(text):
        score -= 1.25
        flags.append("vague")
    if len(words) < 15:
        score -= 1.25
        flags.append("vague")
    if relevance < 0.08 and len(words) > 30:
        score -= 1.0
        flags.append("off_topic")

    final = max(0, min(5, round(score)))
    return {
        "score": final,
        "covered": covered[:4],
        "missing": missing[:4],
        "flags": sorted(set(flags)),
        "signal": (
            f"Offline scoring: {len(hit)} of {min(len(terms), _TERM_CAP)} topic terms present "
            f"in a {len(words)}-word answer."
        ),
    }


def _ack(evaluation: dict[str, Any]) -> str:
    if "asked_for_clarification" in evaluation["flags"]:
        return ""
    if "dont_know" in evaluation["flags"]:
        return random.choice(_ACK_UNKNOWN)
    score = evaluation["score"]
    if score >= 4:
        return random.choice(_ACK_HIGH)
    if score >= 2:
        return random.choice(_ACK_MID)
    return random.choice(_ACK_LOW)


def question_for(slot: Slot, *, deep: bool = False) -> str:
    seeds = [s for s in slot.seeds if s]
    if seeds:
        return seeds[-1] if deep and len(seeds) > 1 else seeds[0]
    bank = qb.probes_for(slot.primary_day, "deep" if deep else "core")
    if bank:
        return bank[0]
    if slot.rubric:
        return f"Talk me through {slot.label.lower()} - specifically, {slot.rubric[0].lower()}."
    return f"Tell me about {slot.label} - what did you build and what decisions did you make?"


def opening(profile: CandidateProfile, slot: Slot) -> str:
    return (
        f"Hi {profile.name.split()[0]} - thanks for making the time. I've had a look at what you "
        f"built during the cohort and I'd like to talk through some of it rather than quiz you. "
        f"{question_for(slot)}"
    )


def follow_up(slot: Slot, evaluation: dict[str, Any]) -> str:
    ack = _ack(evaluation)
    missing = evaluation.get("missing") or []
    if evaluation["score"] >= 4:
        probe = qb.probes_for(slot.primary_day, "deep")
        tail = probe[0] if probe else f"Push one level down on that - what would you change if the scale went up ten times?"
    elif missing:
        point = missing[0]
        tail = (
            f"Stay with that for a second - one piece you didn't touch was "
            f"{point[0].lower() + point[1:]}. Walk me through how you handled that."
        )
    else:
        tail = "Can you make that concrete with something you actually ran into while building it?"
    return f"{ack} {tail}".strip()


def advance(next_slot: Slot | None, evaluation: dict[str, Any], *, deep: bool = False) -> str:
    ack = _ack(evaluation)
    if next_slot is None:
        return f"{ack} That's everything I wanted to cover.".strip()
    transition = random.choice(_TRANSITIONS)
    return f"{ack} {transition} {question_for(next_slot, deep=deep)}".strip()


def clarify(slot: Slot) -> str:
    return (
        "Sure, let me put it another way. "
        + question_for(slot)
    )


def closing_reply(profile: CandidateProfile) -> str:
    return (
        f"That's everything I wanted to cover - thanks, {profile.name.split()[0]}. "
        "I'll write up where you came across strongly and where I'd push you next."
    )


def feedback(profile: CandidateProfile, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the structured debrief from accumulated evaluations."""
    scored = [r for r in records if r.get("evaluation")]
    scores = [r["evaluation"]["score"] for r in scored]
    avg = sum(scores) / len(scores) if scores else 0.0

    # One entry per topic, keeping that topic's best and worst showing - a
    # follow-up is the same topic, not a second data point to list twice.
    best_by_topic: dict[str, dict[str, Any]] = {}
    worst_by_topic: dict[str, dict[str, Any]] = {}
    for r in scored:
        label, score = r["label"], r["evaluation"]["score"]
        if label not in best_by_topic or score > best_by_topic[label]["evaluation"]["score"]:
            best_by_topic[label] = r
        if label not in worst_by_topic or score < worst_by_topic[label]["evaluation"]["score"]:
            worst_by_topic[label] = r

    best = sorted(
        (r for r in best_by_topic.values() if r["evaluation"]["score"] >= 3),
        key=lambda r: -r["evaluation"]["score"],
    )[:3]
    worst = sorted(
        (r for r in worst_by_topic.values() if r["evaluation"]["score"] <= 2),
        key=lambda r: r["evaluation"]["score"],
    )[:3]

    band = (
        "strong across the board" if avg >= 4
        else "solid with real depth in places" if avg >= 3
        else "uneven - the foundations are there but the depth is not yet" if avg >= 2
        else "early; you can describe the work but not yet defend the decisions behind it"
    )

    closing_note = (
        " The questions that went best were the ones where you talked about something you actually "
        "built; the ones that went worst were where the answer stayed at the level of a definition."
        if worst
        else " You answered with specifics throughout, which is the difference between having built "
        "something and having read about it."
    )
    summary = (
        f"{profile.name}, across {len(scored)} questions spanning "
        f"{len({d for r in scored for d in r['days']})} days of the cohort, this interview came out {band}. "
        f"Your record shows a {profile.first_try_rate:.0%} first-try pass rate over "
        f"{profile.missions_completed} missions and {profile.commit_days} active days."
        + closing_note
    )

    strengths = [
        f"{r['label']}: answered with enough specificity to show you built it, not just read about it."
        for r in best
    ] or ["You engaged with every question rather than deflecting, which counts for more than it sounds."]

    gaps = []
    for r in worst:
        missing = (r["evaluation"].get("missing") or [None])[0]
        if missing:
            gaps.append(f"{r['label']}: the answer never reached {missing.lower()}.")
        else:
            gaps.append(f"{r['label']}: the answer stayed general where the question wanted specifics.")
    for sig in profile.gaps[:2]:
        gaps.append(
            f"{sig.title} (Day {sig.day}) is missing from your record entirely - expect an interviewer "
            "to go straight at it."
        )
    gaps = gaps[:4] or ["No significant gaps surfaced in this interview."]

    nxt: list[str] = []
    for r in worst[:2]:
        nxt.append(
            f"Rebuild the {r['label'].lower()} piece end to end and write down the decision you made at "
            "each step, so the reasoning is available under pressure."
        )
    for sig in profile.gaps[:2]:
        nxt.append(
            f"Work through Day {sig.day} ({sig.title}) - even a small version gives you something concrete "
            "to talk about instead of a hypothetical."
        )
    nxt.append(
        "Practise answering 'why did you choose that' for every tool in your capstone; being able to name "
        "the alternative you rejected is what separates a builder from a follower."
    )
    if profile.first_try_rate < 0.4:
        nxt.append(
            "Pick your two lowest-confidence topics and explain each one out loud to someone else; the "
            "gaps surface fast when you cannot lean on notes."
        )

    return {
        "summary": summary,
        "strengths": strengths[:4],
        "gaps": gaps,
        "next": nxt[:5],
    }
