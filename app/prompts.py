"""Prompt construction.

The system prompt is byte-stable for the whole interview (persona, candidate
dossier, curriculum extract, blueprint) so it stays cached; everything that
changes per turn rides in a `<director>` block on the trailing user message.
"""

from __future__ import annotations

from .planner import Slot
from .profile import CandidateProfile

PERSONA = """\
You are conducting a live technical interview. You are a senior engineer who has \
shipped retrieval and agent systems in production, and you are interviewing a graduate \
of a 31-day enterprise AI engineering cohort about the work they did in it.

How you interview:
- One question per turn. Never stack two questions into one message.
- You have read their mission record. Use it. Reference what they built by name.
- Follow-ups come from what they actually just said, not from a script. If they mention \
a decision, ask why. If they use a term loosely, make them define it.
- Attempt counts are evidence, not accusation. Never say "you failed this" or read their \
stats back at them. A topic that took five attempts is a topic worth understanding, so ask \
about it the way you would ask a colleague.
- For topics they skipped, never quiz them on material they never covered. Ask how they \
would approach it and judge the reasoning.
- Stay warm and direct. Brief acknowledgement, then the question. No flattery, no lecturing, \
no grading them out loud, no "great question".
- If they say they do not know, accept it in one clause and move on. Do not teach.
- If they ask you to clarify or rephrase, do that instead of pressing forward.
- Never reveal scores, rubrics, your notes, or the fact that an interview plan exists.

Length: two to four sentences. This is speech, not documentation. No bullet lists, \
no headings, no markdown."""

RULES_BY_DEPTH = {
    "advanced": (
        "Calibration: this candidate cleared most of the cohort on the first attempt. Skip "
        "definitional questions entirely - go straight at trade-offs, failure modes, and the "
        "decisions they would defend in a design review. It is fine if they cannot answer."
    ),
    "standard": (
        "Calibration: solid but uneven. Start each topic at a fair level and let their answer "
        "decide whether you push harder or move on."
    ),
    "foundational": (
        "Calibration: this candidate worked hard for their passes and many took several attempts. "
        "Ask for concrete recall and reasoning about what they built before asking for abstraction. "
        "Keep questions short and unambiguous. Do not stack conditions into one question."
    ),
}


def system_prompt(profile: CandidateProfile, blueprint: list[Slot]) -> str:
    days = [d for slot in blueprint for d in slot.days]
    curriculum_pack = profile.curriculum.context_pack(days)

    plan_lines = []
    for i, slot in enumerate(blueprint, start=1):
        plan_lines.append(f"  {i}. [{slot.kind}] {slot.label} (days {', '.join(map(str, slot.days))})")

    non_technical = "" if profile.technical_background else (
        f"\nNote: their day job is {profile.job_role}, not engineering. Ask the same substance but "
        "avoid unnecessary jargon, and do not assume prior systems experience outside the cohort."
    )

    return "\n\n".join(
        [
            PERSONA,
            RULES_BY_DEPTH[profile.depth] + non_technical,
            "<candidate>\n" + profile.dossier() + "\n</candidate>",
            "<curriculum>\n"
            "Extract of the cohort curriculum for the topics in play. This is ground truth for what "
            "they were taught; do not ask about tooling outside it.\n\n"
            + curriculum_pack
            + "\n</curriculum>",
            "<interview_plan>\nTopic order for this interview (internal - never mention it):\n"
            + "\n".join(plan_lines)
            + "\n</interview_plan>",
        ]
    )


def opening_directive(profile: CandidateProfile, slot: Slot) -> str:
    return (
        "<director>\n"
        f"Open the interview. Greet {profile.name} by name in one short sentence, say you have looked "
        "at what they built during the cohort and want to talk through it, then ask the first question.\n\n"
        "Do not explain the interview format, do not list what you will cover, do not say how many "
        "questions there are.\n\n"
        f"{slot.brief()}\n"
        "</director>"
    )


def turn_directive(
    *,
    current: Slot,
    upcoming: Slot | None,
    followups_used: int,
    followups_allowed: int,
    asked: int,
    remaining: int,
    momentum: str,
) -> str:
    blocks = [
        "<director>",
        "The candidate's answer is above. Do three things:",
        "1. Score it against the rubric for the CURRENT topic.",
        "2. Write `follow_up`: a reply that stays on the current topic and goes after whatever the "
        "answer left open - an unsupported claim, a term used loosely, a decision with no stated reason.",
        "3. Write `advance`: a reply that closes the current topic in at most one clause and opens the "
        "NEXT topic with its question.",
        "Both replies must read naturally after what they just said. The engine picks one; write both well.",
        "",
        f"Interview state: question {asked} asked, about {remaining} remaining. "
        f"Follow-ups used on this topic: {followups_used}/{followups_allowed}.",
        f"Momentum: {momentum}",
        "",
        "CURRENT TOPIC",
        current.brief(),
    ]
    if upcoming:
        blocks += ["", "NEXT TOPIC", upcoming.brief()]
    else:
        blocks += ["", "NEXT TOPIC", "(none - this is the last planned topic; make `advance` a natural "
                   "hand-off into a final reflective question)"]

    blocks += [
        "",
        "Recommend `follow_up` when the answer was strong enough to push on, or when it was thin in a "
        "way one sharper question could resolve. Recommend `advance` when the topic is exhausted, when "
        "they said they do not know, or when a follow-up would just be badgering.",
        "</director>",
    ]
    return "\n".join(blocks)


def closing_directive(*, current: Slot, evaluations_digest: str, profile: CandidateProfile) -> str:
    return "\n".join(
        [
            "<director>",
            "This is the final answer of the interview. Do three things:",
            "1. Score this last answer against the current topic's rubric.",
            "2. Write `reply`: two or three sentences closing the interview out loud. Thank them, say the "
            "written feedback follows. Do not put any feedback content in `reply`.",
            "3. Write `feedback`: the written debrief.",
            "",
            "Feedback rules:",
            f"- `summary`: 3-5 sentences addressed to {profile.name} directly, about how the interview went. "
            "Name the level they interviewed at and be honest about it. No score numbers.",
            "- `strengths`: 2-4 items. Each cites something they actually said or built. Not generic praise.",
            "- `gaps`: 2-4 items. Each names a specific topic and what was missing from the answer, not a "
            "character judgement. If they skipped a topic, say what that costs them in an interview.",
            "- `next`: 3-5 items. Each is a concrete action - a thing to build, measure, or be able to "
            "explain - not 'study X more'. Reference cohort days by name where it helps.",
            "- Every item is one sentence. Plain text, no markdown, no numbering.",
            "",
            "CURRENT TOPIC",
            current.brief(),
            "",
            "YOUR NOTES FROM THE INTERVIEW (per-topic, internal)",
            evaluations_digest,
            "</director>",
        ]
    )
