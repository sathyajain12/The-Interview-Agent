# AI Cohort · Interview Agent

Conducts a personalised, multi-turn technical interview against a candidate's
31-day AI cohort record, then returns structured, actionable feedback.

The interview is built from the candidate's *actual* mission history. A topic
passed on the first attempt and a topic passed on the fifth are both "passed",
but they get very different questions — and a skipped topic gets asked about as
a reasoning problem, never as a gotcha.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload      # http://localhost:8000
```

Open <http://localhost:8000> for the demo UI, or post straight to
`POST /api/interview`. **No API key is required to run it** — see
[Two engines](#two-engines).

---

## The contract

Exactly as specified in `docs-technical-spec.md`.

**Start** — the first request initialises the session:

```bash
curl -X POST localhost:8000/api/interview -H 'content-type: application/json' -d '{
  "sessionId": "abc-123",
  "candidate": { "member": {...}, "missions": [...], "signals": {...} }
}'
```

```json
{ "reply": "Hi Sarah — thanks for making the time…", "done": false }
```

**Turn** — every subsequent request carries the candidate's answer:

```bash
curl -X POST localhost:8000/api/interview -H 'content-type: application/json' \
  -d '{ "sessionId": "abc-123", "message": "We used ChromaDB with metadata filters…" }'
```

**End** — when the interview is complete:

```json
{
  "reply": "That's everything I wanted to cover…",
  "done": true,
  "feedback": { "summary": "…", "strengths": [], "gaps": [], "next": [] }
}
```

Responses also carry two **additive, non-contractual** fields — `progress`
(live coverage counters) and `report` (a per-topic scorecard). A client that
only reads `reply`, `done`, and `feedback` is unaffected.

| Endpoint | Purpose |
|---|---|
| `POST /api/interview` | **The contract.** Start and drive an interview. |
| `GET /api/interview/{id}` | Session inspection: plan, transcript, scorecard. |
| `DELETE /api/interview/{id}` | Abandon a session. |
| `GET /api/candidates` · `GET /api/curriculum` | The supplied fixtures. |
| `GET /health` | Which engine is live, and the configured interview shape. |
| `GET /` | Demo UI. |

---

## How it works

```
candidate.json ──▶ profile ──▶ planner ──▶ blueprint (fixed at session start)
                      │                        │
curriculum.json ──▶ retrieval ─────────────────┤
                                               ▼
                                    interviewer (state machine)
                                       │            ▲
                                one call per turn   │ chooses which reply to send
                                       ▼            │
                                 model: evaluation + follow-up + advance
```

**The split that matters.** An LLM left to run an interview on its own will
happily ask eight questions about embeddings and none about deployment. So
responsibility is divided:

- **The planner owns coverage.** Before the first question is asked it builds a
  blueprint — which topics, in what order, at what depth, probing for what.
  This is what makes ">= 8 questions across >= 4 curriculum days" a structural
  guarantee rather than a hope.
- **The engine owns routing.** Follow up or move on, push harder or ease off,
  keep going or wrap up.
- **The model owns voice.** Phrasing, listening, and judging the answer.

**Blueprint composition.** Slots are chosen from the candidate's record and
interleaved so the interview alternates between comfortable and hard ground:

| Slot | Selected from | Asks about |
|---|---|---|
| `warmup` | Capstone, or their strongest topic | What they actually built |
| `strength_depth` | First-try passes | The trade-off underneath — definitions prove nothing here |
| `shaky_probe` | 3+ attempts, or failed | Whether the struggle produced understanding |
| `gap_bridge` | Skipped, failed, or absent from the record | How they'd *approach* it, anchored on a neighbouring topic they did complete |
| `scenario` | An operational topic they shipped | A broken system to debug, not a definition to recite |
| `integration` | Two completed topics in different modules | The seam between modules that nobody teaches |
| `closing` | — | Honest self-assessment |

**One model call per turn.** The call returns the evaluation *and both* possible
replies — the follow-up and the advance. The engine picks. That keeps turn
latency to a single round trip without handing coverage control to the model.

**Adaptation.** Two consecutive strong answers escalate the next topic's depth;
two weak ones simplify it and drop stacked conditions. A rolling momentum
signal is passed into every turn, follow-up depth is capped per topic (one on
the warmup, none on the closing question), and a request to rephrase re-asks
instead of being scored — and doesn't consume one of the interview's questions.

**Retrieval.** Curriculum grounding is a TF-IDF cosine index over the 31 days,
built in-process. At 31 short documents a vector database would be ceremony:
this gives better precision at zero operational cost and stays deterministic,
which matters because the same index feeds both engines. It selects which days'
objectives and tooling enter the prompt, bridges gaps to adjacent topics, and
supplies the rubric each answer is scored against.

---

## Two engines

The endpoint keeps its contract whether or not a model is reachable.

| | With `ANTHROPIC_API_KEY` | Without |
|---|---|---|
| Questions | Generated by `claude-opus-5`, grounded on the blueprint and curated seed angles | Curated per-day question bank (`app/question_bank.py`) |
| Follow-ups | From what the candidate actually said | From the rubric points the answer missed |
| Evaluation | Model, against the rubric | Prefix-matched term overlap + evasion/concreteness markers |
| Feedback | Model-written, cites what they said | Assembled from accumulated evaluations |

The fallback engages on *any* failure — no key, network error, refusal,
malformed output — per call, not per session. Every model call goes through
structured outputs (`output_config.format`), so responses are schema-valid or
they are discarded.

Calibration of the offline scorer across all 31 days: strong answers score 5/5,
hand-waves 0/5, on-topic-but-wrong-day ~1.4/5. It is a genuine fallback, not a
stub — but the model path is meaningfully better at judging *nuance*, which is
what the offline scorer's keyword matching cannot see.

Force it on for a deterministic demo: `INTERVIEW_FORCE_OFFLINE=1`.

---

## Configuration

Everything is environment-driven; see `.env.example`. The defaults matter most:

| Variable | Default | Notes |
|---|---|---|
| `INTERVIEW_MODEL` | `claude-opus-5` | |
| `INTERVIEW_MIN_QUESTIONS` / `INTERVIEW_MIN_DAYS` | `8` / `4` | The spec's floor, enforced by the planner |
| `INTERVIEW_MAX_QUESTIONS` | `14` | Hard stop so an interview always terminates |
| `INTERVIEW_TURN_EFFORT` | `low` | Turns are latency-sensitive; the closing report runs at `medium` |
| `INTERVIEW_SESSION_TTL` | `21600` | Sessions are in-memory with a TTL — long-term history is out of scope |

Requests carry `fallbacks: "default"` so a safety decline is re-served by
another model inside the same call; if the beta flag isn't enabled on the
account, the client notices once and continues without it.

---

## Tests

```bash
python -m pytest tests -q     # 57 tests, no API calls
```

Covering the contract shape, **all 20 supplied candidates driven to completion**
with the coverage floor asserted on each, session isolation, idempotent
re-polling, post-completion requests, planner targeting (e.g. Sarah Johnson's
skipped Day 29 must appear as a gap probe), depth calibration across profiles,
retrieval precision, scorer discrimination, and both model paths — a stubbed
Brain for the happy path and a failing one for the fallback.

---

## Layout

```
app/
  main.py           FastAPI surface — the contract lives here
  interviewer.py    Turn state machine: routing, adaptation, termination
  planner.py        Blueprint construction — owns coverage guarantees
  profile.py        Mission record → interview signal (confidence, gaps, depth)
  curriculum.py     TF-IDF retrieval over the 31 days
  question_bank.py  Curated probes, gap framings, and incident scenarios per day
  prompts.py        System prompt (cached) + per-turn directives
  llm.py            Anthropic transport, structured outputs, no-raise contract
  heuristics.py     Offline engine: scoring, questions, feedback
  store.py          In-memory sessions with TTL and per-session locking
static/index.html   Demo UI (no build step, no CDN)
data/               Supplied curriculum + candidate fixtures
```

---

## Notes and limits

- **Session state is in-process.** Multiple replicas need sticky routing or a
  shared store; `store.py` is the single seam to change. Persistent accounts and
  long-term history were out of scope.
- **The scorecard is diagnostic, not a hiring signal.** Scores exist to steer
  the interview and structure the debrief; they aren't calibrated against
  outcomes and are never shown to the candidate mid-interview.
- **The offline scorer reads keywords, not meaning.** It reliably separates
  substance from evasion and off-topic from on-topic; it cannot tell a correct
  explanation from a confident wrong one. That's what the model path is for.
- **All data here is synthetic**, supplied for the hackathon.
