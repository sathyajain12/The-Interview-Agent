"""Curated probe questions per curriculum day.

Two jobs. Offline (no API key) this bank *is* the interviewer's vocabulary.
Online it is passed to the model as seed angles, which keeps generated
questions anchored to what the cohort actually taught instead of drifting into
generic LLM trivia.

`core` questions verify the candidate did the work. `deep` questions assume
they did and go after the engineering trade-off underneath it.
"""

from __future__ import annotations

DAY_PROBES: dict[int, dict[str, list[str]]] = {
    1: {
        "core": ["Why does each project get its own virtual environment instead of installing packages globally?"],
        "deep": ["Your project runs locally but fails on a teammate's machine with an import error. Walk me through how you'd isolate whether that's an environment problem or a code problem."],
    },
    2: {
        "core": ["What actually changes when you run a model locally through Ollama versus calling a hosted API?"],
        "deep": ["When would you deliberately choose a local model over a hosted one for a production feature, and what do you give up?"],
    },
    3: {
        "core": ["How did you wire the React frontend to the FastAPI backend, and where did requests break first?"],
        "deep": ["Your frontend calls the backend and gets a CORS error in the browser but curl works fine. Explain what's actually happening."],
    },
    4: {
        "core": ["When you loaded the claims CSV into SQLite, what cleaning did the data actually need before it was queryable?"],
        "deep": ["You have a question a user could ask either as SQL or as semantic search. How do you decide which one owns it?"],
    },
    5: {
        "core": ["Extracting text from PDFs versus scanned forms needed different tooling. What was different and why?"],
        "deep": ["OCR output is noisy. How does that noise propagate all the way to a wrong chatbot answer, and where would you catch it?"],
    },
    6: {
        "core": ["How did you decide chunk size and overlap when splitting documents for the knowledge base?"],
        "deep": ["Chunk too small and you lose context; chunk too large and retrieval gets imprecise. How did you actually measure which side you were erring on?"],
    },
    7: {
        "core": ["Explain what an embedding is to someone who has never seen one, and what property makes it useful for search."],
        "deep": ["Two sentences with almost no words in common score high similarity, and two near-identical sentences score low. Explain how both are possible."],
    },
    8: {
        "core": ["You compared Chroma and Pinecone. Which did you pick for the project and what decided it?"],
        "deep": ["What is a vector database actually giving you that a Postgres table with a distance function wouldn't, once you're past a toy dataset?"],
    },
    9: {
        "core": ["What metadata did you attach to each chunk when you indexed it, and what did that metadata let you do later?"],
        "deep": ["How did you verify that every knowledge base chunk actually made it into the index? What would a silent partial-index failure have looked like?"],
    },
    10: {
        "core": ["Your query router decides between SQL, vector search, or both. What signal does it route on?"],
        "deep": ["When you merge results from SQL and vector search, they're scored on completely different scales. How did you rank a combined result set honestly?"],
    },
    11: {
        "core": ["Walk me through one request end to end in your RAG pipeline, from user question to grounded answer."],
        "deep": ["Retrieval returns three chunks and none of them answer the question. What should your pipeline do, and what did yours actually do?"],
    },
    12: {
        "core": ["What's the practical difference between zero-shot, few-shot, and chain-of-thought prompting in the work you did?"],
        "deep": ["You had several system prompt variants. How did you compare them without just eyeballing the outputs and picking a favourite?"],
    },
    13: {
        "core": ["What does function calling actually give you that parsing JSON out of a text response doesn't?"],
        "deep": ["The model calls the right tool with a malformed argument that your Pydantic model rejects. Where does that error go and what does the user see?"],
    },
    14: {
        "core": ["When is fine-tuning the right answer instead of better prompting or better retrieval?"],
        "deep": ["A stakeholder says the chatbot 'doesn't sound right' and asks you to fine-tune. How do you figure out whether fine-tuning is even the right lever?"],
    },
    15: {
        "core": ["What does LoRA change about the training process compared with full fine-tuning?"],
        "deep": ["How would you prove that a fine-tuned model is actually better than the base model rather than just different?"],
    },
    16: {
        "core": ["How did you structure the /chat endpoint, and how does it keep one user's conversation separate from another's?"],
        "deep": ["Your endpoint has to do retrieval, tool calls, and generation before it can respond. How did you keep that from becoming a slow, unobservable black box?"],
    },
    17: {
        "core": ["How did the frontend keep conversation history in sync with the backend across a page refresh?"],
        "deep": ["What state genuinely belongs in the frontend versus the backend in a chat app, and what breaks when you get that split wrong?"],
    },
    18: {
        "core": ["What changes in the backend when you switch from returning a full response to streaming tokens?"],
        "deep": ["A stream drops halfway through. What does the user see, what does your server think happened, and how did you handle it?"],
    },
    19: {
        "core": ["How did you attach citations to answers so a user could actually verify a claim?"],
        "deep": ["The model produces a well-formatted answer with a citation that doesn't support the claim. How would you catch that before the user does?"],
    },
    20: {
        "core": ["How did you keep conversations coherent without sending the entire history on every request?"],
        "deep": ["Summarising old turns to save tokens loses information. How did you decide what was safe to compress and what had to stay verbatim?"],
    },
    21: {
        "core": ["What does the ReAct loop actually do on each iteration, and what makes it stop?"],
        "deep": ["You read the reasoning trace and the agent picked the wrong tool. Is that a tool description problem, a prompt problem, or a model problem, and how do you tell?"],
    },
    22: {
        "core": ["Describe your agent topology - who routes, who specialises, and how work gets handed off."],
        "deep": ["Multi-agent costs you latency and tokens on every request. Where did it measurably beat a single agent, and where was it just architecture for its own sake?"],
    },
    23: {
        "core": ["What problem does MCP solve that a normal REST API or a plain function call doesn't?"],
        "deep": ["You exposed chatbot capabilities as MCP tools. What did you have to change about how those capabilities were described for a client to use them correctly?"],
    },
    24: {
        "core": ["When you swapped mock tools for live MCP calls, what failure modes appeared that weren't there before?"],
        "deep": ["A tool call times out mid-conversation. Walk me through retry, timeout, and what the user sees while that's happening."],
    },
    25: {
        "core": ["How did you build an evaluation set, and what were you actually measuring on each answer?"],
        "deep": ["Retrieval accuracy is high but end-to-end answer quality is poor. What does that tell you about where the problem is?"],
    },
    26: {
        "core": ["Where were the tokens actually going in your pipeline, and what did you cut first?"],
        "deep": ["Caching responses for repeated queries sounds free. When is it actively wrong to serve a cached answer?"],
    },
    27: {
        "core": ["What does prompt injection look like against a RAG chatbot, and where do you defend against it?"],
        "deep": ["Your knowledge base is built from scraped and uploaded documents. Explain how a document could attack the system, and what stops it."],
    },
    28: {
        "core": ["What went into the container image versus what stayed as configuration, and why that split?"],
        "deep": ["Your health check passes but the pod serves errors. What did the health check fail to actually check?"],
    },
    29: {
        "core": ["What did you log on every request, and what question were those logs meant to answer at 3am?"],
        "deep": ["Latency looks fine on average but users complain it's slow. What metric were you looking at wrong?"],
    },
    30: {
        "core": ["What did you test end to end before calling it production-ready, and what did that testing catch?"],
        "deep": ["What's still fragile in what you shipped? Be specific - the honest answer here matters more than the flattering one."],
    },
    31: {
        "core": ["Give me the two-minute version of your capstone: what it does, and what's underneath it."],
        "deep": ["If you had to rebuild the capstone from scratch tomorrow, what architectural decision would you reverse and why?"],
    },
}

# Framing for topics the candidate skipped or failed. These test transferable
# reasoning rather than recall, so a blank is never a dead end.
GAP_PROBES: dict[int, str] = {
    6: "You didn't do the knowledge-base build, but you did work with the data around it. If I handed you a mixed pile of PDFs and database rows, how would you turn that into something retrievable?",
    9: "You skipped populating the vector database. Given what you know about embeddings, what would you actually have to store alongside each vector for retrieval to be useful?",
    11: "You skipped the end-to-end RAG mission. Sketch the pipeline for me anyway - what are the stages between a user's question and a grounded answer?",
    13: "Function calling isn't in your record. Given what you did build, how would you let the model trigger a real database lookup rather than guessing at the answer?",
    14: "You skipped fine-tuning. That's a defensible call - convince me it was. When would you have needed it?",
    15: "You skipped the hands-on fine-tuning. Conceptually, what's the trade-off between LoRA and full fine-tuning?",
    19: "You didn't do response formatting. How would you make an answer verifiable by a user who doesn't trust it?",
    20: "Conversation memory isn't in your record. How would you keep a long conversation coherent without blowing the context window?",
    21: "You skipped the LangChain agents mission but did do multi-agent work. What does the reasoning loop inside a single agent actually look like?",
    23: "MCP isn't in your record. What problem do you think a standard protocol for tool access is trying to solve?",
    25: "You skipped evaluation. How would you know whether a change to your prompt made the chatbot better or worse?",
    26: "Performance and cost optimisation isn't in your record. Where would you look first if your token bill tripled overnight?",
    27: "You skipped security and guardrails. What's the most likely way someone abuses a chatbot that reads from documents you don't fully control?",
    28: "You skipped deployment. Walk me through what has to be true for the thing running on your laptop to run for someone else.",
    29: "Monitoring isn't in your record. Your chatbot is live and a user says it's broken. What do you need to have logged to answer that?",
    30: "You skipped production readiness. What's the gap between 'the demo works' and 'this can take real traffic'?",
}

# Incident-style scenarios. Grounded in a shipped topic, these test whether the
# candidate can debug a system rather than describe one.
SCENARIOS: dict[int, str] = {
    7: "Production scenario: users report search returns confidently wrong documents. Embeddings were generated six months ago; the knowledge base has been updated twice since. Where do you look first?",
    8: "Production scenario: retrieval latency was fine at 10k vectors and is now unacceptable at 2M. Walk me through diagnosing that.",
    10: "Production scenario: a user asks 'how many claims did I file last year' and the bot answers with a paragraph from a policy PDF instead of a number. Where's the bug?",
    11: "Production scenario: the bot confidently invents a coverage limit that appears nowhere in your documents. Walk me through finding the cause.",
    12: "Production scenario: the same prompt that was reliable last month now produces inconsistent formatting after a model version bump. What do you do?",
    16: "Production scenario: /chat p99 latency jumps from 2s to 30s with no code deploy. How do you narrow it down?",
    18: "Production scenario: streaming responses hang for some users and work for others. Where do you start?",
    22: "Production scenario: your router agent keeps delegating billing questions to the clinical specialist. How do you debug agent routing?",
    23: "Production scenario: an MCP tool intermittently returns stale data and the agent presents it as current. What's your fix?",
    27: "Production scenario: a user pastes text that makes the bot ignore its system prompt and reveal its instructions. What failed, and what do you change?",
    28: "Production scenario: the container runs fine locally and crash-loops in Kubernetes. What's your first three checks?",
    29: "Production scenario: an incident happened at 3am and your logs can't tell you which tool call failed. What was missing from your instrumentation?",
    30: "Production scenario: you're on call the week after launch. What's the first alert you'd want to exist?",
}


def probes_for(day: int, difficulty: str) -> list[str]:
    entry = DAY_PROBES.get(day)
    if not entry:
        return []
    if difficulty == "deep":
        return entry.get("deep") or entry.get("core", [])
    return entry.get("core") or entry.get("deep", [])


def scenario_for(day: int) -> str | None:
    return SCENARIOS.get(day)


def gap_probe_for(day: int) -> str | None:
    return GAP_PROBES.get(day)
