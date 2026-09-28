"""Module 5 — simulate production traffic so we have live traces to evaluate.

Online evals run on whatever your app actually produces in production. To make
that concrete without a real deployment, this script runs the HR agent over a
batch of *messy, realistic* user questions — typos, ambiguity, and out-of-scope
asks you'd never put in a curated dataset — with **tracing enabled**, so each run
lands as a trace in a LangSmith project.

Once these traces exist, `score_traces.py` evaluates them with the reference-free
evaluators (the online-eval loop), and Module 6 pushes a sample to an annotation
queue for human review.

Every run carries a ``thread_id`` — single questions are one-turn threads, and a
few are real multi-turn conversations. That is what LangSmith's **Trajectory
view** groups on: open a thread and you read the whole session (user messages,
model replies, tool calls and results, once each, in order) instead of digging
through nested runs. Click any step to drop into its underlying trace.

Run:  python module_5_online_evals/production_traffic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import require_langsmith
from hr_agent import run_conversation
from module_5_online_evals.project import PRODUCTION_PROJECT

# Realistic production traffic: a mix of clean policy questions, typo'd and
# colloquial phrasings, an ambiguous one, and an out-of-scope ask. This is the
# long tail a fixed dataset never fully covers — exactly what online evals are for.
PRODUCTION_QUERIES = [
    "how many vacation days do i get?",
    "wen does my helth insurance kick in",  # typos
    "whats the 401k match",
    "can i wfh on mondays?",
    "do i get paid sick days and how many",
    "how long do i have to submit expenses, need receipts?",
    "how much parental leave is there",
    "what's the deal with benefits",  # vague / ambiguous
    "can you book my flight to the offsite?",  # out of scope
    "is the match vested right away or do i have to wait",
]


# Multi-turn sessions. Each list is ONE thread: same thread_id, one root run per
# turn. The follow-ups lean on earlier turns ("them", "it") and one asks something
# the policy text doesn't cover — the shape of real conversations, and the case
# where reading a single run in isolation tells you almost nothing.
PRODUCTION_CONVERSATIONS = [
    [
        "hi, im helping Jordan Lee get set up before day one",
        "create their email and slack accounts",
        "and order a laptop and a headset for them",
    ],
    [
        "how many vacation days do i get?",
        "when can i actually use them",
        "and do unused ones carry over to next year?",  # not in the policy text
    ],
]


def generate_traffic(
    queries: list[str] | None = None,
    conversations: list[list[str]] | None = None,
) -> str:
    """Run the agent over each query and conversation (traced); return the project name.

    Every query becomes a thread — a one-turn thread if it's a lone question — so
    all of it shows up in the Trajectory view, not only the multi-turn sessions.

    Force tracing on and route these runs to the dedicated production project,
    overriding any ambient ``LANGSMITH_PROJECT`` from your ``.env`` (which likely
    points at the offline experiment project). The tracer reads these env vars
    when each run is logged, so setting them here — before we invoke — is enough.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = PRODUCTION_PROJECT
    queries = PRODUCTION_QUERIES if queries is None else queries
    conversations = PRODUCTION_CONVERSATIONS if conversations is None else conversations
    threads = [[q] for q in queries] + conversations
    turns = sum(len(t) for t in threads)
    print(f"Sending {turns} turns in {len(threads)} threads to project "
          f"'{PRODUCTION_PROJECT}' (traced)...\n")
    for i, thread in enumerate(threads, 1):
        results = run_conversation(thread)
        for turn, result in zip(thread, results):
            answer = result["messages"][-1].content
            if isinstance(answer, list):  # some providers return content blocks
                answer = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
            preview = (answer or "").strip().replace("\n", " ")[:80]
            print(f"  [{i:>2}] {turn[:45]:<45} -> {preview}...")
    return PRODUCTION_PROJECT


def main() -> None:
    require_langsmith()
    project = generate_traffic()
    threads = len(PRODUCTION_QUERIES) + len(PRODUCTION_CONVERSATIONS)
    print(f"\nDone. {threads} threads sent to project '{project}'.")
    print("Open it in LangSmith → Tracing Projects → Threads, and open one of the "
          "3-turn threads in the Trajectory view.")
    print("Then run:")
    print("  python module_5_online_evals/score_traces.py")


if __name__ == "__main__":
    main()
