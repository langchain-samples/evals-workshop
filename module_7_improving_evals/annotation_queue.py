"""Module 7 — push production traces to an annotation queue for human review.

Few-shot alignment (few_shot_judge.py) needs human-labeled examples. Where do
those labels come from? An **annotation queue**: a review inbox in LangSmith where
a human opens each trace, reads it, and records a verdict. Those verdicts are the
ground truth you align the judge to — and the examples you feed it few-shot.

You don't queue *everything* (humans don't scale). You queue what's worth a human's
attention: the traces your cheap online evals already flagged. This script pulls
recent production traces, uses the deterministic reference-free checks from Module
5 to prioritize the suspicious ones (deflected or empty answers), and adds them to
a queue. A reviewer then labels them in the UI.

Run:  python module_7_improving_evals/annotation_queue.py [--limit 20] [--project NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import require_langsmith
from module_6_online_evals.project import PRODUCTION_PROJECT
from module_6_online_evals.reference_free_evals import DETERMINISTIC_ONLINE_EVALUATORS
from module_6_online_evals.score_traces import _answer_from_run, _question_from_run

QUEUE_NAME = "HR Agent — review for labeling"


def ensure_queue(client: Client, name: str = QUEUE_NAME) -> str:
    """Create the annotation queue if it doesn't exist; return its id (idempotent)."""
    existing = list(client.list_annotation_queues(name=name, limit=1))
    if existing:
        return str(existing[0].id)
    queue = client.create_annotation_queue(
        name=name,
        description="Human review of HR-agent production traces. Verdicts become "
                    "the labels that align our LLM judges (see judge_alignment.py).",
    )
    return str(queue.id)


def _flagged_by_feedback(client: Client, runs: list) -> list:
    """Runs that already carry a low online-eval score (written by score_traces.py).

    This is the production-true selection: an online eval (a rule) scores traffic,
    and the low-scoring traces are exactly the ones worth a human's review. One
    `list_feedback` call covers every run.
    """
    low: set = set()
    feedbacks = client.list_feedback(run_ids=[r.id for r in runs])
    for fb in feedbacks:
        if fb.score is not None and fb.score < 1:
            low.add(str(fb.run_id))
    return [r for r in runs if str(r.id) in low]


def _flagged_by_checks(runs: list) -> list:
    """Fallback selection if no online-eval feedback exists yet: re-run the cheap
    deterministic reference-free checks and flag anything they catch."""
    flagged = []
    for run in runs:
        inputs = {"question": _question_from_run(run)}
        outputs = {"answer": _answer_from_run(run)}
        if any(ev(inputs, outputs)["score"] < 1 for ev in DETERMINISTIC_ONLINE_EVALUATORS):
            flagged.append(run)
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue production traces for human annotation.")
    parser.add_argument("--limit", type=int, default=20, help="How many recent traces to consider.")
    parser.add_argument("--project", default=PRODUCTION_PROJECT)
    parser.add_argument("--all", action="store_true",
                        help="Queue all recent traces, not just the flagged ones.")
    args = parser.parse_args()

    require_langsmith()
    client = Client()

    runs = list(client.list_runs(project_name=args.project, is_root=True, limit=args.limit))
    if not runs:
        raise SystemExit(
            f"No traces in project '{args.project}'. Run "
            "module_6_online_evals/production_traffic.py first."
        )

    if args.all:
        selected, basis = runs, "all recent"
    else:
        selected = _flagged_by_feedback(client, runs)
        basis = "flagged by online evals"
        if not selected:
            selected = _flagged_by_checks(runs)
            basis = "flagged by deterministic checks"
    if not selected:
        print("Nothing flagged. Either run score_traces.py first so the online evals "
              "can flag traces, or re-run here with --all to queue recent traces anyway.")
        return

    queue_id = ensure_queue(client)
    client.add_runs_to_annotation_queue(queue_id, run_ids=[r.id for r in selected])

    print(f"Queued {len(selected)} of {len(runs)} trace(s) ({basis}) to '{QUEUE_NAME}'.")
    print("\nOpen LangSmith → Annotation Queues → that queue, and label each trace.")
    print("Your verdicts become the human labels that:")
    print("  - measure judge agreement, and")
    print("  - serve as few-shot examples to align the judge (see judge_alignment.py).")


if __name__ == "__main__":
    main()
