"""Module 6 — the online-eval loop: score live traces and write feedback back.

This is what an online eval *is*, mechanically: pull recent production traces,
run reference-free evaluators over each, and attach the scores to the trace as
feedback. In LangSmith you'd typically run this continuously as a server-side
**rule** (auto-scores a sample of incoming traces — see the README), but the loop
below is the portable, inspectable version of the same idea, and it's exactly
what Module 8's scheduled monitor reuses.

Compared to an offline `client.evaluate(...)` experiment:
  - the data is live traces, not a fixed dataset;
  - the evaluators take only (inputs, outputs) — no reference (see reference_free_evals.py);
  - results are written as `create_feedback` on each run, not collected into one experiment.

Run:  python module_6_online_evals/score_traces.py [--limit 20] [--deterministic-only]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import require_langsmith
from module_6_online_evals.project import PRODUCTION_PROJECT
from module_6_online_evals.reference_free_evals import (
    DETERMINISTIC_ONLINE_EVALUATORS,
    ONLINE_EVALUATORS,
)


def _question_from_run(run) -> str:
    """Pull the user's question out of a (serialized) agent run's inputs."""
    messages = (run.inputs or {}).get("messages", [])
    # Inputs were {"messages": [("user", q)]}; serialized forms vary, so be lenient.
    for msg in messages:
        if isinstance(msg, (list, tuple)) and len(msg) == 2:
            return str(msg[1])
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
    return ""


def _answer_from_run(run) -> str:
    """Pull the agent's final answer text out of a (serialized) run's outputs."""
    messages = (run.outputs or {}).get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    if isinstance(content, list):  # content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Score recent production traces (online eval).")
    parser.add_argument("--limit", type=int, default=20, help="How many recent root traces to score.")
    parser.add_argument("--deterministic-only", action="store_true",
                        help="Skip the LLM judges (no model key / no model cost).")
    parser.add_argument("--project", default=PRODUCTION_PROJECT)
    args = parser.parse_args()

    require_langsmith()
    client = Client()

    evaluators = DETERMINISTIC_ONLINE_EVALUATORS if args.deterministic_only else ONLINE_EVALUATORS
    runs = list(client.list_runs(project_name=args.project, is_root=True, limit=args.limit))
    if not runs:
        raise SystemExit(
            f"No traces found in project '{args.project}'. Run "
            "production_traffic.py first (or pass --project)."
        )

    print(f"Scoring {len(runs)} trace(s) from '{args.project}' with "
          f"{len(evaluators)} reference-free evaluator(s)...\n")

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        inputs = {"question": _question_from_run(run)}
        outputs = {"answer": _answer_from_run(run)}
        for evaluator in evaluators:
            res = evaluator(inputs, outputs)
            # Attach the score to the trace so it shows up alongside it in LangSmith.
            client.create_feedback(
                run.id, key=res["key"], score=res["score"], comment=res.get("comment"),
            )
            sums[res["key"]] += float(res["score"])
            counts[res["key"]] += 1

    print("=== Online eval aggregate (mean score per metric) ===")
    for metric in sorted(sums):
        print(f"  {metric:<22} {sums[metric] / counts[metric]:.3f}  (n={counts[metric]})")
    print(f"\nFeedback written to {len(runs)} trace(s) in '{args.project}'. "
          "Open the project in LangSmith to see scores on each trace.")


if __name__ == "__main__":
    main()
