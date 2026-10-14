"""Module 8 — a production eval monitor that alerts on quality drift.

Module 5 gated a *pull request*: run the offline suite, fail the build if a
curated metric regresses. That protects you from bad merges. It does nothing about
the agent quietly degrading in production — a provider model update, a creeping
prompt change, a shift in what users ask. For that you need a gate on *production
behavior over time*.

This monitor is that gate. On a schedule (see online-evals.yml) it:
  1. pulls the last N production traces,
  2. scores them with the reference-free evaluators from Module 6,
  3. compares each metric's mean to a committed baseline (baseline.json), and
  4. exits non-zero — failing the scheduled job and triggering an alert — if any
     metric has drifted more than `tolerance` below baseline.

It deliberately reuses Module 6's evaluators and trace extractors: the same
reference-free checks that score a single trace online, aggregated and gated here.

Run:
    python module_8_production_ci/monitor.py                 # check vs baseline
    python module_8_production_ci/monitor.py --deterministic-only
    python module_8_production_ci/monitor.py --write-baseline # recapture baseline
"""

from __future__ import annotations

import argparse
import json
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
from module_6_online_evals.score_traces import _answer_from_run, _question_from_run

BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


def load_baseline() -> dict:
    """Read the committed baseline (plain JSON — no pickle/deserialization)."""
    with open(BASELINE_PATH) as f:
        return json.load(f)


def aggregate(runs, evaluators) -> dict[str, float]:
    """Mean score per metric across the given traces."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        inputs = {"question": _question_from_run(run)}
        outputs = {"answer": _answer_from_run(run)}
        for evaluator in evaluators:
            res = evaluator(inputs, outputs)
            sums[res["key"]] += float(res["score"])
            counts[res["key"]] += 1
    return {k: sums[k] / counts[k] for k in sums}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor production eval quality vs a baseline.")
    parser.add_argument("--limit", type=int, default=50, help="How many recent traces to score.")
    parser.add_argument("--project", default=PRODUCTION_PROJECT)
    parser.add_argument("--deterministic-only", action="store_true",
                        help="Skip the LLM judges (no model key / no model cost).")
    parser.add_argument("--write-baseline", action="store_true",
                        help="Recapture the current means as the new baseline.json.")
    args = parser.parse_args()

    require_langsmith()
    client = Client()

    evaluators = DETERMINISTIC_ONLINE_EVALUATORS if args.deterministic_only else ONLINE_EVALUATORS
    runs = list(client.list_runs(project_name=args.project, is_root=True, limit=args.limit))
    if not runs:
        raise SystemExit(
            f"No traces in project '{args.project}'. Generate some with "
            "module_6_online_evals/production_traffic.py."
        )

    means = aggregate(runs, evaluators)

    if args.write_baseline:
        baseline = load_baseline()
        baseline["metrics"].update({k: round(v, 3) for k, v in means.items()})
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
            f.write("\n")
        print(f"Baseline updated from {len(runs)} trace(s): {baseline['metrics']}")
        return

    baseline = load_baseline()
    tolerance = float(baseline.get("tolerance", 0.1))
    expected = baseline["metrics"]

    print(f"=== Production monitor: '{args.project}' ({len(runs)} traces, "
          f"tolerance {tolerance:.2f}) ===")
    alerts: list[str] = []
    for metric in sorted(means):
        mean = means[metric]
        base = expected.get(metric)
        if base is None:
            print(f"  [ -- ] {metric:<22} mean={mean:.3f}  (no baseline; skipped)")
            continue
        drifted = mean < base - tolerance
        flag = "DRIFT" if drifted else "ok"
        print(f"  [{flag:>5}] {metric:<22} mean={mean:.3f}  baseline={base:.3f}")
        if drifted:
            alerts.append(f"{metric} ({mean:.3f} < {base:.3f} - {tolerance:.2f})")

    if alerts:
        print(f"\n🚨 QUALITY DRIFT — {len(alerts)} metric(s) below baseline:")
        for a in alerts:
            print(f"  - {a}")
        print("\nThe non-zero exit fails the scheduled job; wire it to your alerting "
              "(Slack/PagerDuty/email) in online-evals.yml.")
        sys.exit(1)

    print("\nNo drift — production quality is within tolerance of baseline.")
    sys.exit(0)


if __name__ == "__main__":
    main()
