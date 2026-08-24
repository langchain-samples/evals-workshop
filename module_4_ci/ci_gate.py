"""Module 4 — the CI gate.

Run an experiment over a dataset, aggregate the scores, and EXIT NON-ZERO if
any metric falls below its threshold. That non-zero exit is what fails a CI
build and blocks a regression from merging.

This is the "aggregate gate" pattern: one pass/fail signal for the whole
dataset per metric. (The pytest approach in test_evals.py is the
"per-example gate" alternative — see the README for when to use which.)

By default the gate runs on every example EXCEPT the held-out (scratch) splits,
so an example nobody assigned a split to still gets gated. See resolve_data.

Usage:
    python module_4_ci/ci_gate.py --suite single_turn
    python module_4_ci/ci_gate.py --suite agent
    python module_4_ci/ci_gate.py --suite tool_failures
    python module_4_ci/ci_gate.py --suite single_turn --require-splits
    python module_4_ci/ci_gate.py --suite single_turn --held-out scratch --held-out wip
    python module_4_ci/ci_gate.py --suite single_turn --split gate   # opt-in, narrower
    python module_4_ci/ci_gate.py --suite single_turn --threshold 0.9
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith

# Per-metric pass thresholds. Tune these to your risk tolerance. Safety-style
# metrics (no_forbidden_tools) demand a perfect score; fuzzy quality metrics
# (professional_tone) get more headroom.
THRESHOLDS: dict[str, float] = {
    # single-turn
    "response_not_empty": 1.0,
    "mentions_required_facts": 0.8,
    "no_unsupported_topic": 1.0,
    "correctness": 0.8,
    "groundedness": 0.9,
    "professional_tone": 0.6,
    # agent
    "trajectory_exact_match": 0.5,
    "required_tools_used": 0.9,
    "no_forbidden_tools": 1.0,
    "trajectory_efficiency": 0.7,
    "correct_employee_id": 1.0,
    "tool_args_well_formed": 1.0,
    "trajectory_is_reasonable": 0.8,
    # tool failures (mocked)
    "reports_tool_failure": 1.0,
}

SUITES = ("single_turn", "agent", "tool_failures")

# Splits deliberately kept OUT of the gate — scratch space for tuning prompts
# and few-shot judges. Tune against the gate and the gate stops measuring
# anything. Everything not named here is gated, INCLUDING examples with no
# split assigned; see resolve_data for why that direction matters.
#
# `train` is a legacy alias. These labels are free-form strings — nothing in
# LangSmith enforces them — and this repo renamed test/train to gate/scratch
# because no model is being trained here. But `ensure_dataset()` is idempotent
# and won't relabel a dataset that already exists, so a workspace created
# before the rename still has examples in `train`. Dropping it from this tuple
# would silently start gating someone's scratch examples. Listing a split that
# doesn't exist on a dataset is a no-op.
HELD_OUT_SPLITS: tuple[str, ...] = ("scratch", "train")


def build_suite(suite: str):
    """Return (target_fn, dataset_name, evaluators) for the requested suite."""
    if suite == "single_turn":
        from hr_agent import run_agent
        from hr_agent.trajectory import final_response
        from module_2_single_turn.datasets import ensure_dataset, DATASET_NAME
        from module_2_single_turn.deterministic_evals import DETERMINISTIC_EVALUATORS
        from module_2_single_turn.llm_judge_evals import LLM_JUDGE_EVALUATORS

        def target(inputs: dict) -> dict:
            return {"answer": final_response(run_agent(inputs["question"]))}

        ensure_dataset()
        return target, DATASET_NAME, [*DETERMINISTIC_EVALUATORS, *LLM_JUDGE_EVALUATORS]

    if suite == "agent":
        from hr_agent import run_agent
        from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response
        from module_3_agent_evals.datasets import ensure_dataset, DATASET_NAME
        from module_3_agent_evals.trajectory_evals import TRAJECTORY_EVALUATORS
        from module_3_agent_evals.tool_evals import TOOL_EVALUATORS
        from module_3_agent_evals.llm_trajectory_judge import LLM_TRAJECTORY_EVALUATORS

        def target(inputs: dict) -> dict:
            result = run_agent(inputs["question"])
            return {
                "trajectory": extract_trajectory(result),
                "tool_calls": extract_tool_calls(result),
                "answer": final_response(result),
            }

        ensure_dataset()
        return target, DATASET_NAME, [
            *TRAJECTORY_EVALUATORS, *TOOL_EVALUATORS, *LLM_TRAJECTORY_EVALUATORS,
        ]

    if suite == "tool_failures":
        # Mocked-tool suite: scenarios the real tools can't produce. See
        # module_3_agent_evals/mocked_eval.py.
        from module_3_agent_evals.mock_datasets import ensure_dataset, DATASET_NAME
        from module_3_agent_evals.mocked_eval import target
        from module_3_agent_evals.tool_evals import FAILURE_EVALUATORS
        from module_3_agent_evals.trajectory_evals import no_forbidden_tools, required_tools_used

        ensure_dataset()
        return target, DATASET_NAME, [
            *FAILURE_EVALUATORS, required_tools_used, no_forbidden_tools,
        ]

    raise SystemExit(f"Unknown suite '{suite}'. Choose one of: {', '.join(SUITES)}.")


def _assigned_splits(example) -> list[str]:
    """The splits an example explicitly belongs to.

    LangSmith surfaces membership under ``metadata["dataset_split"]``; an
    example nobody assigned comes back as ``["base"]`` (the implicit default),
    which is what we treat as *unassigned*. Best-effort — used only for the
    accounting line, never for deciding what gets gated.
    """
    metadata = getattr(example, "metadata", None) or {}
    raw = metadata.get("dataset_split") or []
    if isinstance(raw, str):
        raw = [raw]
    return [s for s in raw if s != "base"]


def resolve_data(
    client: Client,
    dataset_name: str,
    *,
    only_split: str | None = None,
    held_out: tuple[str, ...] = HELD_OUT_SPLITS,
    require_splits: bool = False,
):
    """Pick the examples the gate runs on. Returns (examples, scope_label).

    Two modes:

    - ``only_split="gate"`` — evaluate exactly that split, nothing else.
    - default — evaluate *everything except* the ``held_out`` splits.

    The default is an **exclusion, not an inclusion**, and that is the whole
    point. LangSmith drops every example with no explicit split into the
    implicit ``base`` split. Gate on ``splits=["gate"]`` and each example a
    teammate adds through the web UI lands in ``base`` and is silently skipped
    — the gate keeps passing while its coverage quietly shrinks, which is the
    worst failure mode a gate has, because it looks exactly like health.

    Inverting it makes a forgotten split a *false failure* (loud, fixable)
    instead of a *missed test* (invisible).
    """
    if only_split:
        examples = list(client.list_examples(dataset_name=dataset_name, splits=[only_split]))
        if not examples:
            raise SystemExit(
                f"No examples in split '{only_split}' of dataset '{dataset_name}'. "
                "Re-run the module's datasets.py, or drop --split."
            )
        return examples, f"split: {only_split}"

    all_examples = list(client.list_examples(dataset_name=dataset_name))
    if not all_examples:
        raise SystemExit(
            f"Dataset '{dataset_name}' has no examples. Run the module's datasets.py first."
        )

    # Ask the server which examples are held out — authoritative, and immune to
    # however split membership happens to be shaped on the example payload.
    held_ids: set = set()
    for name in held_out:
        try:
            held_ids |= {
                ex.id for ex in client.list_examples(dataset_name=dataset_name, splits=[name])
            }
        except Exception:
            continue  # that split doesn't exist on this dataset — nothing to hold out

    gated = [ex for ex in all_examples if ex.id not in held_ids]
    unassigned = [ex for ex in all_examples if not _assigned_splits(ex)]

    held_label = ", ".join(held_out) or "nothing"
    print(
        f"Dataset '{dataset_name}': {len(all_examples)} examples — "
        f"{len(gated)} gated, {len(held_ids)} held out ({held_label}), "
        f"{len(unassigned)} with no split assigned."
    )

    if unassigned:
        note = (
            f"{len(unassigned)} example(s) have no split assigned, so LangSmith put them "
            f"in 'base'. They ARE being gated — assign them a split to be explicit."
        )
        if require_splits:
            raise SystemExit(f"GATE CONFIG ERROR — {note}")
        print(f"  note: {note}")

    if not gated:
        raise SystemExit(
            f"Every example in '{dataset_name}' is held out ({held_label}). Nothing to gate."
        )

    return gated, f"all except {held_label}"


def aggregate_scores(results) -> dict[str, float]:
    """Mean score per metric across all examples in the experiment."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in results:
        for res in row["evaluation_results"]["results"]:
            if res.score is not None:
                sums[res.key] += float(res.score)
                counts[res.key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an eval suite as a CI gate.")
    parser.add_argument("--suite", choices=list(SUITES), required=True)
    parser.add_argument("--split", default=None,
                        help="Evaluate ONLY this split. Opt-in; skips everything else, "
                             "including examples with no split assigned.")
    parser.add_argument("--held-out", action="append", default=None, metavar="SPLIT",
                        help=f"Split to keep out of the gate (repeatable). "
                             f"Default: {', '.join(HELD_OUT_SPLITS)}.")
    parser.add_argument("--require-splits", action="store_true",
                        help="Fail the gate if any example has no split assigned.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override: apply this single threshold to every metric.")
    args = parser.parse_args()

    held_out = tuple(s.strip() for s in (args.held_out or HELD_OUT_SPLITS) if s.strip())

    require_langsmith()
    client = Client()
    target, dataset_name, evaluators = build_suite(args.suite)
    data, scope_label = resolve_data(
        client,
        dataset_name,
        only_split=args.split,
        held_out=held_out,
        require_splits=args.require_splits,
    )

    results = client.evaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix=f"ci-gate-{args.suite}",
        # This is what actually ties an experiment to a commit. The
        # LANGSMITH_EXPERIMENT env var only affects the pytest plugin, not
        # client.evaluate — see config.experiment_metadata.
        metadata=experiment_metadata(
            suite=args.suite,
            split=args.split or f"all-except:{'+'.join(held_out)}",
            gate=True,
        ),
        description=f"CI gate for the '{args.suite}' suite ({scope_label}).",
        max_concurrency=4,
    )

    means = aggregate_scores(results)

    print(f"\n=== CI gate: {args.suite} [{scope_label}] ===")
    failures: list[str] = []
    for metric in sorted(means):
        mean = means[metric]
        threshold = args.threshold if args.threshold is not None else THRESHOLDS.get(metric, 0.0)
        passed = mean >= threshold
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {metric:<26} mean={mean:.3f}  threshold={threshold:.2f}")
        if not passed:
            failures.append(f"{metric} ({mean:.3f} < {threshold:.2f})")

    if failures:
        print(f"\nGATE FAILED — {len(failures)} metric(s) below threshold:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)  # non-zero exit fails the CI build

    print("\nGATE PASSED — all metrics meet their thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()
