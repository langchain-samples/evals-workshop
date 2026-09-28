"""Module 3 — the stale-data experiment: the same agent, two worlds.

Runs the *same agent* over the same dataset twice:

  recorded  the tools replay what they returned when the trace was recorded
            (`TraceReplayMiddleware`) — the world is frozen.
  drifted   the tools report what the live system says "today" — the start date
            moved, the employee was reclassified. Stands in for the passage of
            time; a real project just waits and the data changes by itself.

Then read the score table. For an agent that behaves correctly in both worlds:

  - ``matches_snapshot_calls`` (absolute ground truth) PASSES when frozen and
    FAILS when drifted. Nothing regressed — the answer key went stale.
  - ``args_follow_tool_outputs`` (an invariant) passes in both. It never named a
    value, so there was nothing to go stale.
  - ``replay_fidelity`` (recorded only) tells you whether the agent stayed on the
    recorded path. If it didn't, that IS a behavior change.

Two remedies, then: **freeze the world** (replay) when the right answer really is
a specific value, or **state the expectation as an invariant** when it isn't.

Run:  python module_3_agent_evals/replay_eval.py [--world both|recorded|drifted]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import DatasetDrivenMockMiddleware, TraceReplayMiddleware, run_agent
from hr_agent.replay import recorded_calls_from_messages
from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response
from module_3_agent_evals.replay_datasets import DATASET_NAME, ensure_dataset
from module_3_agent_evals.replay_evals import REPLAY_EVALUATORS
from module_3_agent_evals.trajectory_evals import no_forbidden_tools, required_tools_used

WORLDS = ("recorded", "drifted")


def make_target(world: str):
    """A target that runs the agent in ``world`` ('recorded' or 'drifted').

    A fresh middleware per example — it holds a call log and (for replay) a
    consumption queue, so sharing one across examples would leak state.
    """
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}")

    def target(inputs: dict) -> dict:
        if world == "recorded":
            world_mw = TraceReplayMiddleware(inputs["recorded_calls"])
        else:
            # strict=False: tools the example doesn't override run for real —
            # the drifted world only changes the facts that drifted.
            world_mw = DatasetDrivenMockMiddleware(
                inputs.get("drifted_tool_outputs", {}), strict=False
            )
        result = run_agent(inputs["question"], middleware=[world_mw])
        out = {
            "trajectory": extract_trajectory(result),
            "tool_calls": extract_tool_calls(result),
            "calls": recorded_calls_from_messages(result),
            "answer": final_response(result),
        }
        if world == "recorded":
            out["replay"] = world_mw.report()
        return out

    return target


def score_table(results) -> dict[str, float]:
    """Mean score per evaluator key across an experiment's rows."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in results:
        for r in row["evaluation_results"]["results"]:
            if r.score is not None:
                sums[r.key] += float(r.score)
                counts[r.key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def run_world(world: str, data, client: Client, *, upload_results: bool = True):
    """Run one experiment; return ``(results, {evaluator: mean_score})``."""
    results = client.evaluate(
        make_target(world),
        data=data,
        evaluators=[*REPLAY_EVALUATORS, required_tools_used, no_forbidden_tools],
        experiment_prefix=f"module-3-replay-{world}",
        metadata=experiment_metadata(module=3, suite="point-in-time-replay", world=world),
        description=f"Same agent, {world} world. See replay_eval.py for how to read the pair.",
        max_concurrency=2,
        upload_results=upload_results,
    )
    return results, score_table(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-in-time replay experiment.")
    parser.add_argument("--world", choices=("both", *WORLDS), default="both")
    args = parser.parse_args()

    require_langsmith()
    client = Client()
    ensure_dataset(client)

    worlds = WORLDS if args.world == "both" else (args.world,)
    tables: dict[str, dict[str, float]] = {}
    for world in worlds:
        results, tables[world] = run_world(world, DATASET_NAME, client)
        print(f"\n[{world}] experiment: "
              f"{getattr(results, 'experiment_name', '(see https://smith.langchain.com)')}")

    keys = ["replay_fidelity", "matches_snapshot_calls", "args_follow_tool_outputs"]
    print("\nMean score by world (1.0 = every example passed):\n")
    print(f"  {'evaluator':<28}" + "".join(f"{w:>11}" for w in worlds))
    for key in keys:
        cells = "".join(f"{tables[w].get(key, float('nan')):>11.2f}" for w in worlds)
        print(f"  {key:<28}{cells}")

    if len(worlds) == 2:
        print(
            "\nHow to read it: the agent is identical in both columns. If "
            "'matches_snapshot_calls' drops from recorded to drifted while "
            "'args_follow_tool_outputs' holds, you are looking at a stale answer key, "
            "not a regression. 'replay_fidelity' below 1.0 in the recorded column is the "
            "opposite — the agent left the recorded path, and that is a real change."
        )


if __name__ == "__main__":
    main()
