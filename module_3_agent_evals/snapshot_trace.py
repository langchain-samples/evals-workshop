"""Module 3 — snapshot a trace's tool outputs so it can be replayed later.

This is the step that turns "the agent did something interesting in production"
into "here is a test case that stays valid after the data changes": read a
LangSmith trace, pull out every tool call and the output it got, and (optionally)
file that as a dataset example. `replay_eval.py` then re-runs the agent against
those frozen outputs.

    python module_3_agent_evals/snapshot_trace.py --run-id <root-run-uuid>            # inspect
    python module_3_agent_evals/snapshot_trace.py --run-id <uuid> --last-turn         # one turn of a thread
    python module_3_agent_evals/snapshot_trace.py --run-id <uuid> --add-to-dataset    # write it

**Dry run by default.** Without ``--add-to-dataset`` nothing is written — you get
the recorded calls printed as JSON. Do that first, on one of your own traces:
LangSmith stores serialized messages in a few shapes, and this reader is lenient
about them, but you want to see that *your* traces come out looking right.

**No ground truth is invented.** The new example has ``outputs = {}`` and
``needs_review: true``. What the agent *should* have done is a human's call —
add ``snapshot_calls`` and/or ``derived_args`` (see `replay_datasets.py`) in the
UI. It lands in the ``scratch`` split so the CI gate doesn't score an example
with no answer key; move it to ``gate`` once someone has filled one in.

⚠️  **Traces of a real HR agent carry employee PII.** Recorded tool outputs are
whole employee records. This script does not redact anything — same stance as
`module_6_improving_evals/curate_dataset.py`, and for the same reason. Dry-run
first, and give the dataset the access controls of the project it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import require_langsmith
from hr_agent.replay import recorded_calls_from_run, validate_recorded
from module_5_online_evals.score_traces import _question_from_run

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DEFAULT_DATASET = "hr-onboarding/replay-onboarding/from-production"

MAX_QUESTION_CHARS = 4_000


def build_example(run, *, last_turn_only: bool = False) -> dict:
    """Turn a LangSmith run into a replay example (pure — no network, no writes).

    Raises ``ValueError`` if the run has nothing replayable or its recording
    fails validation (unknown tool, oversized payload, too many calls).
    """
    calls = validate_recorded(recorded_calls_from_run(run, last_turn_only=last_turn_only))
    if not calls:
        raise ValueError("No completed tool calls in this run — nothing to replay.")
    started = getattr(run, "start_time", None)
    return {
        "inputs": {
            "question": _question_from_run(run)[:MAX_QUESTION_CHARS],
            "recorded_calls": calls,
        },
        "outputs": {},  # deliberately empty: the answer key is a human's job
        "metadata": {
            "source": "trace-snapshot",
            "source_run_id": str(run.id),
            # The point in time this world is frozen at.
            "as_of": started.date().isoformat() if started else None,
            "needs_review": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot a trace's tool outputs for point-in-time replay.",
        epilog=(
            "WARNING: recorded tool outputs are copied VERBATIM and can contain "
            "employee PII. Nothing is redacted. Dry-run first."
        ),
    )
    parser.add_argument("--run-id", required=True, help="The agent's ROOT run id (a trace id).")
    parser.add_argument("--last-turn", action="store_true",
                        help="Keep only calls after the last human message (for threads).")
    parser.add_argument("--add-to-dataset", action="store_true",
                        help="Write the example. Without this flag nothing is written.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="scratch",
                        help="Split for the new example. 'scratch' until it has an answer key.")
    args = parser.parse_args()

    require_langsmith()
    client = Client()
    run = client.read_run(args.run_id)

    try:
        example = build_example(run, last_turn_only=args.last_turn)
    except ValueError as e:
        raise SystemExit(f"Cannot snapshot run {args.run_id}: {e}")

    calls = example["inputs"]["recorded_calls"]
    print(f"Run {args.run_id}: {len(calls)} recorded tool call(s), "
          f"as of {example['metadata']['as_of']}.\n")
    print(json.dumps(example["inputs"], indent=2, default=str))

    if not args.add_to_dataset:
        print("\n--add-to-dataset not given; nothing written.")
        return

    # Idempotent on source_run_id, like curate_dataset.py.
    if client.has_dataset(dataset_name=args.dataset):
        dataset_id = client.read_dataset(dataset_name=args.dataset).id
        for ex in client.list_examples(dataset_id=dataset_id):
            if ex.source_run_id and str(ex.source_run_id) == str(run.id):
                raise SystemExit(f"Run {run.id} is already in '{args.dataset}'; nothing to add.")
    else:
        dataset_id = client.create_dataset(
            dataset_name=args.dataset,
            description="Replay examples snapshotted from production traces. "
                        "Each links back to its source run.",
            metadata={"owner": "evals-workshop", "capability": "point-in-time-replay",
                      "source": "trace-snapshot", "requires_replay": True},
        ).id

    client.create_examples(
        dataset_id=dataset_id,
        inputs=[example["inputs"]],
        outputs=[example["outputs"]],
        metadata=[example["metadata"]],
        splits=[args.split],
        source_run_ids=[run.id],
    )
    print(f"\nAdded 1 example to '{args.dataset}' (split: {args.split}).")
    print("REMINDER: recorded outputs are verbatim — review them for PII before sharing.")
    print("Next: open the example in LangSmith and add an answer key "
          "(`snapshot_calls` and/or `derived_args`), then move it to the 'gate' split.")


if __name__ == "__main__":
    main()
