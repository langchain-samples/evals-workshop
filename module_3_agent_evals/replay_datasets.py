"""Module 3 — the replay dataset: examples that carry a frozen world.

`mock_datasets.py` builds worlds that *never existed* (outages). This dataset
carries worlds that *did* exist: each example holds the tool calls and outputs
recorded from a run, so the agent can be re-run later against exactly what it
saw the first time.

Why bother — point-in-time ground truth decays. Take "schedule Jordan Lee's
orientation". When the trace was recorded, Jordan's start date was 2026-06-15,
so the agent scheduled 2026-06-15 and we wrote that down as the expected answer.
A month later HR moves the start date. Re-run the agent *live* and it correctly
schedules the new date — and fails the test, because the ground truth described
the world as it was. Nothing regressed. The data drifted.

Each example therefore carries:

- ``inputs["recorded_calls"]``        — the frozen world (name, args, output per call).
- ``inputs["drifted_tool_outputs"]``  — DEMO ONLY: what the live system says "today".
  Lets `replay_eval.py` show the decay without waiting a month. Real datasets
  don't have this; the live system just changes on its own.
- ``outputs["snapshot_calls"]``       — ground truth written from the trace (absolute).
- ``outputs["derived_args"]``         — ground truth as an *invariant* (relative to
  what the tools returned). This is the alternative to freezing, and it survives
  drift without a recording.

The recorded calls below are hand-written stand-ins for what
`snapshot_trace.py` extracts from a real run.

Run:  python module_3_agent_evals/replay_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/replay-onboarding/v1"

EXAMPLES = [
    {
        "inputs": {
            "question": "Schedule orientation for Jordan Lee on their start date.",
            "recorded_calls": [
                {
                    "name": "lookup_employee",
                    "args": {"name": "Jordan Lee"},
                    "output": {
                        "employee_id": "E1007", "full_name": "Jordan Lee",
                        "title": "Software Engineer", "department": "Engineering",
                        "start_date": "2026-06-15", "manager": "Priya Anand",
                        "benefits_plan": "standard",
                    },
                },
                {
                    "name": "schedule_orientation",
                    "args": {"employee_id": "E1007", "date": "2026-06-15"},
                    "output": {
                        "status": "scheduled", "confirmation_id": "ORI-E1007-20260615",
                        "employee_id": "E1007", "date": "2026-06-15",
                    },
                },
            ],
            # DEMO ONLY — HR moved the start date after the trace was recorded.
            "drifted_tool_outputs": {
                "lookup_employee": {
                    "employee_id": "E1007", "full_name": "Jordan Lee",
                    "title": "Software Engineer", "department": "Engineering",
                    "start_date": "2026-07-06", "manager": "Priya Anand",
                    "benefits_plan": "standard",
                },
            },
        },
        "outputs": {
            # Absolute: true only while the world matches the recording.
            "snapshot_calls": [
                {"name": "schedule_orientation", "args": {"employee_id": "E1007", "date": "2026-06-15"}},
            ],
            # Relative: true in any world — the date passed must be the start
            # date the lookup returned.
            "derived_args": [
                {"tool": "schedule_orientation", "arg": "date",
                 "from_tool": "lookup_employee", "field": "start_date"},
            ],
        },
        "metadata": {"scenario": "start-date-drift", "difficulty": "easy", "source": "replay"},
        "split": "gate",
    },
    {
        "inputs": {
            "question": "What benefits plan is Alex Chen on? Give me the plan details.",
            "recorded_calls": [
                {
                    "name": "lookup_employee",
                    "args": {"name": "Alex Chen"},
                    "output": {
                        "employee_id": "E1009", "full_name": "Alex Chen",
                        "title": "Engineering Manager", "department": "Engineering",
                        "start_date": "2026-07-01", "manager": "Morgan Doyle",
                        "benefits_plan": "executive",
                    },
                },
                {
                    "name": "get_benefits_info",
                    "args": {"plan": "executive"},
                    "output": {"plan": "executive", "note": "recorded executive-plan details"},
                },
            ],
            # DEMO ONLY — Alex was reclassified after the trace was recorded.
            "drifted_tool_outputs": {
                "lookup_employee": {
                    "employee_id": "E1009", "full_name": "Alex Chen",
                    "title": "Engineering Manager", "department": "Engineering",
                    "start_date": "2026-07-01", "manager": "Morgan Doyle",
                    "benefits_plan": "standard",
                },
            },
        },
        "outputs": {
            "snapshot_calls": [
                {"name": "get_benefits_info", "args": {"plan": "executive"}},
            ],
            "derived_args": [
                {"tool": "get_benefits_info", "arg": "plan",
                 "from_tool": "lookup_employee", "field": "benefits_plan"},
            ],
        },
        "metadata": {"scenario": "plan-reclassified", "difficulty": "medium", "source": "replay"},
        "split": "gate",
    },
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the replay dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Module 3: examples that carry a recorded (frozen) world so the agent "
            "can be re-run against exactly what its tools returned at the time."
        ),
        metadata={
            "owner": "evals-workshop",
            "capability": "point-in-time-replay",
            "module": 3,
            "complexity": "high",
            "source": "replay",
            "requires_replay": True,
        },
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        metadata=[e["metadata"] for e in EXAMPLES],
        splits=[e["split"] for e in EXAMPLES],
    )
    return DATASET_NAME


if __name__ == "__main__":
    from config import require_langsmith

    require_langsmith()
    name = ensure_dataset()
    print(f"Dataset ready: {name} ({len(EXAMPLES)} examples)")
