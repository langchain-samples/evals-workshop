"""Module 3 — multi-step onboarding task dataset.

Now the agent has to *do* things, not just answer. Each example is an
onboarding request whose ground truth is the expected tool-call trajectory
(plus the expected args for the key tool, and a list of tools that must NOT be
called). This is what trajectory and tool evaluators score against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/tool-selection/v1"

# Note: lists/dicts inside example outputs are fine — LangSmith stores them as
# JSON. Evaluators read them back from reference_outputs.
EXAMPLES = [
    {
        "inputs": {
            "question": "Please get our new hire Jordan Lee set up: create their email and "
                        "Slack accounts and order them a laptop."
        },
        "outputs": {
            "expected_trajectory": [
                "lookup_employee",
                "create_it_account",
                "create_it_account",
                "provision_equipment",
            ],
            "required_tools": ["lookup_employee", "create_it_account", "provision_equipment"],
            # The agent must use Jordan's real id (E1007) once it looks them up.
            "expected_employee_id": "E1007",
            "forbidden_tools": ["schedule_orientation"],
        },
        "metadata": {"task_type": "multi-provision", "steps": 4, "difficulty": "medium", "source": "hand-written"},
        "split": "gate",
    },
    {
        "inputs": {
            "question": "Schedule orientation for Sam Rivera on their start date."
        },
        "outputs": {
            "expected_trajectory": ["lookup_employee", "schedule_orientation"],
            "required_tools": ["lookup_employee", "schedule_orientation"],
            "expected_employee_id": "E1008",
            "forbidden_tools": ["create_it_account", "provision_equipment"],
        },
        "metadata": {"task_type": "scheduling", "steps": 2, "difficulty": "easy", "source": "hand-written"},
        "split": "gate",
    },
    {
        "inputs": {
            "question": "Alex Chen starts soon — set up their VPN access, order a laptop and a "
                        "monitor, and schedule their orientation for 2026-07-01."
        },
        "outputs": {
            "expected_trajectory": [
                "lookup_employee",
                "create_it_account",
                "provision_equipment",
                "provision_equipment",
                "schedule_orientation",
            ],
            "required_tools": [
                "lookup_employee", "create_it_account", "provision_equipment", "schedule_orientation",
            ],
            "expected_employee_id": "E1009",
            "forbidden_tools": [],
        },
        "metadata": {"task_type": "multi-provision", "steps": 5, "difficulty": "hard", "source": "hand-written"},
        "split": "scratch",
    },
    {
        "inputs": {
            "question": "What benefits plan is Jordan Lee on, and what's the 401k match for it?"
        },
        "outputs": {
            # A read-only request: look up the employee, then their plan. No
            # provisioning should happen.
            "expected_trajectory": ["lookup_employee", "get_benefits_info"],
            "required_tools": ["lookup_employee", "get_benefits_info"],
            "expected_employee_id": "E1007",
            "forbidden_tools": ["create_it_account", "provision_equipment", "schedule_orientation"],
        },
        "metadata": {"task_type": "read-only", "steps": 2, "difficulty": "medium", "source": "hand-written"},
        "split": "gate",
    },
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the trajectory dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Module 3: multi-step onboarding tasks with expected tool trajectories.",
        metadata={
            "owner": "evals-workshop",
            "capability": "tool-selection",
            "module": 3,
            "complexity": "high",
            "source": "hand-written",
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


def examples_for_split(client: Client, split: str | None = None):
    """Return the examples in one split — what you pass to `evaluate(data=...)`."""
    ensure_dataset(client)
    if split is None:
        return DATASET_NAME
    return list(client.list_examples(dataset_name=DATASET_NAME, splits=[split]))


if __name__ == "__main__":
    from config import require_langsmith

    require_langsmith()
    name = ensure_dataset()
    counts: dict[str, int] = {}
    for e in EXAMPLES:
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    print(f"Dataset ready: {name} ({len(EXAMPLES)} examples; splits: {counts})")
