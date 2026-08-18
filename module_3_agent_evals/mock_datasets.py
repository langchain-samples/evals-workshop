"""Module 3 — the tool-failure dataset (only reachable with mocked tools).

`datasets.py` covers the happy path. This dataset covers what happens when the
world misbehaves: a lookup that finds nobody, a provisioning system that's down,
an API that half-succeeds. **The real tools in `hr_agent/tools.py` can never
return those states** — they always succeed against static data. So without
mocking, this entire class of behavior is untestable.

Each example carries its own mock tool outputs under
``inputs["tool_outputs"]``. The middleware in `hr_agent/mocking.py` reads them
and returns them instead of calling the real tool, so one dataset row fully
describes one world state.

Why ``inputs`` and not ``outputs``? Because `client.evaluate` only passes
``inputs``, ``attachments``, and ``metadata`` to a target function — there is no
``example`` argument (see `langsmith.evaluation._runner._get_target_args`). Mocks
also *are* inputs conceptually: they describe the scenario the agent faces, not
the ground truth about how it should behave.

Run:  python module_3_agent_evals/mock_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/tool-failures/v1"

EXAMPLES = [
    {
        "inputs": {
            "question": "Create an email account for our new hire Jordan Lee.",
            "tool_outputs": {
                "lookup_employee": {
                    "employee_id": "E1007", "full_name": "Jordan Lee",
                    "start_date": "2026-06-15", "benefits_plan": "standard",
                },
                "create_it_account": {
                    "status": "failed",
                    "error": "IT provisioning system is down for maintenance.",
                },
            },
        },
        "outputs": {
            "required_tools": ["lookup_employee", "create_it_account"],
            "should_report_failure": True,
            # The agent must NOT claim success when the tool said "failed".
            "forbidden_phrases": ["successfully created", "account is ready", "all set"],
        },
        "metadata": {"scenario": "downstream-outage", "difficulty": "hard", "source": "mocked"},
        "split": "test",
    },
    {
        "inputs": {
            "question": "Order a laptop for Taylor Brooks.",
            "tool_outputs": {
                "lookup_employee": {"error": "No employee found with name 'Taylor Brooks'."},
            },
        },
        "outputs": {
            "required_tools": ["lookup_employee"],
            "should_report_failure": True,
            # With no employee_id it must not invent one and provision anyway.
            "forbidden_tools": ["provision_equipment"],
            "forbidden_phrases": ["ordered", "on its way"],
        },
        "metadata": {"scenario": "unknown-employee", "difficulty": "medium", "source": "mocked"},
        "split": "test",
    },
    {
        "inputs": {
            "question": "Set up Sam Rivera's Slack account and order them a monitor.",
            "tool_outputs": {
                "lookup_employee": {
                    "employee_id": "E1008", "full_name": "Sam Rivera",
                    "start_date": "2026-06-22", "benefits_plan": "standard",
                },
                "create_it_account": {
                    "status": "created", "account_id": "ACCT-E1008-SLACK",
                    "employee_id": "E1008", "system": "slack",
                },
                # Partial failure: one action worked, the other didn't.
                "provision_equipment": {
                    "status": "backordered",
                    "error": "No monitors in stock; estimated 3 weeks.",
                },
            },
        },
        "outputs": {
            "required_tools": ["lookup_employee", "create_it_account", "provision_equipment"],
            "should_report_failure": True,
            "forbidden_phrases": ["both are ready", "everything is set up"],
        },
        "metadata": {"scenario": "partial-failure", "difficulty": "hard", "source": "mocked"},
        "split": "test",
    },
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the tool-failure dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Module 3: tool-failure scenarios with per-example mocked tool "
            "outputs. Tests agent behavior the real tools cannot produce."
        ),
        metadata={
            "owner": "evals-workshop",
            "capability": "failure-handling",
            "module": 3,
            "complexity": "high",
            "source": "mocked",
            "requires_mocking": True,
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
