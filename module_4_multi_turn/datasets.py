"""Module 4 — turn-level multi-turn dataset.

Each example is a conversation so far plus the user's next message:

    inputs = {"messages": [
        {"role": "user",      "content": "Hi, I'm helping Jordan Lee get set up."},
        {"role": "assistant", "content": "Happy to help! Jordan Lee is E1007..."},
        {"role": "user",      "content": "Create their Slack account."},   # <- the turn under test
    ]}

The agent gets the whole list and we grade only what it does in reply to the last
message. That is the fix Module 3 points to: make each example *be a turn*, so the
run boundary and the assertion boundary are the same thing.

The earlier turns are fixed text, not re-generated. Every run sees exactly the
same history, so a failure is about this turn and nothing else. The trade-off:
the assistant turns are plain text, not the tool calls a real thread would hold.
To test against real history, snapshot a production turn instead
(``snapshot_trace.py --last-turn`` in Module 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/multi-turn/v1"


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str) -> dict:
    return {"role": "assistant", "content": text}


# expected_calls:  every one must appear in THIS turn's tool calls (args are a subset match)
# forbidden_calls: none may appear in this turn (same partial match)
# expected_facts:  substrings the reply must contain
EXAMPLES = [
    {
        "id": "pronoun",
        "messages": [
            _u("Hi, I'm helping Jordan Lee get set up before day one."),
            _a("Happy to help get Jordan Lee ready! I found their record: Software Engineer "
               "in Engineering, starting 2026-06-15, employee ID E1007. What do you need?"),
            _u("Create their Slack account."),
        ],
        "outputs": {
            "expected_calls": [{"name": "create_it_account", "args": {"employee_id": "E1007", "system": "slack"}}],
            "forbidden_calls": [],
            "expected_facts": [],
        },
        "metadata": {"skill": "reference-resolution"},
        "split": "gate",
    },
    {
        # "the same" refers back two turns; the employee changes. The classic
        # failure is reusing Jordan's id because it's the one in context.
        "id": "switch-subject",
        "messages": [
            _u("Set up an email account for Jordan Lee."),
            _a("Done! Jordan Lee (E1007) now has an email account: ACCT-E1007-EMAIL."),
            _u("Great. Now do the same for Sam Rivera."),
        ],
        "outputs": {
            "expected_calls": [{"name": "create_it_account", "args": {"employee_id": "E1008", "system": "email"}}],
            "forbidden_calls": [{"name": "create_it_account", "args": {"employee_id": "E1007"}}],
            "expected_facts": [],
        },
        "metadata": {"skill": "context-bleed"},
        "split": "gate",
    },
    {
        "id": "correction",
        "messages": [
            _u("Please order a laptop and a monitor for Alex Chen."),
            _a("Alex Chen is E1009. I can order a laptop and a monitor. Shall I go ahead?"),
            _u("Actually, swap the monitor for a headset. Then go ahead."),
        ],
        "outputs": {
            "expected_calls": [
                {"name": "provision_equipment", "args": {"employee_id": "E1009", "equipment_type": "laptop"}},
                {"name": "provision_equipment", "args": {"employee_id": "E1009", "equipment_type": "headset"}},
            ],
            "forbidden_calls": [{"name": "provision_equipment", "args": {"equipment_type": "monitor"}}],
            "expected_facts": [],
        },
        "metadata": {"skill": "correction"},
        "split": "gate",
    },
    {
        # The subject (Sam) was set two turns ago; this turn only supplies a date.
        "id": "answer-to-clarification",
        "messages": [
            _u("I need to schedule orientation for Sam Rivera."),
            _a("Sure! Sam Rivera is E1008. What date would you like?"),
            _u("July 6th, 2026."),
        ],
        "outputs": {
            "expected_calls": [{"name": "schedule_orientation", "args": {"employee_id": "E1008", "date": "2026-07-06"}}],
            "forbidden_calls": [],
            "expected_facts": [],
        },
        "metadata": {"skill": "reference-resolution"},
        "split": "gate",
    },
    {
        # "them" = vacation days. The 90-day rule wasn't said yet, so it has to
        # be looked up, not recalled from the earlier reply.
        "id": "policy-follow-up",
        "messages": [
            _u("How many vacation days do I get?"),
            _a("New full-time employees accrue 15 paid vacation days per year, at 1.25 days per month."),
            _u("When can I actually use them?"),
        ],
        "outputs": {
            "expected_calls": [{"name": "lookup_hr_policy", "args": {"topic": "vacation"}}],
            "forbidden_calls": [],
            "expected_facts": ["90"],
            "policy_topic": "vacation",
        },
        "metadata": {"skill": "reference-resolution"},
        "split": "gate",
    },
    {
        # The policy says nothing about carry-over. A good reply says so.
        "id": "unanswerable-follow-up",
        "messages": [
            _u("How many vacation days do I get?"),
            _a("New full-time employees accrue 15 paid vacation days per year, at 1.25 days "
               "per month, usable after the 90-day probationary period."),
            _u("And do unused ones carry over to next year?"),
        ],
        "outputs": {
            "expected_calls": [],
            "forbidden_calls": [],
            "expected_facts": [],
            "policy_topic": "vacation",
        },
        "metadata": {"skill": "groundedness"},
        "split": "scratch",
    },
]


def ensure_dataset(client: Client | None = None) -> str:
    """Create the turn-level dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Module 4: conversation history + next user message; grade the next turn.",
        metadata={
            "owner": "evals-workshop",
            "capability": "multi-turn",
            "module": 4,
            "source": "hand-written",
        },
    )
    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=[
            {
                "inputs": {"messages": e["messages"]},
                "outputs": e["outputs"],
                "metadata": {"example_id": e["id"], "n_turns": len(e["messages"]), **e["metadata"]},
                "split": e["split"],
            }
            for e in EXAMPLES
        ],
    )
    return DATASET_NAME


if __name__ == "__main__":
    from config import require_langsmith

    require_langsmith()
    print(f"Dataset ready: {ensure_dataset()} ({len(EXAMPLES)} examples)")
