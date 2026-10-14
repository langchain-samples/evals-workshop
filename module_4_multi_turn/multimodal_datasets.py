"""Module 4 — multimodal multi-turn: files arrive across turns.

An expense claim rarely lands in one message. The employee sends receipts, the
reviewer replies that one is missing, and the employee sends it — or withdraws
the charge, or sends the wrong file. Each example is one such conversation.

Every file from every turn lives in the example's ``attachments``, and each turn
lists the names of the files it carried:

    inputs = {"submitted_on": "2026-09-25", "turns": [
        {"role": "user", "text": "Conference expenses.", "attachments": ["receipt_1", "receipt_2", "card_statement"]},
        {"role": "assistant", "text": "The $48.00 parking charge needs a receipt."},
        {"role": "user", "text": "Here it is.", "attachments": ["receipt_3"]},
    ]}
    attachments = {"receipt_1": ..., "receipt_2": ..., "card_statement": ..., "receipt_3": ...}

The answer key is the claim's state after the LAST turn, in the same shape as
Module 2's, so Module 2's deterministic evaluators grade it unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from module_2_single_turn.multimodal_datasets import build_attachments
from module_2_single_turn.multimodal_fixtures import Receipt, Statement

DATASET_NAME = "hr-onboarding/expense-receipts-multi-turn/v1"

_CONFERENCE_FILES = {
    "receipt_1": Receipt("Noodle Bar", "2026-09-22", (("Lunch special", 21.00),), tax=1.80),
    "receipt_2": Receipt(
        "Paper Trail Supplies", "2026-09-22", (("Notebooks (3)", 17.85), ("Pens", 11.25)), tax=2.35,
    ),
    "card_statement": Statement((
        ("2026-09-22", "NOODLE BAR 0412", 22.80),
        ("2026-09-22", "PAPER TRAIL SUPPLIES", 31.45),
        ("2026-09-22", "GATE 12 PARKING", 48.00),
    )),
}
_CONFERENCE_OPENING = [
    {"role": "user", "text": "Conference day expenses. Card statement attached for all charges.",
     "attachments": ["receipt_1", "receipt_2", "card_statement"]},
    {"role": "assistant", "text": "Thanks! The statement has three charges but there are only two "
     "receipts. The $48.00 GATE 12 PARKING charge is over $25, so it needs a receipt."},
]

# "receipts" names the files that should count as receipts once the conversation
# is over. A replaced or wrong file is deliberately left out.
EXAMPLES = [
    {
        "id": "receipt-arrives-later",
        "submitted_on": "2026-09-25",
        "files": {**_CONFERENCE_FILES,
                  "receipt_3": Receipt("Gate 12 Parking", "2026-09-22", (("All-day parking", 48.00),))},
        "turns": [*_CONFERENCE_OPENING,
                  {"role": "user", "text": "Here's the parking receipt.", "attachments": ["receipt_3"]}],
        "receipts": ["receipt_1", "receipt_2", "receipt_3"],
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"skill": "file-arrives-later"},
    },
    {
        # No new file: a sentence in turn 3 overrides what the CSV in turn 1 says.
        "id": "charge-withdrawn",
        "submitted_on": "2026-09-25",
        "files": _CONFERENCE_FILES,
        "turns": [*_CONFERENCE_OPENING,
                  {"role": "user", "text": "Oh, the parking was personal. Please drop it from the claim."}],
        "receipts": ["receipt_1", "receipt_2"],
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"skill": "correction"},
    },
    {
        # The "receipt" sent in turn 3 is the welcome banner. Still missing.
        "id": "wrong-file-sent",
        "submitted_on": "2026-09-25",
        "files": {**_CONFERENCE_FILES, "photo": "not_a_receipt"},
        "turns": [*_CONFERENCE_OPENING,
                  {"role": "user", "text": "Here's the parking receipt.", "attachments": ["photo"]}],
        "receipts": ["receipt_1", "receipt_2"],
        "decision": "needs_receipt",
        "missing_receipt_keywords": ["parking"],
        "metadata": {"skill": "distractor"},
    },
    {
        # receipt_2 is superseded by receipt_3; only the replacement counts.
        "id": "receipt-replaced",
        "submitted_on": "2026-09-20",
        "files": {
            "receipt_1": Receipt("Metro Cab Co", "2026-09-14", (("Fare", 36.40), ("Tip", 5.70))),
            "receipt_2": Receipt("Harbor Inn", "2026-09-14", (("Room, 1 night", 169.00), ("Occupancy tax", 20.00))),
            "receipt_3": Receipt("Harbor Inn", "2026-09-14", (("Room, 1 night", 149.00), ("Occupancy tax", 18.00))),
        },
        "turns": [
            {"role": "user", "text": "Two receipts from the Boston orientation trip.",
             "attachments": ["receipt_1", "receipt_2"]},
            {"role": "assistant", "text": "Got both: Metro Cab Co for $42.10 and Harbor Inn for "
             "$189.00, both on 2026-09-14. Both are within 30 days."},
            {"role": "user", "text": "Sorry, the hotel one was wrong. They rebilled me at the "
             "corporate rate. Here's the corrected hotel receipt.", "attachments": ["receipt_3"]},
        ],
        "receipts": ["receipt_1", "receipt_3"],
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"skill": "file-replaced"},
    },
]


def reference_outputs(example: dict) -> dict:
    files = example["files"]
    return {
        "decision": example["decision"],
        "receipts": [
            {"source": name, "vendor": files[name].vendor, "date": files[name].date, "total": files[name].total}
            for name in example["receipts"]
        ],
        "missing_receipt_keywords": example["missing_receipt_keywords"],
    }


def ensure_dataset(client: Client | None = None) -> str:
    """Create the multimodal multi-turn dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Module 4: expense claims made over several turns; files attached in different turns.",
        metadata={"owner": "evals-workshop", "capability": "expense-receipts", "module": 4,
                  "modality": "image+text", "source": "synthetic"},
    )
    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=[
            {
                "inputs": {"submitted_on": e["submitted_on"], "turns": e["turns"]},
                "outputs": reference_outputs(e),
                "metadata": {"example_id": e["id"], "n_turns": len(e["turns"]),
                             "n_attachments": len(e["files"]), **e["metadata"]},
                "split": "gate",
                "attachments": build_attachments(e),  # every turn's files, one example
            }
            for e in EXAMPLES
        ],
    )
    return DATASET_NAME


if __name__ == "__main__":
    from config import require_langsmith

    require_langsmith()
    print(f"Dataset ready: {ensure_dataset()} ({len(EXAMPLES)} conversations)")
