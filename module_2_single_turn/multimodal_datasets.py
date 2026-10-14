"""Module 2 — a multimodal dataset: expense claims with receipt images attached.

Each example is one expense claim. The text goes in ``inputs`` as usual; the files
go in ``attachments`` — a dict of named files, so a single example can carry
several of them:

    {
        "inputs":  {"note": "...", "submitted_on": "2026-09-30"},
        "outputs": {"decision": "needs_receipt", "receipts": [...], ...},
        "attachments": {
            "receipt_1":      ("image/png", <bytes>),
            "receipt_2":      ("image/png", <bytes>),
            "card_statement": ("text/csv",  <bytes>),
        },
    }

LangSmith stores each attachment as a file next to the example (not as base64
inside ``inputs``), renders it in the UI, and hands it to your target and
evaluators as ``attachments[name] = {"reader", "mime_type", "presigned_url"}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from module_2_single_turn.multimodal_fixtures import (
    Receipt,
    Statement,
    not_a_receipt_png,
    receipt_png,
)

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/expense-receipts/v1"

# Each attachment value is a Receipt (drawn as a PNG), a Statement (CSV), or the
# string "not_a_receipt". The answer key below is derived from these same specs.
EXAMPLES = [
    {
        "id": "single-receipt",
        "inputs": {"note": "Lunch with my onboarding buddy.", "submitted_on": "2026-09-10"},
        "files": {
            "receipt_1": Receipt(
                "Corner Deli", "2026-09-02",
                (("Turkey sandwich", 11.50), ("Iced tea", 3.25)), tax=1.31,
            ),
        },
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"scenario": "single", "image_quality": "clean", "n_attachments": 1},
        "split": "gate",
    },
    {
        "id": "two-receipts",
        "inputs": {"note": "Travel to the Boston office for orientation.", "submitted_on": "2026-09-20"},
        "files": {
            "receipt_1": Receipt(
                "Metro Cab Co", "2026-09-14", (("Fare", 36.40), ("Tip", 5.70)),
            ),
            "receipt_2": Receipt(
                "Harbor Inn", "2026-09-14",
                (("Room, 1 night", 169.00), ("Occupancy tax", 20.00)),
            ),
        },
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"scenario": "multi", "image_quality": "clean", "n_attachments": 2},
        "split": "gate",
    },
    {
        # The interesting one: the answer depends on reading ACROSS attachments.
        # The statement has three charges; only two have receipts, and the third is
        # over the $25 receipt threshold.
        "id": "statement-missing-receipt",
        "inputs": {
            "note": "Conference day expenses. Card statement attached for all charges.",
            "submitted_on": "2026-09-25",
        },
        "files": {
            "receipt_1": Receipt("Noodle Bar", "2026-09-22", (("Lunch special", 21.00),), tax=1.80),
            "receipt_2": Receipt(
                "Paper Trail Supplies", "2026-09-22",
                (("Notebooks (3)", 17.85), ("Pens", 11.25)), tax=2.35,
            ),
            "card_statement": Statement((
                ("2026-09-22", "NOODLE BAR 0412", 22.80),
                ("2026-09-22", "PAPER TRAIL SUPPLIES", 31.45),
                ("2026-09-22", "GATE 12 PARKING", 48.00),
            )),
        },
        "decision": "needs_receipt",
        "missing_receipt_keywords": ["parking"],
        "metadata": {"scenario": "cross-attachment", "image_quality": "clean", "n_attachments": 3},
        "split": "gate",
    },
    {
        "id": "late-submission",
        "inputs": {"note": "Shuttle from the airport on my first day.", "submitted_on": "2026-09-15"},
        "files": {
            "receipt_1": Receipt("Airport Shuttle Express", "2026-07-30", (("One-way fare", 28.00),)),
        },
        "decision": "reject",  # 47 days later; the policy allows 30
        "missing_receipt_keywords": [],
        "metadata": {"scenario": "single", "image_quality": "clean", "n_attachments": 1},
        "split": "gate",
    },
    {
        # One real receipt plus an image that isn't a receipt at all.
        "id": "wrong-attachment",
        "inputs": {"note": "Book for onboarding. Receipt attached.", "submitted_on": "2026-09-09"},
        "files": {
            "receipt_1": Receipt("Lantern Bookshop", "2026-09-05", (("Team handbook", 34.99),), tax=2.80),
            "photo": "not_a_receipt",
        },
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"scenario": "distractor", "image_quality": "clean", "n_attachments": 2},
        "split": "gate",
    },
    {
        "id": "phone-photo",
        "inputs": {"note": "Printing for the new-hire welcome packet.", "submitted_on": "2026-09-18"},
        "files": {
            "receipt_1": Receipt(
                "Copy Center", "2026-09-16",
                (("Color prints (40)", 22.00), ("Binding", 3.60)), tax=2.00, degraded=True,
            ),
        },
        "decision": "approve",
        "missing_receipt_keywords": [],
        "metadata": {"scenario": "single", "image_quality": "degraded", "n_attachments": 1},
        "split": "scratch",
    },
]


def build_attachments(example: dict) -> dict[str, tuple[str, bytes]]:
    """Render an example's files into ``{name: (mime_type, bytes)}``."""
    out: dict[str, tuple[str, bytes]] = {}
    for name, spec in example["files"].items():
        if isinstance(spec, Receipt):
            out[name] = ("image/png", receipt_png(spec))
        elif isinstance(spec, Statement):
            out[name] = ("text/csv", spec.to_csv())
        elif spec == "not_a_receipt":
            out[name] = ("image/png", not_a_receipt_png())
        else:
            raise ValueError(f"unknown file spec for {name!r}: {spec!r}")
    return out


def reference_outputs(example: dict) -> dict:
    """The answer key, derived from the same specs that drew the images."""
    receipts = [
        {"source": name, "vendor": spec.vendor, "date": spec.date, "total": spec.total}
        for name, spec in example["files"].items()
        if isinstance(spec, Receipt)
    ]
    return {
        "decision": example["decision"],
        "receipts": receipts,
        "missing_receipt_keywords": example["missing_receipt_keywords"],
    }


def ensure_dataset(client: Client | None = None) -> str:
    """Create the multimodal dataset if needed; return its name (idempotent)."""
    client = client or Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Module 2: expense claims with receipt images (and a CSV) as attachments.",
        metadata={
            "owner": "evals-workshop",
            "capability": "expense-receipts",
            "module": 2,
            "modality": "image+text",
            "source": "synthetic",
        },
    )
    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=[
            {
                "inputs": e["inputs"],
                "outputs": reference_outputs(e),
                "metadata": {"example_id": e["id"], **e["metadata"]},
                "split": e["split"],
                # Several files in ONE example. Bytes, not file paths: passing a
                # Path would need dangerously_allow_filesystem=True.
                "attachments": build_attachments(e),
            }
            for e in EXAMPLES
        ],
    )
    return DATASET_NAME


def files_from_attachments(attachments: dict) -> list[tuple[str, str, bytes]]:
    """Turn the ``attachments`` a target/evaluator receives into ``(name, mime, bytes)``.

    Each value carries a ``reader``. Rewind it after reading: the runner only
    resets readers after the target and after the last evaluator, so a second
    evaluator reading the same file would otherwise get empty bytes.
    """
    files = []
    for name in sorted(attachments):  # stable order; dict order isn't guaranteed
        info = attachments[name]
        reader = info["reader"]
        data = reader.read()
        reader.seek(0)
        files.append((name, info.get("mime_type") or "application/octet-stream", data))
    return files


if __name__ == "__main__":
    from config import require_langsmith

    require_langsmith()
    name = ensure_dataset()
    n_files = sum(len(e["files"]) for e in EXAMPLES)
    print(f"Dataset ready: {name} ({len(EXAMPLES)} examples, {n_files} attachments)")
