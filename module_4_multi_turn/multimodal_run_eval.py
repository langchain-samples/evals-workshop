"""Module 4 — run the multimodal multi-turn experiment.

The target puts each turn's files back into that turn's message, using the names
listed in the turn, then asks for the claim's state after the last message.
Module 2's evaluators grade the result; only the input got longer.

Run:  python module_4_multi_turn/multimodal_run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent.receipts import review_expense_thread
from module_2_single_turn.multimodal_datasets import files_from_attachments
from module_2_single_turn.multimodal_evals import DETERMINISTIC_EVALUATORS
from module_4_multi_turn.multimodal_datasets import DATASET_NAME, ensure_dataset


def build_turns(inputs: dict, attachments: dict) -> list[dict]:
    """Attach each turn's files to it, by name. An unknown name is an error, not a skip."""
    by_name = {name: (name, mime, data) for name, mime, data in files_from_attachments(attachments)}
    turns = []
    for turn in inputs["turns"]:
        names = turn.get("attachments", [])
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(f"Turn references attachments the example doesn't have: {unknown}")
        turns.append({"role": turn["role"], "text": turn["text"], "files": [by_name[n] for n in names]})
    return turns


def target(inputs: dict, attachments: dict) -> dict:
    review = review_expense_thread(build_turns(inputs, attachments), inputs["submitted_on"])
    return review.model_dump()


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=DETERMINISTIC_EVALUATORS,
        experiment_prefix="module-4-multimodal-multi-turn",
        metadata=experiment_metadata(module=4, suite="multimodal-multi-turn"),
        description="Expense claims over several turns; receipts attached in different turns.",
        max_concurrency=4,
    )

    print("\nMultimodal multi-turn experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nGroup by metadata 'skill': a correction in words and a replaced file are the hard ones.")


if __name__ == "__main__":
    main()
