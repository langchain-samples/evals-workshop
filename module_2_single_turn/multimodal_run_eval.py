"""Module 2 — run the multimodal experiment: receipt images in, typed decision out.

The target takes ``(inputs, attachments)``. Naming the second parameter
``attachments`` is all it takes for ``evaluate`` to download each example's files
and pass them in. The same goes for evaluators.

Run:  python module_2_single_turn/multimodal_run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent.receipts import review_expense_claim
from module_2_single_turn.multimodal_datasets import (
    DATASET_NAME,
    ensure_dataset,
    files_from_attachments,
)
from module_2_single_turn.multimodal_evals import (
    DETERMINISTIC_EVALUATORS,
    LLM_JUDGE_EVALUATORS,
)


def target(inputs: dict, attachments: dict) -> dict:
    """Review one expense claim using every file attached to the example."""
    review = review_expense_claim(
        note=inputs["note"],
        submitted_on=inputs["submitted_on"],
        files=files_from_attachments(attachments),
    )
    return review.model_dump()


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[*DETERMINISTIC_EVALUATORS, *LLM_JUDGE_EVALUATORS],
        experiment_prefix="module-2-multimodal",
        metadata=experiment_metadata(module=2, suite="multimodal"),
        description="Expense claims with receipt images + CSV attachments; deterministic + image-aware judge.",
        max_concurrency=4,
    )

    print("\nMultimodal experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nIn LangSmith, open an example to see its attachments rendered, and group "
          "by metadata 'image_quality' or 'scenario' to compare slices.")


if __name__ == "__main__":
    main()
