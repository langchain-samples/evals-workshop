"""Module 2 — run the full single-turn experiment.

Combines deterministic AND LLM-as-judge evaluators in one experiment, exactly
as you would in practice: cheap objective checks alongside model-graded
quality. Open the result in LangSmith and compare the columns side by side.

Run:  python module_2_single_turn/run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import run_agent
from hr_agent.trajectory import final_response
from module_2_single_turn.datasets import ensure_dataset, DATASET_NAME
from module_2_single_turn.deterministic_evals import DETERMINISTIC_EVALUATORS
from module_2_single_turn.llm_judge_evals import LLM_JUDGE_EVALUATORS


def target(inputs: dict) -> dict:
    """Run the HR agent and return its final free-text answer."""
    result = run_agent(inputs["question"])
    return {"answer": final_response(result)}


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[*DETERMINISTIC_EVALUATORS, *LLM_JUDGE_EVALUATORS],
        experiment_prefix="module-2-single-turn",
        metadata=experiment_metadata(module=2, suite="single-turn"),
        description="Deterministic + LLM-judge evaluators over single-turn policy Q&A.",
        max_concurrency=4,
    )

    print("\nSingle-turn experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nIn LangSmith, sort by 'correctness' or 'groundedness' to find the "
          "weakest answers, then read the judge's reasoning in the comment.")


if __name__ == "__main__":
    main()
