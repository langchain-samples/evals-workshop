"""Module 2 — structured output + shape validation.

A very common production pattern: you don't want free text, you want a typed
object your code can act on. The first thing to evaluate then is "did the
model return the right SHAPE?" — a purely deterministic check.

This file defines a structured-answer target and runs a tiny experiment scored
ONLY by the deterministic `structured_answer_is_valid` evaluator.

Run:  python module_2_single_turn/structured_output.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from config import experiment_metadata, get_judge, require_langsmith
from module_2_single_turn.datasets import ensure_dataset, DATASET_NAME
from module_2_single_turn.deterministic_evals import structured_answer_is_valid
from hr_agent.knowledge import policy_corpus, resolve_topic
from hr_agent.tools import lookup_hr_policy


class PolicyAnswer(BaseModel):
    """The structured shape we ask the model to return."""
    answer: str = Field(description="A concise, friendly answer for the new hire.")
    policy_topic: str = Field(
        description="One of: vacation, sick_leave, remote_work, health_insurance, 401k, parental_leave, expenses."
    )
    follow_up_needed: bool = Field(
        description="True if the employee should follow up with HR for specifics."
    )


def structured_target(inputs: dict) -> dict:
    """Answer the question as a typed PolicyAnswer object.

    We give the model the relevant policy text (a tiny bit of RAG) and force
    structured output. The eval then checks the *shape* is valid.
    """
    question = inputs["question"]
    model = get_judge().with_structured_output(PolicyAnswer)
    # Pull candidate policy text so the answer can be grounded. resolve_topic
    # maps the question to a known topic key; passing the raw question here
    # would always miss the lookup and return the "no policy found" fallback.
    topic = resolve_topic(question)
    context = lookup_hr_policy.invoke({"topic": topic}) if topic else policy_corpus()
    result: PolicyAnswer = model.invoke(
        f"Policy context: {context}\n\nEmployee question: {question}\n\n"
        "Answer as a PolicyAnswer."
    )
    # Return as a plain dict under 'structured' so the shape evaluator can read it.
    return {"structured": result.model_dump()}


def main() -> None:
    require_langsmith()
    from langsmith import Client

    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        structured_target,
        data=DATASET_NAME,
        evaluators=[structured_answer_is_valid],
        experiment_prefix="module-2-structured-shape",
        metadata=experiment_metadata(module=2, suite="structured-output"),
        description="Shape validation of structured PolicyAnswer output.",
        max_concurrency=4,
    )
    print("\nStructured-output shape experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))


if __name__ == "__main__":
    main()
