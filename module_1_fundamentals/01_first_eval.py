"""Module 1 — your first end-to-end evaluation.

The smallest possible eval that still has all four moving parts:
  1. DATASET     — 3 hand-written policy questions with expected facts.
  2. TARGET      — the HR agent.
  3. EVALUATOR   — one deterministic check (did the answer contain the fact?).
  4. EXPERIMENT  — client.evaluate runs the target over the dataset and scores it.

Run:  python module_1_fundamentals/01_first_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import run_agent
from hr_agent.trajectory import final_response

# Dataset naming convention: {domain}/{capability}/{version_or_variant}.
# Slash-separated names sort and filter cleanly once you have dozens of them —
# far better than prose titles. See the top-level README.
DATASET_NAME = "hr-onboarding/policy-qa/intro"


# --- 1. DATASET ----------------------------------------------------------
# Each example: inputs (what the target receives) + outputs (ground truth the
# evaluator compares against). LangSmith calls the ground-truth side "outputs"
# on the example; evaluators receive it as `reference_outputs`.
EXAMPLES = [
    {
        "inputs": {"question": "How many vacation days do new employees get?"},
        "outputs": {"expected_fact": "15"},
        "metadata": {"policy_topic": "vacation", "difficulty": "easy"},
        "split": "gate",
    },
    {
        "inputs": {"question": "How many paid sick days are there per year?"},
        "outputs": {"expected_fact": "10"},
        "metadata": {"policy_topic": "sick_leave", "difficulty": "easy"},
        "split": "gate",
    },
    {
        "inputs": {"question": "What is the 401(k) match?"},
        "outputs": {"expected_fact": "4%"},
        "metadata": {"policy_topic": "401k", "difficulty": "easy"},
        "split": "scratch",
    },
]


def ensure_dataset(client: Client) -> str:
    """Create the dataset if it doesn't exist; return its name (idempotent)."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Module 1: three HR policy questions for the first eval.",
        # Dataset-level metadata is how you find this again among hundreds.
        metadata={
            "owner": "evals-workshop",
            "capability": "policy-qa",
            "module": 1,
            "source": "hand-written",
        },
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        # Per-example metadata (tag it) and splits (slice it). Both are
        # per-example sequences, positionally aligned with inputs/outputs.
        metadata=[e["metadata"] for e in EXAMPLES],
        splits=[e["split"] for e in EXAMPLES],
    )
    return DATASET_NAME


# --- 2. TARGET -----------------------------------------------------------
def target(inputs: dict) -> dict:
    """Run the agent on one example's inputs and return its outputs."""
    result = run_agent(inputs["question"])
    return {"answer": final_response(result)}


# --- 3. EVALUATOR --------------------------------------------------------
def mentions_expected_fact(outputs: dict, reference_outputs: dict) -> dict:
    """Deterministic: did the agent's answer contain the expected fact?"""
    answer = outputs.get("answer", "")
    expected = reference_outputs.get("expected_fact", "")
    hit = expected.lower() in answer.lower()
    return {
        "key": "mentions_expected_fact",
        "score": 1 if hit else 0,
        "comment": f"Looking for '{expected}' in the answer: {'found' if hit else 'missing'}.",
    }


# --- 4. EXPERIMENT -------------------------------------------------------
def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[mentions_expected_fact],
        experiment_prefix="module-1-first-eval",
        # Tag the experiment with the code + config that produced it, so a
        # regression in the UI traces back to a commit and a model.
        metadata=experiment_metadata(module=1, suite="first-eval"),
        description="Module 1: first end-to-end eval over three policy questions.",
        max_concurrency=3,
    )

    print("\nExperiment complete. Open it in LangSmith to inspect per-example scores:")
    # The results object knows where it lives in the UI.
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))


if __name__ == "__main__":
    main()
