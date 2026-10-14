"""Module 4 — run the turn-level experiment.

Each example's ``messages`` go to the agent as real role-tagged messages. We keep
only what the agent added after them: this turn's tool calls and final reply.

Run:  python module_4_multi_turn/run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import build_agent, extract_tool_calls, final_response
from module_4_multi_turn.datasets import DATASET_NAME, ensure_dataset
from module_4_multi_turn.turn_evals import DETERMINISTIC_EVALUATORS, LLM_JUDGE_EVALUATORS

ROLES = {"user", "assistant"}


def target(inputs: dict) -> dict:
    """Run the agent on the conversation so far; return only the new turn."""
    history = inputs["messages"]
    if not history or history[-1]["role"] != "user":
        raise ValueError("The conversation must end with a user message.")
    if any(m["role"] not in ROLES for m in history):
        raise ValueError("Messages must be 'user' or 'assistant'.")
    result = build_agent().invoke({"messages": [(m["role"], m["content"]) for m in history]})
    # The result holds the input history followed by everything this turn added.
    new = {"messages": result["messages"][len(history):]}
    return {"answer": final_response(new), "tool_calls": extract_tool_calls(new)}


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[*DETERMINISTIC_EVALUATORS, *LLM_JUDGE_EVALUATORS],
        experiment_prefix="module-4-turn-level",
        metadata=experiment_metadata(module=4, suite="multi-turn"),
        description="Conversation history in, next turn graded: tool calls, facts, context use.",
        max_concurrency=4,
    )

    print("\nTurn-level experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nGroup by metadata 'skill' to see which multi-turn skill is weakest.")


if __name__ == "__main__":
    main()
