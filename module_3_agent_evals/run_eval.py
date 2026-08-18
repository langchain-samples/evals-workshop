"""Module 3 — run the full agent (trajectory + tool) experiment.

The target here returns BOTH the trajectory (tool names) and the full tool
calls (names + args), plus the final answer — so every evaluator family has
what it needs from a single agent run.

Run:  python module_3_agent_evals/run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import run_agent
from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response
from module_3_agent_evals.datasets import ensure_dataset, DATASET_NAME
from module_3_agent_evals.trajectory_evals import TRAJECTORY_EVALUATORS
from module_3_agent_evals.tool_evals import TOOL_EVALUATORS
from module_3_agent_evals.llm_trajectory_judge import LLM_TRAJECTORY_EVALUATORS


def target(inputs: dict) -> dict:
    """Run the agent once; surface trajectory, tool calls, and final answer."""
    result = run_agent(inputs["question"])
    return {
        "trajectory": extract_trajectory(result),
        "tool_calls": extract_tool_calls(result),
        "answer": final_response(result),
    }


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            *TRAJECTORY_EVALUATORS,      # exact match, required tools, forbidden, efficiency
            *TOOL_EVALUATORS,            # correct employee_id, well-formed args
            *LLM_TRAJECTORY_EVALUATORS,  # holistic "was this reasonable?"
        ],
        experiment_prefix="module-3-agent-evals",
        metadata=experiment_metadata(module=3, suite="agent-trajectories"),
        description="Trajectory, tool-argument, and LLM-judge evaluators over multi-step onboarding tasks.",
        max_concurrency=4,
    )

    print("\nAgent-eval experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nLook at how the lenses disagree: an agent can ace 'required_tools_used' "
          "yet fail 'trajectory_efficiency' or 'no_forbidden_tools'. That spread is "
          "the diagnosis.")


if __name__ == "__main__":
    main()
