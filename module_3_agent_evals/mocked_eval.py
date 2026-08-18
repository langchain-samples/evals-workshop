"""Module 3 — run the tool-failure experiment with dataset-driven mocks.

`run_eval.py` evaluates the agent against its real (always-succeeding) tools.
This runs the *same agent* against mocked tools whose outputs come from the
dataset, so we can score behavior that is otherwise unreachable: what does the
agent do when provisioning fails, when the employee doesn't exist, when one of
two actions succeeds?

The pattern:

    inputs["tool_outputs"]  ->  DatasetDrivenMockMiddleware  ->  build_agent(middleware=[...])

The agent's own code is untouched. Only the world around it is swapped, which
is what keeps this an eval of the agent rather than an eval of a different app.

Run:  python module_3_agent_evals/mocked_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import experiment_metadata, require_langsmith
from hr_agent import DatasetDrivenMockMiddleware, run_agent
from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response
from module_3_agent_evals.mock_datasets import DATASET_NAME, ensure_dataset
from module_3_agent_evals.tool_evals import FAILURE_EVALUATORS
from module_3_agent_evals.trajectory_evals import no_forbidden_tools, required_tools_used


def target(inputs: dict) -> dict:
    """Run the agent with this example's mocked tool outputs.

    A fresh middleware per example — it holds the mock table and a call log, so
    sharing one across examples would leak state between test cases.

    ``strict=False``: if the agent calls a tool this example didn't mock, fall
    through to the real tool rather than erroring. For these scenarios that's the
    forgiving choice; flip it to True when an unmocked call should be a hard
    failure (e.g. in CI, where hitting a real system would be a bug).
    """
    mock = DatasetDrivenMockMiddleware(inputs.get("tool_outputs", {}), strict=False)
    result = run_agent(inputs["question"], middleware=[mock])
    return {
        "trajectory": extract_trajectory(result),
        "tool_calls": extract_tool_calls(result),
        "answer": final_response(result),
        # What the middleware actually intercepted — handy when a score looks
        # wrong and you want to know which tools really ran.
        "mocked_calls": mock.calls_made,
    }


def main() -> None:
    require_langsmith()
    client = Client()
    ensure_dataset(client)

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[*FAILURE_EVALUATORS, required_tools_used, no_forbidden_tools],
        experiment_prefix="module-3-tool-failures",
        metadata=experiment_metadata(module=3, suite="tool-failures", mocked=True),
        description="Agent behavior under mocked tool failures (outage, unknown employee, partial failure).",
        max_concurrency=3,
    )

    print("\nMocked tool-failure experiment complete:")
    print(getattr(results, "experiment_name", "(see https://smith.langchain.com)"))
    print("\nThe interesting column is 'reports_tool_failure': an agent that scores 0 "
          "told the user an action succeeded when the tool said it failed. You cannot "
          "catch that without mocking, because the real tools never fail.")


if __name__ == "__main__":
    main()
