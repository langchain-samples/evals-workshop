"""Module 4 — per-example eval tests via the langsmith[pytest] plugin.

This is the "per-example gate" pattern: each dataset example becomes a pytest
test case, so CI shows you *exactly which* example regressed, not just an
aggregate. The `@pytest.mark.langsmith` decorator also logs each case (inputs,
outputs, feedback) to LangSmith so the run shows up as a test experiment.

Run:
    pytest module_4_ci/test_evals.py -v --langsmith-output

We assert only on the DETERMINISTIC, safety-critical metrics here — they're
cheap, stable, and the right thing to *block a merge* on. Fuzzy LLM-judge
metrics are better tracked as trends via the aggregate gate (ci_gate.py) than
as hard per-example asserts, because a single judge call can be noisy.

Two things conftest.py sets up for these tests:
  - **Split filtering.** Everything runs here EXCEPT the held-out splits
    (`train`), which are scratch space for tuning prompts and judges — gating
    on them would mean tuning against your own gate. Note the direction: we
    exclude scratch rather than include `test`, so an example nobody assigned a
    split to still gets gated. See `ci_gate.resolve_data` for why.
  - **Response caching.** Locally, model API calls are recorded to
    `fixtures/cassettes/` and replayed, so re-running is fast and free. In CI
    (`CI=true`) caching is off — a gate replaying stale responses can't detect a
    real regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import testing as t

from conftest import CACHE_MARK
from config import experiment_metadata
from hr_agent import run_agent
from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response
from module_2_single_turn.datasets import EXAMPLES as SINGLE_TURN_EXAMPLES
from module_2_single_turn.deterministic_evals import (
    mentions_required_facts,
    response_not_empty,
)
from module_3_agent_evals.datasets import EXAMPLES as AGENT_EXAMPLES
from module_3_agent_evals.tool_evals import correct_employee_id
from module_3_agent_evals.trajectory_evals import no_forbidden_tools, required_tools_used
from module_4_ci.ci_gate import HELD_OUT_SPLITS

# Gate on everything except the held-out splits — see the module docstring.
# `e.get("split")` is None for an example that never declared one; None is not
# in HELD_OUT_SPLITS, so it gets gated rather than silently skipped.
def _gated(examples: list[dict]) -> list[dict]:
    return [e for e in examples if e.get("split") not in HELD_OUT_SPLITS]


SINGLE_TURN_GATED = _gated(SINGLE_TURN_EXAMPLES)
AGENT_GATED = _gated(AGENT_EXAMPLES)

# Tag the pytest experiment with commit/branch/model, same as the aggregate gate.
LS_MARK = {**CACHE_MARK, "experiment_metadata": experiment_metadata(suite="ci-pytest")}


def _safe(fn, *args, **kwargs):
    """Call a langsmith.testing logger, ignoring the error raised when tracing
    is disabled. Lets these tests run as plain `pytest` (asserts only) OR as
    `LANGSMITH_TRACING=true pytest --langsmith-output` (asserts + logged runs).
    """
    try:
        fn(*args, **kwargs)
    except ValueError:
        pass  # "log_* should only be called ... with tracing enabled"


# --- Single-turn: every policy answer must be non-empty and hit its facts ---
@pytest.mark.langsmith(**LS_MARK)
@pytest.mark.parametrize("example", SINGLE_TURN_GATED, ids=lambda e: e["outputs"]["policy_topic"])
def test_single_turn_answer(example):
    inputs, reference = example["inputs"], example["outputs"]
    _safe(t.log_inputs, inputs)

    outputs = {"answer": final_response(run_agent(inputs["question"]))}
    _safe(t.log_outputs, outputs)

    facts = mentions_required_facts(outputs, reference)
    _safe(t.log_feedback, key=facts["key"], score=facts["score"])
    _safe(t.log_feedback, key="response_not_empty", score=response_not_empty(outputs)["score"])

    assert response_not_empty(outputs)["score"] == 1, "Answer was empty."
    # Require at least half the facts present — tune for your bar.
    assert facts["score"] >= 0.5, facts["comment"]


# --- Agent: never call a forbidden tool; use the right id; do required steps ---
@pytest.mark.langsmith(**LS_MARK)
@pytest.mark.parametrize("example", AGENT_GATED, ids=lambda e: e["inputs"]["question"][:40])
def test_agent_trajectory(example):
    inputs, reference = example["inputs"], example["outputs"]
    _safe(t.log_inputs, inputs)

    result = run_agent(inputs["question"])
    outputs = {
        "trajectory": extract_trajectory(result),
        "tool_calls": extract_tool_calls(result),
    }
    _safe(t.log_outputs, outputs)

    forbidden = no_forbidden_tools(outputs, reference)
    employee = correct_employee_id(outputs, reference)
    required = required_tools_used(outputs, reference)
    for r in (forbidden, employee, required):
        _safe(t.log_feedback, key=r["key"], score=r["score"])

    # Safety + correctness asserts that SHOULD block a merge.
    assert forbidden["score"] == 1, forbidden["comment"]
    assert employee["score"] == 1, employee["comment"]
    assert required["score"] >= 0.9, required["comment"]
