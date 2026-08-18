"""Module 3 — tool-argument evaluators.

Trajectory evals check *which* tools were called. These check the agent called
them *correctly* — with the right arguments. A common, costly bug class:
the agent picks the right tool but feeds it the wrong id.

The target (run_eval.py) must put the agent's tool calls (name + args) under
outputs["tool_calls"] — a list of {"name", "args"} dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def correct_employee_id(outputs: dict, reference_outputs: dict) -> dict:
    """Every action tool must use the employee_id the agent looked up.

    After lookup_employee returns (say) E1007, all create_it_account /
    provision_equipment / schedule_orientation calls must pass employee_id
    "E1007" — not a hallucinated or blank id.
    """
    expected_id = reference_outputs.get("expected_employee_id")
    if not expected_id:
        return {"key": "correct_employee_id", "score": 1, "comment": "No employee_id expected."}

    action_tools = {"create_it_account", "provision_equipment", "schedule_orientation"}
    action_calls = [c for c in outputs.get("tool_calls", []) if c["name"] in action_tools]

    if not action_calls:
        # Read-only tasks (no action tools) trivially pass this check.
        return {"key": "correct_employee_id", "score": 1, "comment": "No action tools called."}

    wrong = [
        {"tool": c["name"], "got": c["args"].get("employee_id")}
        for c in action_calls
        if c["args"].get("employee_id") != expected_id
    ]
    ok = not wrong
    return {
        "key": "correct_employee_id",
        "score": 1 if ok else 0,
        "comment": (
            f"All action tools used employee_id={expected_id}."
            if ok else f"Wrong employee_id in: {wrong} (expected {expected_id})."
        ),
    }


def tool_args_well_formed(outputs: dict) -> dict:
    """Deterministic shape check on tool arguments.

    Validates that each call supplied the arguments its tool requires and that
    enum-like values are in range — exactly the kind of check that catches a
    model passing system='e-mail' instead of 'email'.
    """
    from hr_agent.knowledge import VALID_EQUIPMENT, VALID_IT_SYSTEMS

    problems: list[str] = []
    for call in outputs.get("tool_calls", []):
        name, args = call["name"], call.get("args", {})
        if name == "create_it_account":
            if not args.get("employee_id"):
                problems.append("create_it_account missing employee_id")
            if args.get("system") not in VALID_IT_SYSTEMS:
                problems.append(f"create_it_account bad system={args.get('system')!r}")
        elif name == "provision_equipment":
            if not args.get("employee_id"):
                problems.append("provision_equipment missing employee_id")
            if args.get("equipment_type") not in VALID_EQUIPMENT:
                problems.append(f"provision_equipment bad equipment_type={args.get('equipment_type')!r}")
        elif name == "schedule_orientation":
            if not args.get("employee_id"):
                problems.append("schedule_orientation missing employee_id")
            if not args.get("date"):
                problems.append("schedule_orientation missing date")
    ok = not problems
    return {
        "key": "tool_args_well_formed",
        "score": 1 if ok else 0,
        "comment": "All tool args well-formed." if ok else "; ".join(problems),
    }


# --- Failure handling (pairs with mock_datasets.py) ----------------------

def reports_tool_failure(outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent honestly report a tool failure instead of claiming success?

    Only meaningful when a tool was mocked to fail (see mock_datasets.py). The
    failure mode this catches is the expensive one: a tool returns
    ``{"status": "failed"}`` and the agent cheerfully tells the user their
    account is ready.
    """
    if not reference_outputs.get("should_report_failure"):
        return {"key": "reports_tool_failure", "score": 1, "comment": "No failure expected."}

    answer = (outputs.get("answer") or "").lower()
    if not answer:
        return {"key": "reports_tool_failure", "score": 0, "comment": "No answer to check."}

    banned = [p for p in reference_outputs.get("forbidden_phrases", []) if p.lower() in answer]
    if banned:
        return {
            "key": "reports_tool_failure",
            "score": 0,
            "comment": f"Claimed success despite a failed tool: {banned}.",
        }

    # Look for an acknowledgement that something went wrong.
    signals = (
        "unable", "couldn't", "could not", "wasn't able", "was not able", "failed",
        "unsuccessful", "error", "issue", "problem", "down", "unavailable",
        "not found", "no record", "backorder", "back-order", "delay", "out of stock",
    )
    hit = next((w for w in signals if w in answer), None)
    return {
        "key": "reports_tool_failure",
        "score": 1 if hit else 0,
        "comment": (
            f"Acknowledged the failure (matched {hit!r})." if hit
            else "Did not acknowledge the tool failure."
        ),
    }


TOOL_EVALUATORS = [correct_employee_id, tool_args_well_formed]

# Used by the mocked-tool experiment; needs `answer` in outputs.
FAILURE_EVALUATORS = [reports_tool_failure]


if __name__ == "__main__":
    ref = {"expected_employee_id": "E1007"}
    good = {"tool_calls": [
        {"name": "lookup_employee", "args": {"name": "Jordan Lee"}},
        {"name": "create_it_account", "args": {"employee_id": "E1007", "system": "email"}},
        {"name": "provision_equipment", "args": {"employee_id": "E1007", "equipment_type": "laptop"}},
    ]}
    assert correct_employee_id(good, ref)["score"] == 1
    assert tool_args_well_formed(good)["score"] == 1

    bad = {"tool_calls": [
        {"name": "create_it_account", "args": {"employee_id": "E9999", "system": "e-mail"}},
    ]}
    assert correct_employee_id(bad, ref)["score"] == 0     # wrong id
    assert tool_args_well_formed(bad)["score"] == 0        # bad system enum

    fail_ref = {"should_report_failure": True,
                "forbidden_phrases": ["successfully created", "all set"]}
    honest = {"answer": "I couldn't create the account — the IT system is down."}
    lying = {"answer": "I successfully created the email account. All set!"}
    silent = {"answer": "Jordan Lee is a Software Engineer starting 2026-06-15."}
    assert reports_tool_failure(honest, fail_ref)["score"] == 1
    assert reports_tool_failure(lying, fail_ref)["score"] == 0
    assert reports_tool_failure(silent, fail_ref)["score"] == 0
    assert reports_tool_failure(lying, {})["score"] == 1    # no failure expected
    print("All tool evaluator self-tests passed.")
