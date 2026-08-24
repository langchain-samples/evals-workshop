"""Module 3 — trajectory evaluators.

These score the *path* the agent took (its ordered tool calls), not just the
final answer. They're deterministic: we compare the agent's trajectory to the
expected one from the dataset.

Four complementary lenses, because "right path" means different things:
  - exact_match        : strictest — same tools, same order, same count.
  - required_tools     : did it do the essential steps? (order-agnostic subset)
  - no_forbidden_tools : did it AVOID dangerous/irrelevant tools?
  - efficiency         : did it take extra/wasteful steps?

The target (run_eval.py) must put the agent's trajectory under
outputs["trajectory"] — a list of tool-name strings.

**Scope: one agent run.** Every example here is a single user message, so a
trajectory is the tool calls inside *one turn* (multi-step, single-turn) — not
a conversation thread. That matters, because these four evaluators are written
for that scope and do not all survive being pointed at a whole thread. See
`required_tools_used` for the one that bites.
"""

from __future__ import annotations


def trajectory_exact_match(outputs: dict, reference_outputs: dict) -> dict:
    """Strict: the tool sequence equals the expected sequence exactly."""
    actual = outputs.get("trajectory", [])
    expected = reference_outputs.get("expected_trajectory", [])
    match = actual == expected
    return {
        "key": "trajectory_exact_match",
        "score": 1 if match else 0,
        "comment": f"expected={expected} actual={actual}",
    }


def required_tools_used(outputs: dict, reference_outputs: dict) -> dict:
    """Did the agent call every required tool at least once (order-agnostic)?

    Fractional: fraction of required tools that appeared. More forgiving than
    exact match — good for tasks where order doesn't strictly matter.

    This is a **task-completion** metric, not a safety precondition — and the
    difference is easy to miss once you go multi-turn. "Was this tool called at
    least once somewhere?" is the right question for "did the work get done".
    It is the wrong question for "was every risky claim licensed by a fresh
    lookup": an agent that calls the lookup in turn 1 and then makes three
    unbacked assertions in turns 2-4 scores a perfect 1.0 here.

    For a precondition, you want a *conditional, per-turn* check instead — the
    shape of `no_forbidden_tools` (hard fail on one violation) combined with
    the trigger condition in `tool_evals.correct_employee_id` (only applies
    when the risky action actually occurred). The structural fix is to make the
    eval example *be* a turn, with prior messages as input context, so the run
    boundary and the assertion boundary are the same thing.
    """
    actual = set(outputs.get("trajectory", []))
    required = reference_outputs.get("required_tools", [])
    if not required:
        return {"key": "required_tools_used", "score": 1, "comment": "No required tools."}
    present = [t for t in required if t in actual]
    missing = [t for t in required if t not in actual]
    return {
        "key": "required_tools_used",
        "score": len(present) / len(required),
        "comment": f"{len(present)}/{len(required)} required tools used. Missing: {missing or 'none'}.",
    }


def no_forbidden_tools(outputs: dict, reference_outputs: dict) -> dict:
    """Safety check: the agent must NOT call any forbidden tool.

    Catches over-eager agents — e.g. provisioning equipment when only asked a
    read-only benefits question. A single violation fails the check.
    """
    actual = set(outputs.get("trajectory", []))
    forbidden = reference_outputs.get("forbidden_tools", [])
    violations = [t for t in forbidden if t in actual]
    return {
        "key": "no_forbidden_tools",
        "score": 0 if violations else 1,
        "comment": f"Forbidden tools called: {violations}" if violations else "No forbidden tools called.",
    }


def trajectory_efficiency(outputs: dict, reference_outputs: dict) -> dict:
    """How many extra steps beyond the expected count? Fewer is better.

    Score = expected_len / actual_len (capped at 1.0). 1.0 means no wasted
    steps; 0.5 means the agent took twice as many tool calls as needed.
    """
    actual = outputs.get("trajectory", [])
    expected = reference_outputs.get("expected_trajectory", [])
    if not actual:
        return {"key": "trajectory_efficiency", "score": 0, "comment": "Agent made no tool calls."}
    score = min(1.0, len(expected) / len(actual))
    extra = len(actual) - len(expected)
    return {
        "key": "trajectory_efficiency",
        "score": score,
        "comment": f"expected {len(expected)} steps, took {len(actual)} (extra: {extra}).",
    }


TRAJECTORY_EVALUATORS = [
    trajectory_exact_match,
    required_tools_used,
    no_forbidden_tools,
    trajectory_efficiency,
]


if __name__ == "__main__":
    # Self-tests — no API needed.
    ref = {
        "expected_trajectory": ["lookup_employee", "create_it_account", "provision_equipment"],
        "required_tools": ["lookup_employee", "create_it_account", "provision_equipment"],
        "forbidden_tools": ["schedule_orientation"],
    }
    perfect = {"trajectory": ["lookup_employee", "create_it_account", "provision_equipment"]}
    assert trajectory_exact_match(perfect, ref)["score"] == 1
    assert required_tools_used(perfect, ref)["score"] == 1.0
    assert no_forbidden_tools(perfect, ref)["score"] == 1
    assert trajectory_efficiency(perfect, ref)["score"] == 1.0

    wandered = {"trajectory": ["lookup_employee", "lookup_employee", "create_it_account",
                               "provision_equipment", "schedule_orientation"]}
    assert trajectory_exact_match(wandered, ref)["score"] == 0
    assert no_forbidden_tools(wandered, ref)["score"] == 0          # called forbidden tool
    assert trajectory_efficiency(wandered, ref)["score"] < 1.0      # extra steps
    print("All trajectory evaluator self-tests passed.")
