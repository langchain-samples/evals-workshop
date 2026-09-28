"""Module 3 — evaluators for point-in-time (replay) evals.

Three evaluators, and the *contrast between two of them* is the lesson:

- ``matches_snapshot_calls``  — ground truth written from a trace: "the agent
  should call schedule_orientation with date=2026-06-15". **Absolute.** True only
  while the world still looks like it did when the trace was recorded.
- ``args_follow_tool_outputs`` — ground truth as an *invariant*: "the date the agent
  schedules must be the start_date the lookup returned." **Relative.** True in
  any world, recorded or drifted, and needs no recording at all.
- ``replay_fidelity``          — a health check on the replay itself: did the agent
  stay on the recorded path? Look at it before trusting anything else.

Run both experiments (`replay_eval.py`) and the pattern is stark: the absolute
check passes against the frozen world and fails against a drifted one — for an
agent that behaved *identically well* in both — while the invariant passes in
both. When you can state the expectation as an invariant, prefer it. When you
can't (the right answer really is a specific value), freeze the world.

Targets must return ``outputs["tool_calls"]`` (name + args), and — for the
invariant — ``outputs["calls"]``, the recorded-call view (name + args + output).
Replay runs also return ``outputs["replay"]``, the middleware's ``report()``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _same(a: Any, b: Any) -> bool:
    """Value equality, case/whitespace-insensitive for strings (tools normalize too)."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def _as_dict(output: Any) -> dict | None:
    """A tool output as a dict, whether it arrived as a dict or a JSON string."""
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def replay_fidelity(outputs: dict) -> dict:
    """Did the agent stay on the recorded path? Score 0 on any replay miss.

    A miss means the agent asked for something the recording can't answer — it
    took a different route than it did in the trace. That's not a replay bug; it
    is the behavior change you're hunting for, and it also means every other
    score on that example is measured against an error result, not the recorded
    world. Read this column first.
    """
    report = outputs.get("replay")
    if report is None:
        return {"key": "replay_fidelity", "score": 1, "comment": "Not a replay run."}
    misses = report.get("misses", [])
    if misses:
        names = ", ".join(f"{m['name']}({m['args']})" for m in misses)
        return {
            "key": "replay_fidelity",
            "score": 0,
            "comment": f"{len(misses)} call(s) had no recorded output: {names}.",
        }
    return {
        "key": "replay_fidelity",
        "score": 1,
        "comment": (
            f"All {report.get('calls', 0)} call(s) served from the recording"
            + (f" ({report['unused']} recorded call(s) never requested)." if report.get("unused") else ".")
        ),
    }


def matches_snapshot_calls(outputs: dict, reference_outputs: dict) -> dict:
    """ABSOLUTE ground truth: each expected call appears with exactly these args.

    Fraction of ``snapshot_calls`` found. Brittle by design — it is the check that
    decays when the world moves, and the reason replay exists.
    """
    expected = reference_outputs.get("snapshot_calls") or []
    if not expected:
        return {"key": "matches_snapshot_calls", "score": 1, "comment": "No snapshot calls expected."}
    calls = outputs.get("tool_calls", [])
    missing = []
    for want in expected:
        found = any(
            c["name"] == want["name"]
            and all(_same(c.get("args", {}).get(k), v) for k, v in want["args"].items())
            for c in calls
        )
        if not found:
            missing.append(f"{want['name']}({want['args']})")
    hit = len(expected) - len(missing)
    return {
        "key": "matches_snapshot_calls",
        "score": hit / len(expected),
        "comment": (
            "Every snapshot call was made with the recorded args."
            if not missing else f"Missing snapshot call(s): {'; '.join(missing)}."
        ),
    }


def args_follow_tool_outputs(outputs: dict, reference_outputs: dict) -> dict:
    """RELATIVE ground truth: an argument must come from what a tool returned.

    ``derived_args`` entries look like
    ``{"tool": "schedule_orientation", "arg": "date", "from_tool": "lookup_employee",
    "field": "start_date"}``: every call to ``tool`` must pass, as ``arg``, the
    ``field`` value that ``from_tool`` actually returned *in this run*. Holds in
    the recorded world and the drifted one, because it never names the value.

    Scores 0 if the expected call wasn't made or the source output can't be read —
    an invariant with nothing to check is not a pass.
    """
    specs = reference_outputs.get("derived_args") or []
    if not specs:
        return {"key": "args_follow_tool_outputs", "score": 1, "comment": "No derived args expected."}

    calls = outputs.get("calls", [])
    problems: list[str] = []
    for spec in specs:
        source = next(
            (c for c in calls if c["name"] == spec["from_tool"] and _as_dict(c.get("output"))),
            None,
        )
        if source is None:
            problems.append(f"no readable output from {spec['from_tool']}")
            continue
        truth = _as_dict(source["output"]).get(spec["field"])
        if truth is None:
            problems.append(f"{spec['from_tool']} returned no {spec['field']!r}")
            continue
        targets = [c for c in calls if c["name"] == spec["tool"]]
        if not targets:
            problems.append(f"{spec['tool']} was never called")
            continue
        for c in targets:
            got = c.get("args", {}).get(spec["arg"])
            if not _same(got, truth):
                problems.append(
                    f"{spec['tool']}.{spec['arg']}={got!r}, but {spec['from_tool']} "
                    f"returned {spec['field']}={truth!r}"
                )
    ok = not problems
    return {
        "key": "args_follow_tool_outputs",
        "score": 1 if ok else 0,
        "comment": "Every derived arg matches the tool output it came from." if ok else "; ".join(problems),
    }


REPLAY_EVALUATORS = [replay_fidelity, matches_snapshot_calls, args_follow_tool_outputs]


if __name__ == "__main__":
    ref = {
        "snapshot_calls": [{"name": "schedule_orientation",
                            "args": {"employee_id": "E1007", "date": "2026-06-15"}}],
        "derived_args": [{"tool": "schedule_orientation", "arg": "date",
                          "from_tool": "lookup_employee", "field": "start_date"}],
    }

    def run(start_date, scheduled_date):
        return {
            "tool_calls": [
                {"name": "lookup_employee", "args": {"name": "Jordan Lee"}},
                {"name": "schedule_orientation", "args": {"employee_id": "E1007", "date": scheduled_date}},
            ],
            "calls": [
                {"name": "lookup_employee", "args": {"name": "Jordan Lee"},
                 "output": json.dumps({"employee_id": "E1007", "start_date": start_date})},
                {"name": "schedule_orientation",
                 "args": {"employee_id": "E1007", "date": scheduled_date}, "output": "{}"},
            ],
        }

    # Frozen world: recorded start date, agent used it. Everything passes.
    frozen = run("2026-06-15", "2026-06-15")
    assert matches_snapshot_calls(frozen, ref)["score"] == 1
    assert args_follow_tool_outputs(frozen, ref)["score"] == 1

    # Drifted world, HEALTHY agent (follows the new date): the absolute check
    # fails an agent that did nothing wrong; the invariant does not.
    drifted = run("2026-07-06", "2026-07-06")
    assert matches_snapshot_calls(drifted, ref)["score"] == 0
    assert args_follow_tool_outputs(drifted, ref)["score"] == 1

    # A genuinely broken agent (ignores the lookup) is caught by the invariant.
    broken = run("2026-07-06", "2026-06-15")
    assert args_follow_tool_outputs(broken, ref)["score"] == 0

    # An invariant with nothing to check is not a pass.
    assert args_follow_tool_outputs({"calls": []}, ref)["score"] == 0
    no_action = run("2026-06-15", "2026-06-15")
    no_action["calls"] = no_action["calls"][:1]
    assert args_follow_tool_outputs(no_action, ref)["score"] == 0
    assert args_follow_tool_outputs(frozen, {})["score"] == 1

    # Strings compare case/whitespace-insensitively; partial credit is fractional.
    assert _same(" Executive ", "executive")
    two = {"snapshot_calls": ref["snapshot_calls"] + [{"name": "provision_equipment", "args": {}}]}
    assert matches_snapshot_calls(frozen, two)["score"] == 0.5

    # replay_fidelity: clean replay, a miss, and a non-replay run.
    assert replay_fidelity({"replay": {"misses": [], "calls": 2, "unused": 0}})["score"] == 1
    miss = replay_fidelity({"replay": {"misses": [{"name": "lookup_employee", "args": {"name": "X"}}],
                                       "calls": 1, "unused": 1}})
    assert miss["score"] == 0 and "lookup_employee" in miss["comment"]
    assert replay_fidelity({})["score"] == 1
    print("All replay evaluator self-tests passed.")
