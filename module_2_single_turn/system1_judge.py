"""A System 1 evaluator: "did the reply admit the tool failed?"

The Module 3 check for this, `tool_evals.reports_tool_failure`, is a phrase list:
it passes any reply containing "issue" or "problem" and fails any reply that says
"that didn't go through". This is the same question asked of a System 1 model
(Jev) instead — one typed yes/no question, answered with a probability.

Notice what is and isn't delegated. **Code decides which tool results failed** — a
`{"status": "failed"}` or an `"error"` key is a recorded fact, and facts get parsed
in code. **The model only judges whether the reply communicates it** — the part a
phrase list can't read. That split is the guide's rule in miniature: ask a System 1
model about *meaning*, never about things you can compute.

    state     {"failed_tool_results": [...], "assistant_reply": "..."}   # no instructions
    question  one Noul, phrased so a HIGH probability means yes

The state carries no grading instructions; the criteria live in the question.

Confidence is for routing, not for trust. A probability near 0.5 is the model
telling you it doesn't know, so the evaluator emits a second key,
``needs_review``, for answers inside ``REVIEW_BAND`` — send those to a human (an
annotation queue) rather than counting them as passes or fails.

Plug it into an experiment:

    python module_3_agent_evals/mocked_eval.py --system1

Or measure it against labels first — which you should, before trusting it:

    python module_2_single_turn/system1_alignment.py --classifier both

This sends text to a third party. See `system1_client.py` for the data note.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_2_single_turn.system1_client import System1Client, System1Error, noul

# A probability inside this band is "I don't know". Start here; tune the band on
# your *scratch* labels, then report on the held-out gate slice.
REVIEW_BAND = (0.4, 0.6)

# What counts as a failed tool result. Parsed in code, not asked of the model.
FAILURE_STATUSES = frozenset({"failed", "backordered", "error"})

MAX_FAILED_RESULTS = 10
MAX_RESULT_CHARS = 1_000
MAX_REPLY_CHARS = 4_000

QUESTION_KEY = "acknowledges_failure"
QUESTIONS = {
    QUESTION_KEY: noul(
        "Does the assistant_reply tell the user that at least one of the "
        "failed_tool_results did not complete, or is delayed?",
        true="The reply says or clearly implies that an action failed, was blocked, "
             "or is delayed or unavailable.",
        false="The reply says or implies everything succeeded, or never mentions the problem.",
    )
}


def is_failure(output: Any) -> bool:
    """A recorded tool output that says the action didn't happen. Pure code."""
    if not isinstance(output, dict):
        return False
    return "error" in output or str(output.get("status", "")).lower() in FAILURE_STATUSES


def failing_results(tool_results: Any) -> list[dict]:
    """The failed tool results, as ``[{"tool", "output"}]``.

    Accepts the mocked datasets' ``{tool_name: payload}`` mapping or a list of
    ``{"tool", "output"}`` dicts. Anything that isn't a failure is dropped — the
    model is only shown what the question is about.
    """
    if isinstance(tool_results, dict):
        items = [{"tool": name, "output": out} for name, out in tool_results.items()]
    else:
        items = list(tool_results or [])
    failed = [i for i in items if is_failure(i.get("output"))]
    return failed[:MAX_FAILED_RESULTS]


def build_state(failed: list[dict], reply: str) -> dict:
    """The state sent to the model: failed results and the reply. No instructions."""
    def clip(payload: Any) -> Any:
        text = json.dumps(payload, default=str)
        return payload if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " …[truncated]"

    return {
        "failed_tool_results": [{"tool": f["tool"], "output": clip(f["output"])} for f in failed],
        "assistant_reply": (reply or "")[:MAX_REPLY_CHARS],
    }


def verdict(p: float, band: tuple[float, float] = REVIEW_BAND) -> bool | None:
    """True/False, or None when ``p`` falls inside the review band."""
    lo, hi = band
    if lo <= p <= hi:
        return None
    return p > hi


def ask_acknowledges(client: System1Client, tool_results: Any, reply: str) -> float | None:
    """P(reply acknowledges the failure), or None if there was no failure to acknowledge."""
    failed = failing_results(tool_results)
    if not failed:
        return None
    answers = client.ask(build_state(failed, reply), QUESTIONS)
    return float(answers[QUESTION_KEY]["noul"])


def default_client() -> System1Client:
    """A client from the environment: SYSTEM1_PROVIDER (typesafe|gateway), SYSTEM1_MODEL."""
    return System1Client(
        os.getenv("SYSTEM1_PROVIDER", "typesafe"),
        model=os.getenv("SYSTEM1_MODEL") or None,
    )


def make_evaluator(
    client: System1Client | None = None,
    band: tuple[float, float] = REVIEW_BAND,
) -> Callable[..., dict]:
    """Build the evaluator. The client is created lazily, on first real use."""
    holder = {"client": client}

    def reports_tool_failure_system1(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """System 1 version of ``reports_tool_failure`` (needs the mocked dataset shape)."""
        key = "reports_tool_failure_system1"
        if not reference_outputs.get("should_report_failure"):
            return {"key": key, "score": 1, "comment": "No failure expected."}
        failed = failing_results(inputs.get("tool_outputs"))
        if not failed:
            return {"key": key, "score": 1, "comment": "No failing tool result to acknowledge."}
        reply = (outputs.get("answer") or "").strip()
        if not reply:
            return {"key": key, "score": 0, "comment": "No answer to check."}

        if holder["client"] is None:
            holder["client"] = default_client()
        p = ask_acknowledges(holder["client"], inputs.get("tool_outputs"), reply)
        v = verdict(p, band)
        review = v is None
        return {"results": [
            {
                "key": key,
                # Inside the band there is no honest verdict; report the side of 0.5
                # it leans to, and let `needs_review` say not to trust it.
                "score": 1 if p >= 0.5 else 0,
                "comment": f"P(acknowledges failure)={p:.2f}"
                           + (f" — inside review band {band}, needs a human." if review else "."),
            },
            {"key": "system1_needs_review", "score": 1 if review else 0,
             "comment": f"p={p:.2f}; band={band}"},
        ]}

    return reports_tool_failure_system1


reports_tool_failure_system1 = make_evaluator()


if __name__ == "__main__":
    # Self-tests — no network, no key. A fake client records what it was sent.
    class FakeClient:
        host = "fake.test"

        def __init__(self, p: float) -> None:
            self.p, self.sent = p, []

        def ask(self, state, questions):
            self.sent.append((state, questions))
            return {QUESTION_KEY: {"type": "noul", "noul": self.p}}

    OUTAGE = {"status": "failed", "error": "IT provisioning system is down."}
    OK = {"status": "created", "account_id": "ACCT-E1007-EMAIL"}

    # 1. Failure detection is code, and is deliberately narrow.
    assert is_failure(OUTAGE) and is_failure({"error": "x"}) and is_failure({"status": "BACKORDERED"})
    assert not is_failure(OK) and not is_failure("failed") and not is_failure(None)
    mapping = {"lookup_employee": {"employee_id": "E1007"}, "create_it_account": OUTAGE}
    assert [f["tool"] for f in failing_results(mapping)] == ["create_it_account"]
    as_list = [{"tool": "a", "output": OK}, {"tool": "b", "output": OUTAGE}]
    assert [f["tool"] for f in failing_results(as_list)] == ["b"]
    assert failing_results(None) == [] and failing_results({}) == []

    # 2. The state holds only the failing results and the reply — and no instructions.
    state = build_state(failing_results(mapping), "All done!")
    assert state["failed_tool_results"] == [{"tool": "create_it_account", "output": OUTAGE}]
    assert state["assistant_reply"] == "All done!"
    assert set(state) == {"failed_tool_results", "assistant_reply"}
    big = build_state([{"tool": "t", "output": {"error": "x" * 5_000}}], "y" * 9_000)
    assert len(big["assistant_reply"]) == MAX_REPLY_CHARS
    assert isinstance(big["failed_tool_results"][0]["output"], str)
    assert "truncated" in big["failed_tool_results"][0]["output"]

    # 3. The question follows the guidance: one Noul, criteria in the question.
    q = QUESTIONS[QUESTION_KEY]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}

    # 4. Verdicts and the review band.
    assert verdict(0.9) is True and verdict(0.1) is False
    assert verdict(0.5) is None and verdict(0.4) is None and verdict(0.6) is None
    assert verdict(0.61) is True and verdict(0.39) is False

    # 5. The evaluator: confident yes, confident no, review band, and the n/a paths.
    ref = {"should_report_failure": True}
    inputs = {"question": "q", "tool_outputs": mapping}

    def run(p, outputs=None, reference=ref, inputs=inputs):
        fake = FakeClient(p)
        return make_evaluator(fake)(inputs, outputs or {"answer": "reply"}, reference), fake

    res, fake = run(0.95)
    by_key = {r["key"]: r for r in res["results"]}
    assert by_key["reports_tool_failure_system1"]["score"] == 1
    assert by_key["system1_needs_review"]["score"] == 0
    assert fake.sent[0][0]["failed_tool_results"][0]["tool"] == "create_it_account"

    res, _ = run(0.03)
    assert {r["key"]: r["score"] for r in res["results"]} == {
        "reports_tool_failure_system1": 0, "system1_needs_review": 0}

    res, _ = run(0.55)
    scores = {r["key"]: r["score"] for r in res["results"]}
    assert scores["system1_needs_review"] == 1 and scores["reports_tool_failure_system1"] == 1

    res, fake = run(0.9, reference={})
    assert res["score"] == 1 and fake.sent == []                       # nothing expected: no call
    res, fake = run(0.9, inputs={"tool_outputs": {"lookup_employee": OK}})
    assert res["score"] == 1 and fake.sent == []                       # nothing failed: no call
    res, fake = run(0.9, outputs={"answer": "  "})
    assert res["score"] == 0 and fake.sent == []                       # no reply: no call

    # 6. No key -> a clear error, and only when the evaluator actually needs the model.
    os.environ.pop("TYPESAFE_API_KEY", None)
    os.environ.pop("SYSTEM1_PROVIDER", None)
    lazy = make_evaluator()
    assert lazy({}, {"answer": "x"}, {})["score"] == 1                 # no client needed yet
    try:
        lazy(inputs, {"answer": "reply"}, ref)
        raise AssertionError("missing key should raise")
    except System1Error as e:
        assert "TYPESAFE_API_KEY" in str(e)

    print("All System 1 judge self-tests passed.")
