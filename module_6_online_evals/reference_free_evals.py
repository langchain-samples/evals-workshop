"""Module 6 — reference-free evaluators (for online evals on live traces).

Offline experiments (Modules 1–5) have a `reference_outputs` to grade against —
the dataset's ground truth. **Online** evals don't: a production trace is a real
user's question the agent has never seen, with no pre-written "right answer". So
online evaluators must judge an output using only what's *in the trace itself*
(the question, the answer, the tools the agent called) plus knowledge you already
hold (your policy corpus). Anything that needs a per-example reference can't run
online.

That constraint is the whole point of this file. Every evaluator here takes only
`(inputs, outputs)` — never `reference_outputs`:

  - response_not_empty   — deterministic: did we answer at all?
  - not_deflected        — deterministic heuristic: did the agent punt instead of help?
  - groundedness         — LLM judge: is every claim supported by the policy corpus?
  - professional_tone    — LLM judge: warm, clear, professional for a new hire?

The LLM judges follow Module 2's pattern (Pydantic structured output, temp-0
judge, `reasoning` surfaced as the comment) — see
module_2_single_turn/llm_judge_evals.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Deterministic, reference-free checks --------------------------------

def response_not_empty(inputs: dict, outputs: dict) -> dict:
    """Did the agent produce a non-trivial answer at all? (No reference needed.)"""
    answer = (outputs.get("answer") or "").strip()
    ok = len(answer) >= 10
    return {
        "key": "response_not_empty",
        "score": 1 if ok else 0,
        "comment": f"Answer length = {len(answer)} chars.",
    }


# Phrases that signal the agent gave up / refused instead of trying to help.
# A reference-free quality signal: in production, watch the *rate* of these —
# a spike usually means a prompt/model/tool change broke something upstream.
_DEFLECTION_MARKERS = (
    "i can't help",
    "i cannot help",
    "i'm not able to",
    "i am not able to",
    "i don't know",
    "i do not know",
    "unable to assist",
    "contact hr",
    "reach out to hr",
    "please contact",
)


def not_deflected(inputs: dict, outputs: dict) -> dict:
    """Heuristic: did the agent actually attempt an answer, or just punt?

    Deterministic and reference-free. Not a correctness check — a deflection can
    be the *right* call for a truly out-of-scope question. It's a rate signal:
    track how often the agent bails, and alert when that rate climbs.
    """
    answer = (outputs.get("answer") or "").lower()
    hit = next((m for m in _DEFLECTION_MARKERS if m in answer), None)
    return {
        "key": "not_deflected",
        "score": 0 if hit else 1,
        "comment": f"Deflection marker: {hit!r}." if hit else "Agent attempted an answer.",
    }


# --- LLM-as-judge, reference-free ----------------------------------------
# Imports for these live inside the functions so the deterministic checks (and
# this file's self-tests) need no model dependency to import.

def groundedness(inputs: dict, outputs: dict) -> dict:
    """LLM judge: is every factual claim supported by the HR policy corpus?

    Reference-free anti-hallucination check. Offline groundedness (Module 2)
    feeds the judge the ONE policy the example is about. Online we don't know the
    topic ahead of time, so we hand the judge the *entire* policy corpus and ask
    whether each claim is supported by ANY of it. Same idea, no per-example
    reference required.
    """
    from pydantic import BaseModel, Field

    from config import get_judge
    from hr_agent.knowledge import HR_POLICIES

    class GroundednessGrade(BaseModel):
        reasoning: str = Field(description="Explain whether every claim is supported.")
        is_grounded: bool = Field(description="True if all claims are supported by the corpus.")

    corpus = "\n".join(f"[{topic}] {text}" for topic, text in HR_POLICIES.items())
    judge = get_judge().with_structured_output(GroundednessGrade)
    prompt = f"""You are checking an HR assistant's answer for hallucination.

Official HR policy corpus (the ONLY source of truth):
\"\"\"{corpus}\"\"\"

User question: {inputs.get('question', '')}
Assistant answer: {outputs.get('answer', '')}

Is every factual claim in the answer supported by the corpus above? Numbers,
dates, or rules not present in the corpus mean it is NOT grounded. An answer that
correctly declines an out-of-scope question (makes no policy claims) IS grounded."""
    grade = judge.invoke(prompt)
    return {
        "key": "groundedness",
        "score": 1 if grade.is_grounded else 0,
        "comment": grade.reasoning,
    }


def professional_tone(inputs: dict, outputs: dict) -> dict:
    """LLM judge: warm, clear, and professional for a new hire? (Reference-free.)"""
    from pydantic import BaseModel, Field

    from config import get_judge

    class ToneGrade(BaseModel):
        reasoning: str = Field(description="Explain the tone judgment.")
        score: int = Field(description="1=poor, 2=acceptable, 3=excellent.", ge=1, le=3)

    judge = get_judge().with_structured_output(ToneGrade)
    prompt = f"""Rate the tone of this HR assistant reply to a new employee.

Reply: {outputs.get('answer', '')}

Score 1 (poor: cold, confusing, or unprofessional), 2 (acceptable), or
3 (excellent: warm, clear, concise, professional)."""
    grade = judge.invoke(prompt)
    # Normalize 1-3 to 0-1 so it sits on the same scale as the other metrics.
    return {
        "key": "professional_tone",
        "score": (grade.score - 1) / 2,
        "comment": f"Raw {grade.score}/3. {grade.reasoning}",
    }


# Deterministic bundle (no API key) and the full online bundle (adds LLM judges).
DETERMINISTIC_ONLINE_EVALUATORS = [response_not_empty, not_deflected]
ONLINE_EVALUATORS = [response_not_empty, not_deflected, groundedness, professional_tone]


if __name__ == "__main__":
    # Self-test the deterministic, reference-free evaluators — no API key needed.
    # Same habit as Module 2: validate the evaluator before you trust it.
    assert response_not_empty({}, {"answer": "You accrue 15 vacation days a year."})["score"] == 1
    assert response_not_empty({}, {"answer": ""})["score"] == 0
    assert response_not_empty({}, {})["score"] == 0

    assert not_deflected({}, {"answer": "You get a 4% 401(k) match after 60 days."})["score"] == 1
    assert not_deflected({}, {"answer": "I can't help with that — please contact HR."})["score"] == 0
    assert not_deflected({}, {"answer": "I don't know, sorry."})["score"] == 0

    print("All reference-free deterministic evaluator self-tests passed.")
