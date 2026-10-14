"""Module 7 — an LLM judge you can align with few-shot examples.

A generic LLM judge grades against a generic notion of "good". But your team has a
*house style* the model can't guess from a one-line rubric — and when the judge's
idea of good drifts from your reviewers', its scores stop being trustworthy. The
fix that needs no fine-tuning: show the judge a handful of **human-labeled
examples** in its prompt. Few-shot examples pull the judge toward your humans'
actual decisions.

This file defines one judge — "does the answer meet our HR house style?" — that
works in two modes:
  - zero-shot:  rubric only (the naive judge)
  - few-shot:   rubric + human-labeled examples (the aligned judge)

`judge_alignment.py` runs both against a held-out set of human labels and shows the
few-shot version agreeing with humans more often.

The catch that makes few-shot matter: the rubric is **deliberately
underspecified**. It tells the judge there's a house convention but not what it is
— exactly the situation you're in with any real team style that lives in
reviewers' heads, not in a doc. A zero-shot judge falls back on generic "is this a
good reply?" and misses the actual convention. The few-shot examples *are* the
convention, so they pull the judge onto it.

(For the curious: the convention these labels encode is "every reply must end with
an explicit invitation to follow up." We don't tell the judge that — the whole
point is that it has to learn it from the labeled examples.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from config import get_judge

RUBRIC = """You grade whether an HR assistant reply meets our team's house style
for new-hire messages.

Our team has a specific convention for how these replies should be written. Reply
PASS if it follows our house style and FAIL if it doesn't. Use your best judgment."""


class StyleGrade(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the judgment.")
    meets_house_style: bool = Field(description="True if the reply meets the house style.")


def _format_examples(examples: list[dict]) -> str:
    """Render labeled examples as a few-shot block for the prompt."""
    lines = []
    for ex in examples:
        verdict = "PASS (meets house style)" if ex["label"] else "FAIL (does not)"
        lines.append(f'Reply: "{ex["answer"]}"\nHuman verdict: {verdict}')
    return "\n\n".join(lines)


def judge_house_style(answer: str, few_shot_examples: list[dict] | None = None) -> dict:
    """Grade one reply against the house style.

    Pass `few_shot_examples` (list of {"answer", "label": bool}) to align the judge
    with human decisions. Omit them for the naive zero-shot judge. Returns the
    standard evaluator dict ``{"key", "score", "comment"}``.
    """
    judge = get_judge().with_structured_output(StyleGrade)

    prompt = RUBRIC
    if few_shot_examples:
        prompt += (
            "\n\nHere are examples of how our reviewers labeled past replies. "
            "Match their judgments:\n\n" + _format_examples(few_shot_examples)
        )
    prompt += f'\n\nNow grade this reply:\nReply: "{answer}"'

    grade = judge.invoke(prompt)
    return {
        "key": "meets_house_style",
        "score": 1 if grade.meets_house_style else 0,
        "comment": grade.reasoning,
    }


if __name__ == "__main__":
    # Tiny smoke test (needs a judge model key). For the real alignment numbers,
    # run judge_alignment.py.
    from config import require_langsmith

    require_langsmith()
    # Correct + warm, but no follow-up invitation -> our convention says FAIL.
    # Zero-shot (no examples) usually can't know that and calls it good.
    reply = ("Welcome aboard! New full-time employees get 15 vacation days a year, "
             "accruing at 1.25 days a month.")
    print("zero-shot:", judge_house_style(reply))
