"""Module 7 — measure (and improve) how well your judge agrees with humans.

A judge you don't trust is worse than no judge. So before you rely on an LLM judge
in production, **evaluate the evaluator**: score a set of outputs your humans have
already labeled, and measure how often the judge agrees with them. That agreement
rate is the judge's accuracy.

This script does exactly that, twice:
  1. zero-shot judge (rubric only)            -> agreement with humans
  2. few-shot judge (rubric + labeled examples) -> agreement with humans

The few-shot examples come from the SAME human labels (the kind you'd harvest from
the annotation queue in annotation_queue.py). We hold out a separate slice to score
on, so we're never grading the judge on the examples we taught it with. You should
see the few-shot judge agree with humans more often — that's alignment, no
fine-tuning required.

Run:  python module_7_improving_evals/judge_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import require_langsmith
from module_7_improving_evals.few_shot_judge import judge_house_style

# A human-labeled "golden" set for the HR house style. These are the labels real
# reviewers produced — the ground truth the judge must match. The convention they
# encode (NOT stated in the rubric): a reply must end with an explicit invitation
# to follow up ("let me know...", "happy to help", "feel free to reach out"). A
# reply can be warm, correct, and concise and still FAIL if it doesn't invite a
# follow-up. That's learnable from examples but not from the vague rubric — which
# is exactly what makes the few-shot lift visible.
LABELED = [
    # correct + ends with a follow-up invitation -> PASS
    {"answer": "You get 15 vacation days a year, accruing at 1.25 per month. Let me "
               "know if you'd like the full policy!", "label": True},
    {"answer": "Health coverage starts the first of the month after your start date. "
               "Happy to help if you have any questions!", "label": True},
    {"answer": "The 401(k) match is 4%, and you're eligible after 60 days. Feel free "
               "to reach out anytime!", "label": True},
    {"answer": "We're hybrid — on-site Tuesday through Thursday. Let me know if you "
               "want the details!", "label": True},
    {"answer": "You have 30 days to submit expenses, with receipts over $25. Reach "
               "out if anything's unclear!", "label": True},
    {"answer": "You get 10 paid sick days a year. Happy to walk through how to log "
               "them — just let me know!", "label": True},
    # correct (often warm/concise) but NO follow-up invitation -> FAIL
    {"answer": "Welcome aboard! You get 15 vacation days per year, accruing at 1.25 "
               "per month.", "label": False},
    {"answer": "Your health coverage begins the first of the month following your "
               "start date.", "label": False},
    {"answer": "The 401(k) match is 4%, and you're eligible after 60 days.", "label": False},
    {"answer": "We follow a hybrid schedule: on-site Tuesday through Thursday, "
               "remote Monday and Friday.", "label": False},
    {"answer": "Welcome! Expenses are due within 30 days, and receipts are required "
               "over $25.", "label": False},
    {"answer": "You receive 10 paid sick days per calendar year.", "label": False},
]

# Split: teach the few-shot judge on a couple labels of each verdict, score both
# judges on the rest. Never score on the examples we taught with.
FEW_SHOT = LABELED[0:2] + LABELED[6:8]  # 2 PASS (with invite) + 2 FAIL (no invite)
EVAL_SET = [ex for ex in LABELED if ex not in FEW_SHOT]


def agreement(use_few_shot: bool) -> tuple[float, list[str]]:
    """Run the judge over EVAL_SET; return (agreement_rate, mismatch_descriptions)."""
    examples = FEW_SHOT if use_few_shot else None
    agree = 0
    mismatches: list[str] = []
    for ex in EVAL_SET:
        result = judge_house_style(ex["answer"], few_shot_examples=examples)
        judge_label = bool(result["score"])
        if judge_label == ex["label"]:
            agree += 1
        else:
            mismatches.append(
                f'human={ex["label"]} judge={judge_label}  "{ex["answer"][:60]}..."'
            )
    return agree / len(EVAL_SET), mismatches


def main() -> None:
    require_langsmith()
    print(f"Evaluating the judge on {len(EVAL_SET)} human-labeled examples "
          f"({len(FEW_SHOT)} held out as few-shot).\n")

    zero_rate, zero_miss = agreement(use_few_shot=False)
    print(f"Zero-shot judge agreement with humans: {zero_rate:.0%}")
    for m in zero_miss:
        print(f"    mismatch: {m}")

    few_rate, few_miss = agreement(use_few_shot=True)
    print(f"\nFew-shot judge agreement with humans:  {few_rate:.0%}")
    for m in few_miss:
        print(f"    mismatch: {m}")

    delta = few_rate - zero_rate
    print(f"\nFew-shot alignment lift: {delta:+.0%}.")
    if delta > 0:
        print("Human-labeled examples pulled the judge toward your reviewers' decisions.")
    else:
        print("No lift this run — judges are noisy on small sets; try more examples "
              "or sharper labels. The method is the point: measure, then align.")


if __name__ == "__main__":
    main()
