"""Evaluate the evaluator — for a System 1 model this time.

`judge_alignment.py` asks how well an LLM judge agrees with human labels. This asks
the same of two cheaper ways to answer one question — "did the reply admit the
tool failed?":

  phrase   the Module 3 check (`tool_evals.reports_tool_failure`): a phrase list.
           Free, deterministic, and runs with no keys.
  system1  a System 1 model (Jev) asked one Noul question. Needs a key, and
           **sends text to a third party** — see below.

Both are scored against `system1_labels.py` on the **gate** split. The **scratch**
split exists so you can tune the review band without contaminating the number you
report; tune on gate and it stops measuring anything.

    python module_6_improving_evals/system1_alignment.py                     # phrase only, offline
    python module_6_improving_evals/system1_alignment.py --classifier both   # + System 1 (needs a key)
    python module_6_improving_evals/system1_alignment.py --split scratch     # where you may tune
    python module_6_improving_evals/system1_alignment.py --self-test         # no network

What to look at:

- **Accuracy is the wrong headline.** The set is balanced, so accuracy is
  interpretable here, but on real traffic failures are rare and a classifier that
  always says "no failure" scores 95%. Read precision and recall for the *True*
  class, and the per-category error table — it says *where* each one fails.
- **`review`** counts answers inside the confidence band. Those aren't errors;
  they're the model saying "ask a human". Accuracy is computed on the decided
  answers only, next to how many were left undecided — a classifier can look
  accurate by declining to answer, so always read the two together.
- **The set is author-written and adversarial by design** (see `system1_labels.py`).
  The phrase baseline's score here shows that phrase lists have failure modes. It
  is not how a phrase list does on your traffic, and nothing here is evidence for
  how System 1 does on yours. Bring 50 of your own labeled replies.

Data note: `--classifier system1|both` sends the (synthetic) labeled text to the
provider you choose, and prints the host first. Nothing is sent unless you pass it
explicitly. TypeSafe documents zero data retention as an enterprise option only.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module_3_agent_evals.tool_evals import reports_tool_failure
from module_6_improving_evals.system1_judge import (
    REVIEW_BAND,
    ask_acknowledges,
    default_client,
    verdict,
)
from module_6_improving_evals.system1_labels import EXAMPLES

Classifier = Callable[[dict], "bool | None"]  # example -> True / False / None (= needs review)


# --- Metrics (pure) -------------------------------------------------------------

def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def score_predictions(examples: list[dict], preds: list["bool | None"]) -> dict:
    """Confusion counts and rates for the True class, over *decided* answers only.

    ``None`` predictions (inside the review band) are counted separately and left
    out of accuracy/precision/recall — but ``coverage`` reports how many were
    decided, because a classifier can look accurate by abstaining.
    """
    tp = fp = fn = tn = review = 0
    errors: list[dict] = []
    for ex, pred in zip(examples, preds):
        if pred is None:
            review += 1
            continue
        if pred and ex["label"]:
            tp += 1
        elif pred and not ex["label"]:
            fp += 1
        elif not pred and ex["label"]:
            fn += 1
        else:
            tn += 1
        if pred != ex["label"]:
            errors.append({"example": ex, "pred": pred})
    decided = tp + fp + fn + tn
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision and recall else (
        0.0 if precision is not None and recall is not None else None)
    return {
        "n": len(examples), "decided": decided, "review": review,
        "coverage": _ratio(decided, len(examples)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _ratio(tp + tn, decided),
        "precision": precision, "recall": recall, "f1": f1,
        "errors": errors,
    }


def _fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


# --- Classifiers ----------------------------------------------------------------

def phrase_classifier(example: dict) -> bool:
    """The Module 3 phrase-list check, called exactly as the experiment calls it.

    No ``forbidden_phrases`` are passed, so this is the "did it use any failure
    word" half of `reports_tool_failure` — the half that matters for the
    comparison. (The other half is a per-example list that has to be authored.)
    """
    result = reports_tool_failure(
        {"answer": example["reply"]}, {"should_report_failure": True}
    )
    return result["score"] == 1


def make_system1_classifier(client, band: tuple[float, float]) -> Classifier:
    def classify(example: dict):
        p = ask_acknowledges(client, example["tool_results"], example["reply"])
        return verdict(p, band)

    return classify


# --- Reporting ------------------------------------------------------------------

def report(name: str, examples: list[dict], classify: Classifier) -> dict:
    preds = [classify(ex) for ex in examples]
    m = score_predictions(examples, preds)
    print(f"\n{name}  ({m['n']} examples: {m['decided']} decided, {m['review']} in review band)")
    print(f"  accuracy {_fmt(m['accuracy'])}   precision {_fmt(m['precision'])}   "
          f"recall {_fmt(m['recall'])}   f1 {_fmt(m['f1'])}   coverage {_fmt(m['coverage'])}")
    print(f"  confusion  TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
    if m["errors"]:
        by_cat: dict[str, Counter] = defaultdict(Counter)
        for e in m["errors"]:
            by_cat[e["example"]["category"]]["fp" if e["pred"] else "fn"] += 1
        print("  errors by category: " + ", ".join(
            f"{cat}: {sum(c.values())} ({'FP' if c['fp'] else 'FN'})" for cat, c in sorted(by_cat.items())))
        for e in m["errors"][:6]:
            ex = e["example"]
            kind = "false positive" if e["pred"] else "false negative"
            print(f"    [{ex['id']}] {kind}: {ex['reply'][:78]!r}")
        if len(m["errors"]) > 6:
            print(f"    … and {len(m['errors']) - 6} more")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Score failure-acknowledgement classifiers against labels.")
    parser.add_argument("--classifier", choices=("phrase", "system1", "both"), default="phrase")
    parser.add_argument("--split", choices=("gate", "scratch", "all"), default="gate",
                        help="Which labels to score. Report on gate; tune the band on scratch.")
    parser.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=REVIEW_BAND,
                        help="Probabilities inside [LO, HI] go to human review.")
    parser.add_argument("--self-test", action="store_true", help="Run offline self-tests and exit.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    band = tuple(args.band)
    if not 0 <= band[0] <= band[1] <= 1:
        raise SystemExit("--band must satisfy 0 <= LO <= HI <= 1.")
    examples = EXAMPLES if args.split == "all" else [e for e in EXAMPLES if e["split"] == args.split]
    print(f"Scoring on the '{args.split}' split: {len(examples)} labeled replies "
          f"({sum(e['label'] for e in examples)} True).")
    if args.split == "scratch":
        print("(scratch is for tuning — don't quote these numbers as your result.)")

    if args.classifier in ("phrase", "both"):
        report("phrase  (tool_evals.reports_tool_failure signal words)", examples, phrase_classifier)

    if args.classifier in ("system1", "both"):
        try:
            client = default_client()
        except Exception as e:  # missing key etc. — say so plainly, don't traceback
            raise SystemExit(f"Can't run the System 1 classifier: {e}")
        print(f"\nSending {len(examples)} synthetic examples to {client.host} "
              f"(model {client.model}). This leaves your machine.")
        report(f"system1 ({client.model}, review band {band})", examples,
               make_system1_classifier(client, band))

    print("\nNext: label ~50 of YOUR replies, re-run on those, and only then decide. "
          "Send the in-band ones to an annotation queue.")


# --- Self-test ------------------------------------------------------------------

def _self_test() -> None:
    ex = lambda label: {"label": label, "category": "X", "id": "x", "reply": "r"}  # noqa: E731

    # Metrics on a hand-checkable case: TP=2 FP=1 FN=1 TN=3, one abstention.
    examples = [ex(True), ex(True), ex(False), ex(True), ex(False), ex(False), ex(False), ex(True)]
    preds = [True, True, True, False, False, False, False, None]
    m = score_predictions(examples, preds)
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["review"], m["decided"]) == (2, 1, 1, 3, 1, 7)
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["accuracy"] - 5 / 7) < 1e-9 and abs(m["coverage"] - 7 / 8) < 1e-9
    assert len(m["errors"]) == 2

    # Abstaining is visible: a classifier that declines everything has no accuracy,
    # zero coverage — not a perfect score.
    m = score_predictions(examples, [None] * 8)
    assert m["accuracy"] is None and m["coverage"] == 0 and m["review"] == 8

    # Degenerate cases don't divide by zero.
    m = score_predictions([ex(False), ex(False)], [False, False])
    assert m["precision"] is None and m["recall"] is None and m["accuracy"] == 1.0

    # The phrase baseline behaves the way system1_labels.py says it was built to:
    # it gets category A right, misses paraphrases (B), and is fooled by claims
    # of success that contain a trigger word (E). If these fail, either the phrase
    # list changed or a label was written wrongly — go and look.
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for e in EXAMPLES:
        by_cat[e["category"]].append(phrase_classifier(e) == e["label"])
    assert all(by_cat["A"]), "phrase list should catch explicit acknowledgements"
    assert not any(by_cat["B"]), "no B example should contain a phrase-list trigger word"
    assert not any(by_cat["E"]), "every E example should contain a phrase-list trigger word"
    assert all(by_cat["C"]) and all(by_cat["D"]), "C/D are silent on failure: nothing to trigger on"

    # The pipeline end to end with a fake System 1 client: a perfectly calibrated
    # oracle scores perfectly, an always-unsure one abstains on everything.
    class Oracle:
        host, model = "fake.test", "oracle"

        def __init__(self, fixed=None):
            self.fixed = fixed
            self.truth = {(e["reply"]): e["label"] for e in EXAMPLES}

        def ask(self, state, questions):
            p = self.fixed if self.fixed is not None else (0.97 if self.truth[state["assistant_reply"]] else 0.03)
            return {"acknowledges_failure": {"type": "noul", "noul": p}}

    gate = [e for e in EXAMPLES if e["split"] == "gate"]
    m = score_predictions(gate, [make_system1_classifier(Oracle(), REVIEW_BAND)(e) for e in gate])
    assert m["accuracy"] == 1.0 and m["review"] == 0
    m = score_predictions(gate, [make_system1_classifier(Oracle(0.5), REVIEW_BAND)(e) for e in gate])
    assert m["review"] == len(gate) and m["accuracy"] is None

    print("All System 1 alignment self-tests passed.")


if __name__ == "__main__":
    main()
