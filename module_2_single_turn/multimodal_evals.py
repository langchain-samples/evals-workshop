"""Module 2 — evaluators for the multimodal (receipt image) eval.

Same rule as text: deterministic first. A model reading an image still produces
text and numbers, and most of what can go wrong is checkable in code:

  - receipts_found        — one receipt per receipt attachment. Catches a dropped
                            file AND a receipt invented from a non-receipt image.
  - receipt_totals_match  — each receipt's total, matched by attachment name.
                            Fractional, so 1 of 2 right reads as 0.5.
  - missing_receipt_flagged — the cross-attachment check: did it spot the
                            statement charge with no receipt?
  - decision_matches      — approve / needs_receipt / reject.

Only one question needs a judge: is the written ``reason`` true to what the files
show? That judge gets the attachments too. An evaluator asks for them by adding
an ``attachments`` parameter, exactly like the target does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

TOTAL_TOLERANCE = 0.01  # dollars


def receipts_found(outputs: dict, reference_outputs: dict) -> dict:
    expected = len(reference_outputs.get("receipts", []))
    got = len(outputs.get("receipts") or [])
    return {
        "key": "receipts_found",
        "score": 1 if got == expected else 0,
        "comment": f"Extracted {got} receipt(s); expected {expected}.",
    }


def receipt_totals_match(outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of expected receipts whose total was read correctly.

    Matched on the attachment name (``source``), not on position or vendor: the
    point is that the number came from the right image.
    """
    expected = reference_outputs.get("receipts", [])
    if not expected:
        return {"key": "receipt_totals_match", "score": 1, "comment": "No receipts expected."}
    by_source = {r.get("source"): r for r in outputs.get("receipts") or []}
    misses = []
    for ref in expected:
        got = by_source.get(ref["source"])
        try:
            ok = got is not None and abs(float(got.get("total")) - ref["total"]) <= TOTAL_TOLERANCE
        except (TypeError, ValueError):
            ok = False
        if not ok:
            seen = "nothing" if got is None else got.get("total")
            misses.append(f"{ref['source']}: expected {ref['total']:.2f}, got {seen}")
    score = (len(expected) - len(misses)) / len(expected)
    return {
        "key": "receipt_totals_match",
        "score": score,
        "comment": "All totals correct." if not misses else "; ".join(misses),
    }


def missing_receipt_flagged(outputs: dict, reference_outputs: dict) -> dict:
    """Did it flag exactly the charges that lack a receipt?

    Keyword match, since the model may write "Gate 12 Parking" or "parking
    ($48.00)". An unexpected flag on a claim with nothing missing scores 0.
    """
    keywords = [k.lower() for k in reference_outputs.get("missing_receipt_keywords", [])]
    flagged = [str(m).lower() for m in outputs.get("missing_receipts") or []]
    if not keywords:
        ok = not flagged
        return {
            "key": "missing_receipt_flagged",
            "score": 1 if ok else 0,
            "comment": "Nothing flagged, correctly." if ok else f"Flagged but nothing missing: {flagged}",
        }
    hits = [k for k in keywords if any(k in f for f in flagged)]
    return {
        "key": "missing_receipt_flagged",
        "score": len(hits) / len(keywords),
        "comment": f"Expected {keywords}; flagged {flagged or 'nothing'}.",
    }


def decision_matches(outputs: dict, reference_outputs: dict) -> dict:
    got, want = outputs.get("decision"), reference_outputs.get("decision")
    return {
        "key": "decision_matches",
        "score": 1 if got == want else 0,
        "comment": f"Decision {got!r}; expected {want!r}.",
    }


DETERMINISTIC_EVALUATORS = [
    receipts_found,
    receipt_totals_match,
    missing_receipt_flagged,
    decision_matches,
]


# --- A judge that looks at the images ------------------------------------


class AttachmentGroundingGrade(BaseModel):
    reasoning: str = Field(description="Which claims you checked against which file.")
    is_grounded: bool = Field(
        description="True if every vendor, date, amount and claim in the review is visible in the files."
    )


def reason_grounded_in_attachments(inputs: dict, outputs: dict, attachments: dict) -> dict:
    """LLM judge: does the review describe what the files actually show?

    The judge is shown the same files as the reviewer. It does not see the
    answer key. That makes it reference-free, so the same evaluator works on
    production traces that have no ground truth.
    """
    from langchain_core.messages import HumanMessage

    from config import get_judge
    from hr_agent.receipts import file_blocks
    from module_2_single_turn.multimodal_datasets import files_from_attachments

    receipts = "\n".join(
        f"- {r.get('source')}: {r.get('vendor')}, {r.get('date')}, ${r.get('total')}"
        for r in outputs.get("receipts") or []
    ) or "(none)"
    # The claim's own fields go in too. Without them the judge marks a correct
    # "submitted on <date>" as unsupported, since no file shows that date.
    prompt = f"""You are checking an expense review against the files it was based on.

Facts from the claim form (true, but not in any file):
- Submitted on: {inputs.get('submitted_on', '')}
- Employee's note: {inputs.get('note', '')}

The review said:
Receipts extracted:
{receipts}
Charges flagged as missing a receipt: {outputs.get('missing_receipts') or 'none'}
Reason: {outputs.get('reason', '')}

Look at every attached file below. Is everything the review states (vendors,
dates, amounts, which file is which, which charges lack receipts) shown in the
files or in the claim-form facts above? Do not judge whether the decision follows the policy — only
whether the review is faithful to the files."""
    files = files_from_attachments(attachments)
    message = HumanMessage(content=[{"type": "text", "text": prompt}, *file_blocks(files)])
    grade = get_judge().with_structured_output(AttachmentGroundingGrade).invoke([message])
    return {
        "key": "reason_grounded_in_attachments",
        "score": 1 if grade.is_grounded else 0,
        "comment": grade.reasoning,
    }


LLM_JUDGE_EVALUATORS = [reason_grounded_in_attachments]


if __name__ == "__main__":
    # Self-tests on hand-built outputs — no API key, no images needed.
    ref = {
        "decision": "needs_receipt",
        "receipts": [
            {"source": "receipt_1", "vendor": "Noodle Bar", "date": "2026-09-22", "total": 22.80},
            {"source": "receipt_2", "vendor": "Paper Trail", "date": "2026-09-22", "total": 31.45},
        ],
        "missing_receipt_keywords": ["parking"],
    }
    good = {
        "decision": "needs_receipt",
        "receipts": [
            {"source": "receipt_1", "total": 22.8},
            {"source": "receipt_2", "total": 31.45},
        ],
        "missing_receipts": ["GATE 12 PARKING $48.00"],
    }
    for ev in DETERMINISTIC_EVALUATORS:
        assert ev(good, ref)["score"] == 1, ev.__name__

    # Totals swapped between files: right numbers, wrong images -> 0.
    swapped = {**good, "receipts": [
        {"source": "receipt_1", "total": 31.45},
        {"source": "receipt_2", "total": 22.80},
    ]}
    assert receipt_totals_match(swapped, ref)["score"] == 0

    # A receipt "read" off the welcome banner: count is wrong.
    invented = {**good, "receipts": [*good["receipts"], {"source": "photo", "total": 0}]}
    assert receipts_found(invented, ref)["score"] == 0

    # Missed the parking charge, and approved.
    missed = {**good, "missing_receipts": [], "decision": "approve"}
    assert missing_receipt_flagged(missed, ref)["score"] == 0
    assert decision_matches(missed, ref)["score"] == 0

    # Flagging something on a clean claim is wrong too.
    clean_ref = {"decision": "approve", "receipts": [], "missing_receipt_keywords": []}
    assert missing_receipt_flagged({"missing_receipts": ["coffee"]}, clean_ref)["score"] == 0

    # Garbage totals don't crash the evaluator.
    assert receipt_totals_match({"receipts": [{"source": "receipt_1", "total": "n/a"}]}, ref)["score"] == 0

    print("All multimodal evaluator self-tests passed.")
