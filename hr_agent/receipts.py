"""Expense-claim review: the HR app's multimodal step.

A new hire submits a claim: a short note plus files (receipt photos, sometimes a
card-statement CSV). This reads every file, extracts each receipt, and decides
whether the claim can be approved under the expenses policy.

It's one structured-output model call, not the tool-calling agent: there's
nothing to look up, and the decision has to come back as a typed object the
expense system can act on. Module 2's multimodal eval grades it.
"""

from __future__ import annotations

import base64
from datetime import date
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from config import AGENT_MODEL
from hr_agent.knowledge import HR_POLICIES

IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
TEXT_TYPES = {"text/csv", "text/plain"}

# (attachment name, mime type, raw bytes)
File = tuple[str, str, bytes]


class ExtractedReceipt(BaseModel):
    source: str = Field(description="Name of the attachment this receipt was read from.")
    vendor: str
    date: str = Field(description="Date printed on the receipt, YYYY-MM-DD.")
    total: float = Field(description="The receipt's final total, including tax and tip.")


class ExpenseReview(BaseModel):
    receipts: list[ExtractedReceipt] = Field(
        description="One entry per attachment that is a receipt. Skip anything that isn't."
    )
    missing_receipts: list[str] = Field(
        description="Charges over $25 on a statement that have no matching receipt."
    )
    reason: str = Field(description="One or two sentences citing what the files show.")
    decision: Literal["approve", "needs_receipt", "reject"]


INSTRUCTIONS = """You review expense claims for the HR team.

Expenses policy (the only rules that apply):
\"\"\"{policy}\"\"\"

The claim was submitted on {submitted_on}. The employee's note: {note}

The attached files follow, each labelled with its name. Extract every receipt.
Ignore attachments that are not receipts. If a card statement is attached, every
charge on it is being claimed. Then decide:
- reject: any receipt is dated more than 30 days before the submission date.
- needs_receipt: a claimed charge over $25 has no receipt.
- approve: otherwise."""


def file_blocks(files: Sequence[File]) -> list[dict]:
    """Turn files into LangChain content blocks, each preceded by its name.

    Images become image blocks; CSV/text is inlined as text. Anything else is
    refused rather than silently dropped, so a new file type fails loudly.
    """
    blocks: list[dict] = []
    for name, mime, data in files:
        blocks.append({"type": "text", "text": f"Attachment `{name}` ({mime}):"})
        if mime in IMAGE_TYPES:
            blocks.append({
                "type": "image",
                "base64": base64.b64encode(data).decode(),
                "mime_type": mime,
            })
        elif mime in TEXT_TYPES:
            blocks.append({"type": "text", "text": data.decode("utf-8", errors="replace")})
        else:
            raise ValueError(f"Unsupported attachment type for {name!r}: {mime}")
    return blocks


def review_expense_claim(
    note: str,
    submitted_on: str,
    files: Sequence[File],
    model: str = AGENT_MODEL,
) -> ExpenseReview:
    """Review one claim and return a typed decision."""
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage

    date.fromisoformat(submitted_on)  # reject a malformed date before spending a call
    prompt = INSTRUCTIONS.format(
        policy=HR_POLICIES["expenses"], submitted_on=submitted_on, note=note
    )
    message = HumanMessage(content=[{"type": "text", "text": prompt}, *file_blocks(files)])
    llm = init_chat_model(model, temperature=0).with_structured_output(ExpenseReview)
    return llm.invoke([message])
