"""Expense-claim review: the HR app's multimodal step.

A new hire submits a claim: a short note plus files (receipt photos, sometimes a
card-statement CSV). This reads every file, extracts each receipt, and decides
whether the claim can be approved under the expenses policy.

It's one structured-output model call, not the tool-calling agent: there's
nothing to look up, and the decision has to come back as a typed object the
expense system can act on. Module 2's multimodal eval grades
``review_expense_claim`` (one message); Module 4 grades ``review_expense_thread``,
where files and corrections arrive over several turns.
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


DECISION_RULES = """- reject: any receipt is dated more than 30 days before the submission date.
- needs_receipt: a claimed charge over $25 has no receipt.
- approve: otherwise."""

INSTRUCTIONS = """You review expense claims for the HR team.

Expenses policy (the only rules that apply):
\"\"\"{policy}\"\"\"

The claim was submitted on {submitted_on}. The employee's note: {note}

The attached files follow, each labelled with its name. Extract every receipt.
Ignore attachments that are not receipts. If a card statement is attached, every
charge on it is being claimed. Then decide:
{rules}"""

THREAD_INSTRUCTIONS = """You review expense claims for the HR team. The claim is
the conversation below: the employee may send files in any message, and you may
already have replied.

Expenses policy (the only rules that apply):
\"\"\"{policy}\"\"\"

The claim was submitted on {submitted_on}.

Review the claim as it stands after the employee's LAST message. Later messages
override earlier ones: if a receipt is replaced, use only the replacement; if a
charge is withdrawn, it is no longer claimed. Each file is labelled with its name.
Ignore attachments that are not receipts. If a card statement is attached, every
charge on it is claimed unless the employee withdrew it. Then decide:
{rules}"""

# One turn of a conversation: {"role": "user" | "assistant", "text": str, "files": [File, ...]}
Turn = dict


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
        policy=HR_POLICIES["expenses"], submitted_on=submitted_on, note=note,
        rules=DECISION_RULES,
    )
    message = HumanMessage(content=[{"type": "text", "text": prompt}, *file_blocks(files)])
    llm = init_chat_model(model, temperature=0).with_structured_output(ExpenseReview)
    return llm.invoke([message])


def review_expense_thread(
    turns: Sequence[Turn],
    submitted_on: str,
    model: str = AGENT_MODEL,
) -> ExpenseReview:
    """Review a claim made over several turns; return the decision after the last one.

    Each turn becomes its own message with its own role, and its files go inside
    that message. The conversation is passed as messages, never pasted into one
    prompt, so earlier turns stay the employee's words rather than instructions.
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    date.fromisoformat(submitted_on)
    if not turns or turns[-1].get("role") != "user":
        raise ValueError("A thread must end with the employee's message.")
    messages = [SystemMessage(THREAD_INSTRUCTIONS.format(
        policy=HR_POLICIES["expenses"], submitted_on=submitted_on, rules=DECISION_RULES,
    ))]
    for turn in turns:
        if turn["role"] == "assistant":
            messages.append(AIMessage(turn["text"]))
        elif turn["role"] == "user":
            content = [{"type": "text", "text": turn["text"]}, *file_blocks(turn.get("files", ()))]
            messages.append(HumanMessage(content=content))
        else:
            raise ValueError(f"Unknown role: {turn['role']!r}")
    llm = init_chat_model(model, temperature=0).with_structured_output(ExpenseReview)
    return llm.invoke(messages)
