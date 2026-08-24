"""Helpers to extract a tool-call trajectory from an agent result.

create_agent returns ``{"messages": [...]}``. The "trajectory" is the ordered
sequence of tool calls the model made along the way. We read it from the
AIMessages' ``tool_calls`` field. Module 3 evaluators compare this against an
expected trajectory.

**Scope: one agent run, not a conversation.** ``run_agent`` invokes the agent
with a single user message, so a trajectory here spans the internal steps of
*one turn* — multi-step, single-turn. It is not a multi-turn thread. If you
extend this to real conversations, decide deliberately whether each evaluator
should score a turn or the whole thread; see the note in
``module_3_agent_evals/trajectory_evals.py``.
"""

from __future__ import annotations


def extract_tool_calls(result: dict) -> list[dict]:
    """Return the ordered tool calls as dicts: {"name": str, "args": dict}.

    Walks every message and collects each tool call in the order it was made.
    """
    calls: list[dict] = []
    for message in result.get("messages", []):
        # AIMessages carry .tool_calls; other message types won't have any.
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            calls.append({"name": tc["name"], "args": tc.get("args", {})})
    return calls


def extract_trajectory(result: dict) -> list[str]:
    """Return just the ordered tool *names* — the trajectory.

    e.g. ["lookup_employee", "create_it_account", "provision_equipment"].
    """
    return [call["name"] for call in extract_tool_calls(result)]


def final_response(result: dict) -> str:
    """Return the text content of the agent's final message."""
    messages = result.get("messages", [])
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    # Some providers return content as a list of blocks; normalize to text.
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content or ""
