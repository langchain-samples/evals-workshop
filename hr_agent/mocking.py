"""Tool-output mocking middleware — run the agent without running its tools.

Why mock tool outputs in an eval?

1. **Cost & speed** — real tools hit real APIs.
2. **Determinism** — a flaky downstream service becomes a flaky eval score.
3. **Isolation** — you're evaluating the *agent's* reasoning, not the tools.
4. **Coverage** — this is the big one. You can only test the states your real
   tools can actually produce. Our HR tools always succeed, so without mocks
   there is no way to ask "what does the agent do when provisioning FAILS?"
5. **Offline/CI** — CI often has no route to production systems.

Mechanically this is a LangChain ``AgentMiddleware`` that overrides
``wrap_tool_call``: it intercepts each tool call before execution and either
returns a canned ``ToolMessage`` or delegates to ``handler`` to run the real
tool.

Two flavors, matching the two ways teams actually do this:

- ``ToolMockMiddleware``          — one static mock table for the whole run.
- ``DatasetDrivenMockMiddleware`` — per-example mocks supplied by the dataset,
  so each test case can set up its own world state.

Security note: mock payloads arrive from a LangSmith dataset, i.e. remote data
that ends up inside an LLM prompt as a tool result. We treat it as untrusted:
tool names are checked against an allowlist of the agent's real tools, payloads
are serialized with ``json.dumps`` (never ``eval``/``pickle``/``langchain_core.load``),
and oversized payloads are rejected rather than forwarded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from hr_agent.tools import HR_TOOLS

# Allowlist: only tools the agent actually has can be mocked. A dataset that
# names anything else is a bug (or something worse) — fail loudly instead of
# injecting an arbitrary string into the model's context as a tool result.
KNOWN_TOOL_NAMES = frozenset(t.name for t in HR_TOOLS)

# Cap on a single serialized mock payload. Keeps a malformed or hostile dataset
# from blowing up the context window.
MAX_MOCK_CHARS = 8_000


def _serialize(payload: Any) -> str:
    """Render a mock payload as the string content of a ToolMessage.

    ``json.dumps`` for structured payloads (what the real tools return),
    ``str`` for plain scalars. No deserialization of arbitrary objects.
    """
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            text = str(payload)
    if len(text) > MAX_MOCK_CHARS:
        raise ValueError(
            f"Mock payload is {len(text)} chars, over the {MAX_MOCK_CHARS} limit."
        )
    return text


def _validate(mocks: dict[str, Any]) -> dict[str, Any]:
    """Reject mocks for tools this agent doesn't have."""
    unknown = sorted(set(mocks) - KNOWN_TOOL_NAMES)
    if unknown:
        raise ValueError(
            f"Mock defined for unknown tool(s): {unknown}. "
            f"Known tools: {sorted(KNOWN_TOOL_NAMES)}."
        )
    return dict(mocks)


class ToolMockMiddleware(AgentMiddleware):
    """Return canned outputs for named tools; run the rest for real.

    >>> agent = build_agent(middleware=[ToolMockMiddleware({
    ...     "lookup_employee": {"employee_id": "E1007", "full_name": "Jordan Lee"},
    ... })])

    Tools without a mock fall through to real execution, so you can mock only
    the slow/expensive/external ones and leave the cheap local ones alone.
    """

    def __init__(self, mock_responses: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.mock_responses = _validate(mock_responses or {})
        self._call_log: list[dict] = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        self._call_log.append({"name": name, "args": request.tool_call.get("args", {})})
        if name in self.mock_responses:
            return ToolMessage(
                content=_serialize(self.mock_responses[name]),
                tool_call_id=request.tool_call["id"],
                name=name,
            )
        return handler(request)  # no mock -> run the real tool

    @property
    def calls_made(self) -> list[dict]:
        """Every tool call this middleware saw, in order (name + args)."""
        return list(self._call_log)


class DatasetDrivenMockMiddleware(ToolMockMiddleware):
    """Per-example mocks, supplied by the dataset rather than hard-coded.

    This is the version that scales: the mock table comes from the example
    itself, so "Jordan Lee's provisioning fails" and "Jordan Lee's provisioning
    succeeds" are two rows in a dataset rather than two code paths.

    ``strict=True`` (the default) raises when the agent calls a tool the example
    didn't mock. That's usually what you want offline — an unmocked call means
    the example is incomplete, and you'd rather know than silently hit a real
    system. ``strict=False`` falls through to the real tool.
    """

    def __init__(self, tool_outputs: dict[str, Any] | None = None, *, strict: bool = True) -> None:
        super().__init__(tool_outputs)
        self.strict = strict

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        if self.strict and name not in self.mock_responses:
            self._call_log.append({"name": name, "args": request.tool_call.get("args", {})})
            raise ValueError(
                f"No mock defined for tool {name!r} (strict mode). "
                f"Mocked tools: {sorted(self.mock_responses)}."
            )
        return super().wrap_tool_call(request, handler)


if __name__ == "__main__":
    # Self-tests for the pure logic — no model or network needed.
    assert _serialize({"status": "ok"}) == '{"status": "ok"}'
    assert _serialize("plain") == "plain"
    try:
        _serialize("x" * (MAX_MOCK_CHARS + 1))
        raise AssertionError("oversized payload should have been rejected")
    except ValueError:
        pass

    assert _validate({"lookup_employee": {}}) == {"lookup_employee": {}}
    try:
        _validate({"drop_database": {}})
        raise AssertionError("unknown tool should have been rejected")
    except ValueError:
        pass

    mw = ToolMockMiddleware({"lookup_employee": {"employee_id": "E1007"}})
    assert mw.calls_made == []
    assert "lookup_employee" in mw.mock_responses
    print("All mocking self-tests passed.")
