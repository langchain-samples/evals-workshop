"""HR Onboarding agent — the system under test for the whole workshop.

Every module evaluates *this* agent, so you only learn one app and focus the
rest of your attention on evaluation technique.

`build_agent`/`run_agent` are imported lazily (via module __getattr__) so that
importing the pure-Python pieces — `hr_agent.knowledge`, `hr_agent.trajectory`
— does NOT require langchain to be installed. That's what lets the
deterministic evaluator self-tests run with zero dependencies.
"""

from hr_agent.trajectory import extract_tool_calls, extract_trajectory, final_response

__all__ = [
    "build_agent",
    "run_agent",
    "extract_trajectory",
    "extract_tool_calls",
    "final_response",
    "DatasetDrivenMockMiddleware",
    "ToolMockMiddleware",
]

_LAZY = {
    "build_agent": "hr_agent.agent",
    "run_agent": "hr_agent.agent",
    "ToolMockMiddleware": "hr_agent.mocking",
    "DatasetDrivenMockMiddleware": "hr_agent.mocking",
}


def __getattr__(name: str):
    # PEP 562 lazy import: only pulls in langchain on first use, so the pure
    # deterministic pieces stay importable with zero dependencies.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
