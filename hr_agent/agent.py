"""The HR Onboarding agent, built with langchain's create_agent.

A single-purpose tool-calling agent: it answers new-hire policy questions and
performs onboarding actions (provisioning accounts/equipment, scheduling
orientation). create_agent runs the standard ReAct loop (model -> tools ->
model -> ...), which is exactly what we want to evaluate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from langchain.agents import create_agent

from config import AGENT_MODEL
from hr_agent.tools import HR_TOOLS

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware

SYSTEM_PROMPT = """You are the HR Onboarding Assistant for a mid-size company.

You help new hires and the people team with two kinds of requests:
1. Answering HR policy and benefits questions.
2. Performing onboarding actions: provisioning IT accounts, ordering
   equipment, and scheduling orientation.

Guidelines:
- For policy questions, ALWAYS call lookup_hr_policy and ground your answer in
  the returned text. Never invent policy numbers or dates.
- For onboarding actions, you usually need an employee_id. Call
  lookup_employee first to get it, then perform the requested actions.
- Be concise, warm, and professional. New hires are your audience.
- If a request is ambiguous or an employee can't be found, say so rather than
  guessing.
"""


def build_agent(
    model: str = AGENT_MODEL,
    middleware: Sequence["AgentMiddleware"] = (),
    checkpointer=None,
):
    """Construct the HR onboarding agent. Returns a compiled LangGraph app.

    ``middleware`` is how evals swap in mocked or replayed tool outputs without
    touching the agent's own code — the system under test stays identical, only
    the world around it changes. See hr_agent/mocking.py and hr_agent/replay.py.

    ``checkpointer`` is what makes a conversation a *thread*: with one, the agent
    remembers earlier turns that share a ``thread_id``. Without it (the default,
    and what every eval in Modules 1-4 uses) each invoke is independent.
    """
    return create_agent(
        model=model,
        tools=HR_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer,
    )


def run_agent(
    question: str,
    model: str = AGENT_MODEL,
    middleware: Sequence["AgentMiddleware"] = (),
) -> dict:
    """Run the agent on a single user message and return the raw result.

    The result is the standard create_agent output: ``{"messages": [...]}``.
    Downstream evaluators read both the final message and the tool-call
    trajectory from this. See hr_agent/trajectory.py for the extractors.
    """
    agent = build_agent(model, middleware=middleware)
    return agent.invoke({"messages": [("user", question)]})


def run_conversation(
    turns: Sequence[str],
    thread_id: str | None = None,
    model: str = AGENT_MODEL,
    middleware: Sequence["AgentMiddleware"] = (),
) -> list[dict]:
    """Run several user messages as ONE thread; return one result per turn.

    Each turn is a separate ``invoke`` (so a separate root run in LangSmith), all
    sharing a ``thread_id`` — which is precisely what LangSmith's **Trajectory
    view** groups on. LangGraph copies ``thread_id`` from the run config into the
    trace metadata and marks each turn's top-level run as ``ls_agent_type: root``
    itself, so there is nothing to set by hand for this agent.

    Each result's ``messages`` holds the *whole* history so far, not just that
    turn — see ``recorded_calls_from_messages(..., last_turn_only=True)`` when you
    need one turn's tool calls.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    thread_id = thread_id or new_thread_id()
    agent = build_agent(model, middleware=middleware, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    return [
        agent.invoke({"messages": [("user", turn)]}, config=config) for turn in turns
    ]


def new_thread_id() -> str:
    """A fresh thread id. UUIDv7 sorts by creation time, which LangSmith recommends."""
    from langsmith import uuid7

    return str(uuid7())


if __name__ == "__main__":
    # Quick manual smoke test: `python -m hr_agent.agent`
    result = run_agent("How many vacation days do new employees get?")
    print(result["messages"][-1].content)
