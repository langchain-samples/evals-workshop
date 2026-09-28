"""Trace replay middleware — re-run the agent against a frozen copy of the world.

The problem this solves: **agent outcomes are only valid at a point in time.**

An agent investigates a record, recommends a fix, and the fix gets applied.
Re-run that same agent next week and the data is now correct — the agent finds
nothing wrong and produces a different (and equally right) outcome. A ground
truth written from the first run, "the agent should recommend fixing X", now
fails a perfectly healthy agent. Your regression suite decays as the world
changes, and you can't tell a regression from drift.

The fix is to freeze the world, not the answer. Record what the tools returned
during a real trace, then replay exactly those outputs while the *agent* — the
prompt, the model, the tool-selection logic — is re-run live. Now a score change
means the agent changed.

    production trace ──► recorded_calls_from_*() ──► dataset example
                                                          │
                     TraceReplayMiddleware(recorded) ◄────┘
                                  │
                     build_agent(middleware=[...])  ──► same agent, frozen world

How this differs from ``mocking.py``:

- ``mocking.py`` returns **one hand-written payload per tool name**. It builds
  worlds that never existed (outages, partial failures).
- ``replay.py`` returns **the recorded output for each specific call** — matched
  on tool name *and* arguments, so two ``lookup_employee`` calls for two
  different people each get the right record. It reproduces a world that *did*
  exist, as it was at the time.

And what it deliberately does not do: **it never guesses.** If the agent makes a
call the recording has no answer for — because your prompt change sent it down a
different path — there is no honest output to return. By default that call gets
an explicit error result and is logged as a *miss* (``on_miss="fail"``), and
you score ``replay_fidelity`` before trusting anything else. A miss is not a
bug in the replay; it's the agent behaving differently from the recording, which
is exactly what you want to know. ``on_miss="live"`` falls through to the real
tool instead — which mixes frozen and live data, so treat those runs as suspect.

Security note: recorded calls arrive from a LangSmith dataset or trace, i.e.
remote data that ends up inside an LLM prompt as a tool result. Same posture as
``mocking.py``: tool names are checked against an allowlist of the agent's real
tools, payloads are handled with ``json`` only (never ``eval``/``pickle``/
``langchain_core.load``), payloads and call counts are capped, and a replay never
executes anything. Traces from a real HR agent carry employee PII; nothing here
redacts it. See the note in ``module_6_improving_evals/curate_dataset.py``.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

# Make the repo root importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from hr_agent.mocking import KNOWN_TOOL_NAMES, _serialize

# A single agent turn rarely makes more than a handful of tool calls. A recording
# with hundreds is malformed (or hostile) — refuse it rather than load it.
MAX_RECORDED_CALLS = 50

MISS_MESSAGE = (
    "Replay miss: the recorded trace has no output for this exact call, so none "
    "can be returned."
)


# --- Recording: turn a finished run into replayable calls ----------------------

def _kind(msg: Any) -> str:
    """'ai' | 'tool' | 'human' | ... for a message object or a serialized dict."""
    if not isinstance(msg, dict):
        return str(getattr(msg, "type", "") or "")
    inner = msg.get("kwargs") if isinstance(msg.get("kwargs"), dict) else (
        msg.get("data") if isinstance(msg.get("data"), dict) else {}
    )
    kind = inner.get("type") or msg.get("type") or ""
    if kind in ("", "constructor") and isinstance(msg.get("id"), list) and msg["id"]:
        # LangChain's constructor serialization: id == [..., "AIMessage"].
        kind = str(msg["id"][-1]).lower().removesuffix("chunk").removesuffix("message")
    return str(kind).lower()


def _field(msg: Any, name: str, default: Any = None) -> Any:
    if not isinstance(msg, dict):
        return getattr(msg, name, default)
    if msg.get(name) is not None:
        return msg[name]
    for wrapper in ("kwargs", "data"):
        inner = msg.get(wrapper)
        if isinstance(inner, dict) and inner.get(name) is not None:
            return inner[name]
    return default


def recorded_calls_from_messages(
    messages: Iterable[Any] | dict,
    *,
    last_turn_only: bool = False,
) -> list[dict]:
    """Pair each tool call with the output it got. Returns replayable calls.

    Accepts an ``agent.invoke`` result (``{"messages": [...]}``), a list of
    message objects, or the *serialized* messages LangSmith stores on a run —
    plain dicts, ``{"type": "ai", "data": {...}}``, or LangChain's constructor
    format. Calls that never got an output (an interrupted run) are dropped:
    there is nothing to replay.

    ``last_turn_only=True`` keeps only calls made after the last human message.
    In a multi-turn thread each turn's output holds the *whole* history so far;
    without this a snapshot of turn 3 silently includes turns 1 and 2.
    """
    if isinstance(messages, dict):
        messages = messages.get("messages", [])
    messages = list(messages)

    if last_turn_only:
        last_human = max(
            (i for i, m in enumerate(messages) if _kind(m) in ("human", "user")),
            default=-1,
        )
        messages = messages[last_human + 1:]

    order: list[str] = []
    calls: dict[str, dict] = {}
    for msg in messages:
        kind = _kind(msg)
        if kind == "ai":
            for tc in _field(msg, "tool_calls", None) or []:
                if not isinstance(tc, dict) or not tc.get("id"):
                    continue
                order.append(tc["id"])
                calls[tc["id"]] = {"name": tc.get("name"), "args": tc.get("args") or {}}
        elif kind == "tool":
            tool_call_id = _field(msg, "tool_call_id")
            if tool_call_id in calls:
                calls[tool_call_id]["output"] = _field(msg, "content", "")

    return [calls[i] for i in order if "output" in calls[i]]


def recorded_calls_from_run(run: Any, *, last_turn_only: bool = False) -> list[dict]:
    """Recorded calls from a LangSmith run (a ``client.read_run`` / ``list_runs`` row).

    Reads the agent root run's ``outputs["messages"]`` — the final message state,
    which carries every tool call and its result. Serialized message shapes vary
    by SDK version, so this is lenient; check it with
    ``module_3_agent_evals/snapshot_trace.py --dry-run`` on one of *your* traces
    before you build a dataset from it.
    """
    outputs = getattr(run, "outputs", None) or {}
    messages = outputs.get("messages")
    if not isinstance(messages, list):
        raise ValueError(
            "Run has no outputs['messages'] list — is this the agent's root run?"
        )
    return recorded_calls_from_messages(messages, last_turn_only=last_turn_only)


# --- Validation ----------------------------------------------------------------

def validate_recorded(calls: Iterable[dict]) -> list[dict]:
    """Check a recording is safe to replay; return normalized copies.

    Rejects: unknown tool names (allowlist), non-dict args, oversized payloads,
    and recordings with too many calls. Raises ``ValueError`` — fail loudly rather
    than put an arbitrary string into the model's context as a tool result.
    """
    calls = list(calls or [])
    if len(calls) > MAX_RECORDED_CALLS:
        raise ValueError(
            f"Recording has {len(calls)} calls, over the {MAX_RECORDED_CALLS} limit."
        )
    clean: list[dict] = []
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"Recorded call #{i} is not an object.")
        name = call.get("name")
        if name not in KNOWN_TOOL_NAMES:
            raise ValueError(
                f"Recorded call #{i} names unknown tool {name!r}. "
                f"Known tools: {sorted(KNOWN_TOOL_NAMES)}."
            )
        args = call.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"Recorded call #{i} ({name}) has non-object args.")
        if "output" not in call:
            raise ValueError(f"Recorded call #{i} ({name}) has no 'output'.")
        _serialize(call["output"])  # raises if over the size cap
        clean.append({"name": name, "args": args, "output": call["output"]})
    return clean


# --- Matching ------------------------------------------------------------------

def _norm(value: Any) -> Any:
    """Normalize for matching: case/whitespace-insensitive strings, sorted keys.

    The model may call ``lookup_employee("jordan lee")`` where the recording has
    ``"Jordan Lee"``; the real tool normalizes both, so the replay should too.
    Deliberately *not* fuzzier than that — a genuinely different argument (another
    employee, another date) must miss, not silently get the wrong record.
    """
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def _key(name: str, args: dict) -> tuple[str, str]:
    return name, json.dumps(_norm(args), sort_keys=True, default=str)


# --- The middleware ------------------------------------------------------------

class TraceReplayMiddleware(AgentMiddleware):
    """Serve tool results from a recorded trace instead of running tools.

    >>> calls = recorded_calls_from_messages(production_result)
    >>> replay = TraceReplayMiddleware(calls)
    >>> result = run_agent(question, middleware=[replay])
    >>> replay.report()["misses"]     # [] means the agent stayed on the recording

    Repeated calls: if the recording holds the same call twice (say, two reads
    that returned different values as the world changed mid-trace) they are served
    in order. Once only one is left it keeps being served — idempotent reads
    shouldn't fail because the agent asked a third time.
    """

    def __init__(self, recorded_calls: Iterable[dict], *, on_miss: str = "fail") -> None:
        super().__init__()
        if on_miss not in ("fail", "live"):
            raise ValueError("on_miss must be 'fail' or 'live'.")
        self.on_miss = on_miss
        self._recorded = validate_recorded(recorded_calls)
        self._queues: dict[tuple[str, str], deque] = {}
        for call in self._recorded:
            self._queues.setdefault(_key(call["name"], call["args"]), deque()).append(call["output"])
        self._served: set[tuple[str, str]] = set()
        self._call_log: list[dict] = []
        self._misses: list[dict] = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        args = request.tool_call.get("args") or {}
        self._call_log.append({"name": name, "args": args})

        key = _key(name, args)
        queue = self._queues.get(key)
        if queue:
            output = queue[0] if len(queue) == 1 else queue.popleft()
            self._served.add(key)
            return ToolMessage(
                content=_serialize(output),
                tool_call_id=request.tool_call["id"],
                name=name,
            )

        self._misses.append({"name": name, "args": args})
        if self.on_miss == "live":
            return handler(request)  # frozen + live data mixed: the caller opted in
        return ToolMessage(
            content=json.dumps({"error": MISS_MESSAGE}),
            tool_call_id=request.tool_call["id"],
            name=name,
            status="error",
        )

    @property
    def calls_made(self) -> list[dict]:
        """Every tool call the agent made, in order (name + args)."""
        return list(self._call_log)

    def report(self) -> dict:
        """Replay accounting: what was recorded, what the agent asked for, what missed.

        ``unused`` counts recorded calls the agent never made — informational
        (the agent may have legitimately skipped a step), unlike ``misses``.
        """
        unused = sum(
            1 for c in self._recorded if _key(c["name"], c["args"]) not in self._served
        )
        return {
            "recorded": len(self._recorded),
            "calls": len(self._call_log),
            "misses": list(self._misses),
            "unused": unused,
        }


if __name__ == "__main__":
    # Self-tests for the pure logic — no model or network needed.
    from langchain_core.messages import AIMessage, HumanMessage

    def _tc(id_, tool, /, **args):
        return {"name": tool, "args": args, "id": id_, "type": "tool_call"}

    history = [
        HumanMessage("Schedule Jordan Lee's orientation."),
        AIMessage("", tool_calls=[_tc("c1", "lookup_employee", name="Jordan Lee")]),
        ToolMessage('{"employee_id": "E1007", "start_date": "2026-06-15"}',
                    tool_call_id="c1", name="lookup_employee"),
        AIMessage("", tool_calls=[_tc("c2", "schedule_orientation",
                                      employee_id="E1007", date="2026-06-15")]),
        ToolMessage('{"status": "scheduled"}', tool_call_id="c2", name="schedule_orientation"),
        AIMessage("Done."),
    ]

    # 1. Recording from real message objects, from a result dict, and per-turn.
    recorded = recorded_calls_from_messages({"messages": history})
    assert [c["name"] for c in recorded] == ["lookup_employee", "schedule_orientation"]
    assert recorded[0]["args"] == {"name": "Jordan Lee"}
    assert recorded[0]["output"].startswith('{"employee_id"')

    two_turns = history + [
        HumanMessage("Now order them a laptop."),
        AIMessage("", tool_calls=[_tc("c3", "provision_equipment",
                                      employee_id="E1007", equipment_type="laptop")]),
        ToolMessage('{"status": "ordered"}', tool_call_id="c3", name="provision_equipment"),
    ]
    assert len(recorded_calls_from_messages(two_turns)) == 3
    assert [c["name"] for c in recorded_calls_from_messages(two_turns, last_turn_only=True)] \
        == ["provision_equipment"]

    # 2. Recording from serialized shapes LangSmith may hand back.
    plain = [
        {"type": "human", "content": "hi"},
        {"type": "ai", "content": "", "tool_calls": [_tc("x1", "lookup_hr_policy", topic="vacation")]},
        {"type": "tool", "tool_call_id": "x1", "content": "15 days"},
    ]
    wrapped = [
        {"type": "ai", "data": {"content": "", "tool_calls": [_tc("x1", "lookup_hr_policy", topic="vacation")]}},
        {"type": "tool", "data": {"tool_call_id": "x1", "content": "15 days"}},
    ]
    constructor = [
        {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "AIMessage"],
         "kwargs": {"content": "", "type": "ai",
                    "tool_calls": [_tc("x1", "lookup_hr_policy", topic="vacation")]}},
        {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "ToolMessage"],
         "kwargs": {"tool_call_id": "x1", "content": "15 days", "type": "tool"}},
    ]
    for shape in (plain, wrapped, constructor):
        got = recorded_calls_from_messages(shape)
        assert got == [{"name": "lookup_hr_policy", "args": {"topic": "vacation"}, "output": "15 days"}], got

    # A call that never got a result (interrupted run) can't be replayed.
    assert recorded_calls_from_messages(history[:2]) == []

    # 3. Validation: allowlist, args shape, missing output, size, count.
    def _rejects(calls):
        try:
            validate_recorded(calls)
        except ValueError:
            return True
        return False

    assert _rejects([{"name": "drop_database", "args": {}, "output": "x"}])
    assert _rejects([{"name": "lookup_employee", "args": "nope", "output": "x"}])
    assert _rejects([{"name": "lookup_employee", "args": {}}])
    assert _rejects([{"name": "lookup_employee", "args": {}, "output": "x" * 9_000}])
    assert _rejects([{"name": "lookup_employee", "args": {}, "output": "x"}] * (MAX_RECORDED_CALLS + 1))
    assert not _rejects(recorded)

    # 4. Matching: case/space-insensitive strings, but different args must miss.
    assert _key("lookup_employee", {"name": " jordan LEE "}) == _key("lookup_employee", {"name": "Jordan Lee"})
    assert _key("lookup_employee", {"name": "Sam Rivera"}) != _key("lookup_employee", {"name": "Jordan Lee"})
    assert _key("schedule_orientation", {"date": "d", "employee_id": "e"}) \
        == _key("schedule_orientation", {"employee_id": "e", "date": "d"})

    # 5. The middleware, driven directly (no agent, no model).
    def _request(tool, call_id, /, **args):
        return ToolCallRequest(tool_call=_tc(call_id, tool, **args), tool=None, state={}, runtime=None)

    def _no_live(_request):
        raise AssertionError("replay must not reach a real tool")

    mw = TraceReplayMiddleware(recorded)
    hit = mw.wrap_tool_call(_request("lookup_employee", "a", name="jordan lee"), _no_live)
    assert hit.content == recorded[0]["output"] and hit.tool_call_id == "a"
    miss = mw.wrap_tool_call(_request("lookup_employee", "b", name="Sam Rivera"), _no_live)
    assert miss.status == "error" and "Replay miss" in miss.content
    assert mw.report()["misses"] == [{"name": "lookup_employee", "args": {"name": "Sam Rivera"}}]
    assert mw.report()["unused"] == 1  # schedule_orientation was never asked for
    assert [c["name"] for c in mw.calls_made] == ["lookup_employee", "lookup_employee"]

    # 6. Repeats are served in order, then the last one sticks.
    drift = TraceReplayMiddleware([
        {"name": "lookup_employee", "args": {"name": "Jordan Lee"}, "output": "before"},
        {"name": "lookup_employee", "args": {"name": "Jordan Lee"}, "output": "after"},
    ])
    served = [drift.wrap_tool_call(_request("lookup_employee", str(i), name="Jordan Lee"), _no_live).content
              for i in range(3)]
    assert served == ["before", "after", "after"], served

    # 7. on_miss="live" hands off to the real handler; bad on_miss is rejected.
    sentinel = ToolMessage("live", tool_call_id="z", name="lookup_employee")
    live = TraceReplayMiddleware(recorded, on_miss="live")
    assert live.wrap_tool_call(_request("lookup_employee", "z", name="Nobody"), lambda r: sentinel) is sentinel
    try:
        TraceReplayMiddleware(recorded, on_miss="guess")
        raise AssertionError("bad on_miss should have been rejected")
    except ValueError:
        pass

    print("All replay self-tests passed.")
