# Module 3 — Agent Evaluations (Trajectory & Tools)

> Goal: stop judging only the final answer and start judging *how* the agent
> got there — the sequence of tool calls and the arguments it passed.

A correct-sounding final message can hide a broken process: the agent called a
destructive tool it shouldn't have, looked up the wrong employee, or took ten
steps to do a two-step job. Agent evals catch these. They're what separates
"the demo worked" from "this is safe to ship."

## Scope: a trajectory here is one agent run, not a conversation

Worth pinning down before anything else, because it is the most common misread
of this module. Every example in `datasets.py` is a **single user message**, and
`run_agent` does one `agent.invoke`. So "trajectory" means the ordered tool
calls *inside one turn* — multi-**step**, single-**turn**. None of this module's
evals run a multi-turn thread. Threads first appear in [Module 5](../module_5_online_evals/),
on live traffic, where LangSmith's Trajectory view reads them (see the end of this file).

The distinction matters the moment you take these evaluators to a real
conversational agent, because the four below do not all mean the same thing at
thread scope:

| Evaluator | At turn scope (here) | Pointed at a whole thread |
|---|---|---|
| `trajectory_exact_match` | strict path check | almost always 0 — threads don't repeat |
| `required_tools_used` | did this turn do the work? | **quietly weaker** — one call anywhere satisfies it forever |
| `no_forbidden_tools` | did this turn stay safe? | still meaningful; one violation anywhere fails |
| `trajectory_efficiency` | wasted steps this turn | conflates turns; needs per-turn normalising |

`required_tools_used` is the one that bites. It is a **task-completion** metric,
not a safety precondition. If your rule is "the agent must call
`get_certifications` before *any* risky claim", a set-membership check over a
thread passes an agent that looked things up in turn 1 and then improvised
through turns 2–4. For a precondition you want a *conditional, per-turn* check
— the hard-fail shape of `no_forbidden_tools` plus the trigger condition in
`tool_evals.correct_employee_id`, which only applies when the risky action
actually happened.

The structural fix, if you go multi-turn: make each eval example **be a turn**,
with the prior messages as input context. Then the run boundary and the
assertion boundary are the same thing and the question stops arising.

## Three things to evaluate about a trajectory

| Lens | Question | Evaluators |
|------|----------|------------|
| **Path** | Did it call the right tools, in a sensible order, without waste or forbidden actions? | `trajectory_evals.py` |
| **Arguments** | Did it call those tools *correctly* (right employee_id, valid enums)? | `tool_evals.py` |
| **Judgment** | Holistically, was this a reasonable way to handle the request? | `llm_trajectory_judge.py` |

## What's here

| File | What it teaches |
|------|-----------------|
| `datasets.py` | 4 multi-step onboarding tasks. Ground truth = `expected_trajectory`, `required_tools`, `expected_employee_id`, `forbidden_tools`. |
| `trajectory_evals.py` | Deterministic path checks: exact match, required-tools subset, **no-forbidden-tools (safety)**, efficiency. |
| `tool_evals.py` | Deterministic arg checks: correct `employee_id` propagation, well-formed args (enum validation). |
| `llm_trajectory_judge.py` | LLM judge over the tool sequence for cases with no single "right" path. |
| `run_eval.py` | One agent run feeds all three lenses. |
| `mock_datasets.py`, `mocked_eval.py` | Hand-written worlds that never existed: outages, unknown employees, partial failures. |
| `replay_datasets.py`, `replay_evals.py`, `replay_eval.py` | A frozen world that *did* exist, and what happens to ground truth when the data moves. **Start here if stale eval data is your problem.** |
| `snapshot_trace.py` | Turn a production trace into a replayable dataset example. |

How the trajectory is captured from the agent lives in
[`hr_agent/trajectory.py`](../hr_agent/trajectory.py) — it walks the result
messages and pulls each `tool_call`'s name and args.

## Run it

```bash
# Self-test the deterministic evaluators — no API key needed.
python module_3_agent_evals/trajectory_evals.py
python module_3_agent_evals/tool_evals.py

# The full agent experiment.
python module_3_agent_evals/run_eval.py
```

## Key ideas to land

1. **The final answer can lie about the process.** Trajectory evals are how
   you verify the agent did the right *things*, not just said the right words.
2. **Exact-match is brittle; layer softer lenses.** `required_tools_used`
   (order-agnostic) and `trajectory_efficiency` (extra-step penalty) tell you
   *how* a non-exact run differs, instead of a flat fail.
3. **Forbidden-tool checks are safety tests.** "Did NOT do X" is often more
   important than "did Y" — e.g. don't provision equipment on a read-only
   benefits question.
4. **Wrong args are a distinct failure mode.** Right tool + wrong `employee_id`
   = right path, wrong outcome. `tool_evals.py` catches it.
5. **Use an LLM judge when there's no canonical path.** For open-ended tasks,
   `trajectory_is_reasonable` grades judgment without you pre-writing every
   acceptable sequence.

> Production tip: [`openevals`](https://github.com/langchain-ai/openevals) ships
> a `trajectory` evaluator family (strict / unordered / superset / subset) plus
> an LLM trajectory judge. Same ideas as here, batteries included.

Next: **Module 4** — wire these experiments into CI so a regression fails the
build before it reaches `main`.

---

## Mocking tool outputs (`mock_datasets.py` + `mocked_eval.py`)

Everything above evaluates the agent against its **real** tools. Those tools
always succeed — they read static data — so there is an entire class of behavior
we cannot reach: what does the agent do when a tool *fails*?

That's what [`hr_agent/mocking.py`](../hr_agent/mocking.py) is for. It's a
LangChain `AgentMiddleware` that overrides `wrap_tool_call` to return a canned
`ToolMessage` instead of executing the tool:

```python
from hr_agent import DatasetDrivenMockMiddleware, run_agent

mock = DatasetDrivenMockMiddleware({
    "lookup_employee":   {"employee_id": "E1007", "full_name": "Jordan Lee"},
    "create_it_account": {"status": "failed", "error": "IT system down"},
})
result = run_agent("Create an email account for Jordan Lee.", middleware=[mock])
```

The agent's own code is untouched — only the world around it changes. That's
what keeps this an eval of the agent rather than an eval of a different app.

### Why mock at all

| Reason | What it buys you |
|--------|------------------|
| **Coverage** | Test states the real tools can't produce — outages, partial failures, empty results. **The main reason here.** |
| **Determinism** | A flaky downstream service becomes a flaky eval score. |
| **Isolation** | You're grading the agent's reasoning, not the tools. |
| **Cost & speed** | No real API calls. |
| **Offline/CI** | CI usually has no route to production systems. |

### Where the mocks live

Each example in `mock_datasets.py` carries its own mock table under
`inputs["tool_outputs"]`, so one dataset row fully describes one world state.

> **Why `inputs` and not `outputs`?** `client.evaluate` only passes `inputs`,
> `attachments`, and `metadata` to a target function — there is no `example`
> argument (see `langsmith.evaluation._runner._get_target_args`). Mocks are also
> conceptually inputs: they describe the *scenario*, not the ground truth.

### The evaluator that matters

`reports_tool_failure` (in `tool_evals.py`) catches the expensive failure mode:
a tool returns `{"status": "failed"}` and the agent cheerfully tells the user
their account is ready. You cannot catch that without mocking.

```bash
python module_3_agent_evals/mock_datasets.py   # create the dataset
python module_3_agent_evals/mocked_eval.py     # run the experiment
```

### A second opinion on `reports_tool_failure`

That evaluator is a phrase list. `python module_3_agent_evals/mocked_eval.py --system1`
adds a System 1 model's answer to the same question, with a `system1_needs_review`
key for low-confidence answers. It sends text to a third party, so it's opt-in — and
you should measure it first; see [the decision guide](../module_2_single_turn/choosing-an-evaluator.md#try-it-on-this-repo).

### Strict vs permissive

`DatasetDrivenMockMiddleware(..., strict=True)` raises if the agent calls a tool
the example didn't mock. Use that in CI, where reaching a real system would be a
bug. `strict=False` falls through to the real tool.

---

## Stale data: when the world changes under your ground truth

An agent's outcome is only valid **at a point in time**. Say it investigates a
record, recommends a fix, and the fix is applied. Re-run the same agent next week
and the data is now correct: it finds nothing wrong and produces a different —
equally right — result. A ground truth written from the first run now fails a
healthy agent. Your regression suite decays as the world moves, and you can't
tell a regression from drift.

Trajectory evals alone don't fix this. They grade the path, and the path *also*
depends on the data the agent found.

### The experiment

```bash
python module_3_agent_evals/replay_evals.py     # self-test, no keys
python module_3_agent_evals/replay_eval.py      # both worlds, prints the comparison
```

`replay_eval.py` runs the **same agent** twice over `replay_datasets.py`:

| World | What the tools return |
|---|---|
| `recorded` | exactly what they returned when the trace was recorded (`TraceReplayMiddleware`) |
| `drifted` | what the live system says "today" — a start date moved, a plan reclassified |

and scores three things:

| Evaluator | Kind | Recorded | Drifted |
|---|---|---|---|
| `matches_snapshot_calls` | **absolute** — "should schedule 2026-06-15" | pass | **fail** |
| `args_follow_tool_outputs` | **invariant** — "date must equal what the lookup returned" | pass | pass |
| `replay_fidelity` | did the agent stay on the recorded path? | see below | n/a |

The agent is behaving correctly in both columns. The absolute check fails anyway,
because its answer key describes a world that no longer exists. That is the
failure to learn to recognize.

### Two remedies

1. **State the expectation as an invariant** when you can. It never names the
   value, so there is nothing to go stale, and it needs no recording. This is the
   cheaper fix — reach for it first.
2. **Freeze the world** when the right answer really is a specific value. Record
   what the tools returned during a real trace and replay it:

```python
from hr_agent import TraceReplayMiddleware, recorded_calls_from_messages, run_agent

recorded = recorded_calls_from_messages(production_result)   # or ...from_run(run)
replay = TraceReplayMiddleware(recorded)
result = run_agent(question, middleware=[replay])
replay.report()     # {"recorded": 2, "calls": 2, "misses": [], "unused": 0}
```

The tools are never executed; the *agent* — prompt, model, tool selection — is
re-run live. Now a score change means the agent changed.

### How replay differs from mocking

| | `mocking.py` | `replay.py` |
|---|---|---|
| Returns | one hand-written payload per tool **name** | the recorded output for each specific **call** |
| Matches on | tool name | tool name **and arguments** (case-insensitive strings) |
| Models | worlds that never existed | a world that did, as it was |
| On an unknown call | strict: raise, or fall through to the real tool | **miss**: explicit error result, logged, never guessed |

Matching on arguments is what lets two `lookup_employee` calls for two different
people each get the right record.

### Read `replay_fidelity` first

A replay **miss** means the agent asked for something the recording can't answer —
it took a different route than in the trace. That's not a bug in the replay; it's
the behavior change you were looking for (your prompt edit sent it somewhere new).
It also means every other score on that example was measured against an error
result, not the recorded world. So a miss is a *finding*, not noise. Two policies:

- `on_miss="fail"` (default) — the call gets `{"error": "Replay miss …"}`; you score
  `replay_fidelity` and investigate.
- `on_miss="live"` — fall through to the real tool. This *mixes* frozen and live
  data; treat those runs as suspect.

### Caveats worth saying out loud

- **Replay freezes the tools, not the agent's non-determinism.** The model still
  varies run to run; you're removing one source of noise, not all of it.
- **It only covers paths the recording covers.** A trace is one path through the
  agent. Replaying it tests that path against the current agent, not the space of
  paths. Record more traces, and use invariants for the rest.
- **Recordings age too.** A frozen world is a snapshot; it stays valid because it is
  *labeled* as one. `snapshot_trace.py` records an `as_of` date on every example.
  Re-record when the real system's behavior changes in a way you care about.
- **A recording is a copy of production data.** For a real HR agent that means
  employee records. Nothing here redacts them; see the PII notes in
  [Module 6](../module_6_improving_evals/).

### Getting a recording from a real trace

```bash
# Inspect first — dry run is the default; nothing is written.
python module_3_agent_evals/snapshot_trace.py --run-id <root-run-uuid>

# One turn of a multi-turn thread (each turn's output holds the whole history).
python module_3_agent_evals/snapshot_trace.py --run-id <uuid> --last-turn

# Write it as a dataset example (lands in the `scratch` split).
python module_3_agent_evals/snapshot_trace.py --run-id <uuid> --add-to-dataset
```

The example is created with **empty outputs and `needs_review: true`** — what the
agent *should* have done is a human's call. Add `snapshot_calls` and/or
`derived_args` in the UI, then move it to `gate`. LangSmith stores serialized
messages in a few shapes; `recorded_calls_from_run` is lenient about them, but run
the dry run on one of *your* traces before trusting it.

---

## Reading a trajectory in LangSmith

Everything above computes trajectories in code. LangSmith also renders them: the
**Trajectory view** shows a whole session as one readable path — user messages,
model replies, tool calls and results, each once, in order — instead of a tree of
nested runs. Click any step to open its underlying trace.

It needs traces grouped into **threads** (a shared `thread_id`). This module's
evals are single-turn, so there's nothing to group here; generate real threads
with [Module 5](../module_5_online_evals/)'s traffic script, then open one. The
same view is where you'd eyeball *why* a `trajectory_efficiency` score is low.

Code-side, `hr_agent.run_conversation(turns)` is how a thread gets made: one
`invoke` per turn, sharing a checkpointer and a `thread_id`.

When you later point these evaluators at a whole thread rather than one turn, the
table at the top of this file is the warning label.

