# Module 3 — Agent Evaluations (Trajectory & Tools)

> Goal: stop judging only the final answer and start judging *how* the agent
> got there — the sequence of tool calls and the arguments it passed.

A correct-sounding final message can hide a broken process: the agent called a
destructive tool it shouldn't have, looked up the wrong employee, or took ten
steps to do a two-step job. Agent evals catch these. They're what separates
"the demo worked" from "this is safe to ship."

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

### Strict vs permissive

`DatasetDrivenMockMiddleware(..., strict=True)` raises if the agent calls a tool
the example didn't mock. Use that in CI, where reaching a real system would be a
bug. `strict=False` falls through to the real tool.
