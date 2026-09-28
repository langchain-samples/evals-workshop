# Module 5 — Experiments vs. Tracing (Online Evals)

> Goal: understand the two places evals run — **offline experiments** on a curated
> dataset (Modules 1–4) and **online evals** on live production traces — and run
> the online loop end-to-end on the HR agent.

Everything so far has been *offline*: a fixed dataset with ground truth, scored in
an experiment before you ship. That catches regressions you can anticipate. It
says nothing about the questions real users actually ask — the typos, the
ambiguity, the out-of-scope requests your dataset never imagined. **Online evals**
close that gap by scoring production traces as they happen.

## Offline vs. online

| | **Offline experiment** (Modules 1–4) | **Online eval** (this module) |
|---|---|---|
| Data | curated dataset, fixed | live production traces |
| Ground truth | yes — `reference_outputs` per example | **none** — must be reference-free |
| When | pre-merge / pre-release | continuously, in production |
| Answers | "did this change regress the cases I curated?" | "how is the agent doing on real traffic *right now*?" |
| Output | one experiment, comparable across runs | feedback attached to each trace |

The hard constraint online is **no per-example reference**. A live trace is a
question the agent has never seen, with no pre-written answer — so an online
evaluator may use only what's in the trace (question, answer, tool calls) plus
knowledge you already hold (your policy corpus). That's why
[`reference_free_evals.py`](reference_free_evals.py) takes only `(inputs, outputs)`,
never `reference_outputs`.

## The data flywheel

Offline and online aren't rivals — they feed each other:

```
   online evals on live traces
        │  surface real failures / new question shapes
        ▼
   curate those traces into the dataset
        │
        ▼
   offline gate (Module 4) regression-tests them forever
```

A trace your online eval flags is the single most valuable thing to add to your
offline dataset — it's a real failure, not a hypothetical one.

## What's here

| File | What it teaches |
|------|-----------------|
| `production_traffic.py` | Sends messy, realistic queries — including a few **multi-turn conversations** — through the agent **with tracing on**, so traces land in a LangSmith project as **threads**. Your stand-in for production. |
| `reference_free_evals.py` | Evaluators that need **no** ground truth: `response_not_empty`, `not_deflected` (deterministic), `groundedness`, `professional_tone` (LLM judges). Self-tested. |
| `score_traces.py` | The online-eval loop: pull recent traces, score them reference-free, write the scores back as feedback. |

## Run it

```bash
# Self-test the deterministic reference-free evaluators — no API key needed.
python module_5_online_evals/reference_free_evals.py

# 1. Generate live traces (needs LANGSMITH + model keys).
python module_5_online_evals/production_traffic.py

# 2. Score those traces and write feedback back onto them.
python module_5_online_evals/score_traces.py
#    --deterministic-only   skip the LLM judges (no model cost)
#    --limit 50             score more traces
```

## Key ideas to land

1. **Online evals have no reference.** This is the defining constraint, not a
   limitation — design evaluators that judge from the trace + knowledge you hold.
2. **Watch rates, not single scores.** Online, the signal is a *trend* — the
   deflection rate creeping up, groundedness dipping after a prompt change.
3. **The flywheel beats either half alone.** Online finds the failures; offline
   locks them down so they never come back.
4. **Reference-free evaluators are reusable.** Module 7's production monitor runs
   these exact evaluators on a schedule.

## Doing this for real: server-side rules

`score_traces.py` runs the loop from your machine — perfect for learning and for
the scheduled monitor in Module 7. In production you usually don't poll at all:
you attach the evaluator to the project as a **rule / automation** so LangSmith
runs it automatically on a sampled fraction of incoming traces (e.g. 10%), with no
job to operate.

- In the UI: **Tracing project → Rules → + New Rule → Run an evaluator**, set a
  sampling rate, and pick/author the evaluator.
- From code: the `langsmith-evaluator` skill ships an `upload_evaluators.py`
  helper that registers a code evaluator as a project rule with `--project` and
  `--sample-rate`. Same reference-free evaluators, attached server-side instead
  of looped here. *(That script lives in the skill, not in this repo.)*

> Production tip: sampling matters. Scoring 100% of high-volume traffic with an LLM
> judge gets expensive fast — sample for the online signal, and run the full set
> only on curated datasets offline.

## Reading production sessions: the Trajectory view

Scores tell you *that* something went wrong. To see *what happened* you have to
read the session, and a raw trace is a poor way to do it: nested runs, model calls
inside tool calls inside agent steps. LangSmith's **Trajectory view** projects a
thread down to the conversation the agent actually had — user messages, model
replies, tool calls and their results, each shown once, in order, across the main
agent and any subagents. Click a step to drop into its underlying trace with the
nested runs, timings and retries intact.

`production_traffic.py` now sends **threads**: every lone question is a one-turn
thread, and two are real three-turn conversations. Open the project, go to
**Threads**, and open one.

```bash
python module_5_online_evals/production_traffic.py
```

**What makes a thread show up.** Two pieces of run metadata, per the
[Trajectory view docs](https://docs.langchain.com/langsmith/trajectory-view-integrations):

| Key | Meaning |
|---|---|
| `thread_id` | on every run in the conversation — groups turns into one thread |
| `ls_agent_type: "root"` | on each turn's top-level run — marks the main conversation |

`hr_agent.run_conversation()` makes a thread the standard LangGraph way: a
checkpointer plus a shared `thread_id` in the run config, one `invoke` per turn.
The docs list LangChain and LangGraph as integrations that set both keys for you.
We confirmed `thread_id` reaches the trace metadata; **we could not confirm
`ls_agent_type` from this side**, so if a thread doesn't render, open a turn's root
run → *Metadata* and check both keys are present, and set the missing one yourself.
(Other frameworks differ — e.g. with OpenAI-style clients you set `thread_id`
manually. The docs cover each.)

**Two things to know:**

- **Turn scope vs. thread scope.** Each turn is its own root run, and each turn's
  output holds the *whole history so far*. Score a turn's tool calls with
  `recorded_calls_from_messages(result, last_turn_only=True)`, not the raw
  messages — otherwise turn 3 is credited with turns 1 and 2. The table at the top
  of [Module 3](../module_3_agent_evals/README.md) is what changes when an
  evaluator sees a thread.
- **It's new and has rough edges.** The view launched in late September 2026; one
  open SDK issue reports the first turn intermittently missing
  ([langsmith-sdk #3594](https://github.com/langchain-ai/langsmith-sdk/issues/3594)).
  If turn 1 is missing, the trace still has it.

LangSmith's announcement says trajectories can be scored with online evaluators and
routed to annotation queues or datasets — the same flywheel as above, applied to
whole sessions. Look in your workspace's UI for the exact controls; we haven't
scripted that here.

**Try it.** Open the "vacation days" thread. Turn 3 asks about carry-over, which
the policy text never mentions. Does the agent say so, or invent an answer?
`groundedness` will give a number; the Trajectory view shows you *why*.

## Beyond scoring: Engine

Everything in Modules 5–6 is a loop you run by hand: find bad traces, look at them,
label a few, curate a dataset. **LangSmith Engine** runs that loop for you against a
tracing project — it clusters recurring failures into *issues*, diagnoses them, and
proposes fixes and dataset examples. It reads the scores this module writes, so
`score_traces.py`'s feedback on `hr-agent-production` is its input. The full
walk-through, what it costs, and how it maps onto what you built, is in
[Module 6](../module_6_improving_evals/README.md#engine-the-same-loop-automated).

Next: **Module 6** — when the online judges disagree with your humans, align them
with annotation and few-shot examples.
