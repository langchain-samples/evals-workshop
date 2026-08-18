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
| `production_traffic.py` | Sends messy, realistic queries through the agent **with tracing on**, so traces land in a LangSmith project. Your stand-in for production. |
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

Next: **Module 6** — when the online judges disagree with your humans, align them
with annotation and few-shot examples.
