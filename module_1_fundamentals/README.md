# Module 1 — Evaluation Fundamentals

> Goal: understand the four moving parts of *any* evaluation, then run one
> end-to-end against the HR agent.

## Why evaluate at all?

You can't improve what you can't measure. Prompt tweaks, model upgrades, and
new tools all *feel* like improvements — evals tell you whether they actually
are, and catch the regressions you didn't expect. An eval is just a
**repeatable, automated answer to "is the agent doing the right thing?"**

## The four moving parts

Every evaluation — single-turn or agentic, deterministic or LLM-judged — is
built from the same four pieces:

| Part | What it is | In this workshop |
|------|-----------|------------------|
| **Dataset** | A set of *examples*, each with `inputs` and (optionally) `reference_outputs` (the expected answer / "ground truth"). | `datasets.py` in each module |
| **Target** | The thing under test. Given an example's `inputs`, produce `outputs`. | the HR agent (`run_agent`) |
| **Evaluator** | A function that scores one example: `(inputs, outputs, reference_outputs) -> score`. | `*_evals.py` files |
| **Experiment** | One run of the target over the whole dataset, scored by the evaluators. Results are tracked in LangSmith. | `client.evaluate(...)` |

```
            ┌──────────┐
 inputs ───▶│  TARGET  │───▶ outputs ──┐
            └──────────┘                │
                                        ▼
 reference_outputs ──────────────▶ ┌───────────┐
                                   │ EVALUATOR │──▶ score
                                   └───────────┘
            (repeat over every example in the DATASET = one EXPERIMENT)
```

## Two families of evaluators

This distinction drives the rest of the workshop:

- **Deterministic / code evaluators** — plain functions. Exact match, regex,
  "is the JSON the right shape?", "did it call the right tools?". Fast, free,
  100% reproducible. Use them wherever the correctness criterion is objective.
- **LLM-as-judge evaluators** — use a model to grade fuzzy qualities:
  correctness against a reference, groundedness, tone, helpfulness. Use them
  when "right" is a matter of meaning, not string equality.

> Rule of thumb: **reach for a deterministic evaluator first.** Only use an LLM
> judge for things a deterministic check genuinely can't capture.

## Run it

```bash
# from the repo root, with your .env set up
python module_1_fundamentals/01_first_eval.py
```

`01_first_eval.py` is deliberately tiny: a 3-example dataset, the HR agent as
the target, and one deterministic evaluator ("did the answer mention the
expected number of vacation days?"). It prints a link to the experiment in
LangSmith. Open it and look at the per-example scores — that view is where
you'll live during the rest of the workshop.

Once this clicks, move on to **Module 2**, where we add real deterministic
shape checks and LLM-as-judge evaluators.

---

## A note on the dataset name

This module's dataset is called `hr-onboarding/policy-qa/intro`, not
"HR Workshop — Module 1". The convention is
`{domain}/{capability}/{version_or_variant}`, and it exists for the same reason
you namespace anything: once you have forty datasets, prose titles are
unsearchable and slash-separated ones sort into a tree. Module 2 covers the rest
of the dataset conventions (splits and metadata).
