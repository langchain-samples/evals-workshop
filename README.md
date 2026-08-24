# Evaluations Workshop — from Fundamentals to Production

A hands-on, progressive introduction to evaluating LLM applications, built
around a single example app: an **HR Onboarding agent**. You learn the app
once, then spend the rest of the workshop on evaluation technique — from a
first single-turn eval all the way to monitoring evals in production.

Stack: **Python + LangChain (`create_agent`) + LangSmith**.

The workshop runs in two sessions:

- **Session 1 — Foundations (Modules 1–4):** what an eval is, single-turn and
  agent evals, and gating a PR with an offline suite.
- **Session 2 — Evals for Production (Modules 5–7):** online evals on live
  traces, improving evaluators with annotation + few-shot, and monitoring
  production for quality drift.

---

## Who this is for

Development teams meeting evaluations for the first time. We start with
concepts and build up. No prior eval experience assumed; basic Python and
comfort with the command line is enough.

## The arc

```
SESSION 1 — Foundations                                    SESSION 2 — Evals for Production
Module 1          Module 2            Module 3          Module 4   │  Module 5            Module 6              Module 7
Fundamentals  ->  Single-turn evals -> Agent evals  ->  Evals in   │  Online evals    ->  Improving evals   ->  Production
(the 4 parts)     (judge the answer)  (the process)     CI (gate)  │  (live traces)       (annotate+few-shot)    monitor (drift)
```

### Session 1 — Foundations

| Module | You learn to… | Evaluators introduced |
|--------|---------------|------------------------|
| **[1 — Fundamentals](module_1_fundamentals/)** | Name the 4 parts of any eval; run one end-to-end. | first deterministic check |
| **[2 — Single-turn](module_2_single_turn/)** | Judge a single answer. | deterministic (facts, **shape validation**) + LLM-judge (correctness, groundedness, tone) |
| **[3 — Agent evals](module_3_agent_evals/)** | Judge the *trajectory* and tool calls, not just the answer. Mock tool outputs to reach failure states. | trajectory (exact / required / forbidden / efficiency), tool-args, LLM trajectory judge, **failure handling** |
| **[4 — CI](module_4_ci/)** | Gate a build on eval results. | pytest per-example gate + aggregate threshold gate + GitHub Actions |

### Session 2 — Evals for Production

| Module | You learn to… | What's new |
|--------|---------------|------------|
| **[5 — Online evals](module_5_online_evals/)** | Tell offline experiments from online evals; score *live traces* with no ground truth. | reference-free evaluators, scoring traces + writing feedback, the data flywheel |
| **[6 — Improving evals](module_6_improving_evals/)** | Align an LLM judge to your humans; promote traces into a dataset. | annotation queues, "evaluate the evaluator", **few-shot judge alignment**, **production→dataset curation** |
| **[7 — Production CI](module_7_production_ci/)** | Monitor production for quality drift. | scheduled monitor, baseline/drift alerting, scheduled GitHub Actions |

The same HR agent (`hr_agent/`) is the system-under-test in every module.

## The example app — HR Onboarding agent

Lives in [`hr_agent/`](hr_agent/). A tool-calling agent that:
- answers HR **policy/benefits questions** (single-turn material), and
- performs **onboarding actions** — provisioning accounts, ordering equipment,
  scheduling orientation (multi-step trajectory material).

Tools are deterministic (they read static mock data), so the same input always
produces the same tool output. That reproducibility is what makes evaluation
meaningful — you measure the *model's* behavior, not flaky downstream systems.

The flip side: tools that always succeed can't test what happens when they
*don't*. [`hr_agent/mocking.py`](hr_agent/mocking.py) is middleware that swaps
tool outputs for canned ones — per test case — so Module 3 can evaluate the
agent against outages, unknown employees, and partial failures the real tools
could never return.

---

## Setup

```bash
# 1. Python 3.10+ and a virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Configure keys
cp .env.example .env      # then edit: LANGSMITH_API_KEY + ANTHROPIC_API_KEY
```

You need:
- a **LangSmith API key** (smith.langchain.com → Settings → API Keys), and
- an **Anthropic API key** (or set `WORKSHOP_MODEL=openai:gpt-4o-mini` and an
  `OPENAI_API_KEY` — see `config.py` / `.env.example`).

### Smoke test (no keys needed)

The deterministic evaluators self-test as pure functions:

```bash
python module_2_single_turn/deterministic_evals.py
python module_3_agent_evals/trajectory_evals.py
python module_3_agent_evals/tool_evals.py
python module_5_online_evals/reference_free_evals.py
```

### Run a module (keys needed)

```bash
# Session 1 — offline experiments
python module_1_fundamentals/01_first_eval.py
python module_2_single_turn/run_eval.py
python module_3_agent_evals/run_eval.py
python module_3_agent_evals/mocked_eval.py           # mocked tool failures
python module_4_ci/ci_gate.py --suite agent

# Session 2 — production evals
python module_5_online_evals/production_traffic.py   # create live traces
python module_5_online_evals/score_traces.py         # online-eval loop
python module_6_improving_evals/judge_alignment.py   # zero-shot vs few-shot
python module_6_improving_evals/curate_dataset.py    # traces -> dataset (flywheel)
python module_7_production_ci/monitor.py             # drift vs baseline
```

Each prints a link/name to open the experiment (or project) in LangSmith.

> **A note on the GitHub Actions workflows.** Both ship as *reference material*
> with `workflow_dispatch` as their only trigger — they never run on a PR, a
> push, or a schedule out of the box. Eval runs make real, paid model calls and
> need API keys, so a workshop repo that fired them automatically would bill
> anyone who forked it and fail for anyone who hadn't configured secrets. Each
> file documents exactly which triggers to uncomment to arm it for real. See
> [module 4](module_4_ci/) and [module 7](module_7_production_ci/).

---

## Repo map

```
hr_agent/                  # the system under test (agent + tools + mock data)
# --- Session 1: Foundations ---
module_1_fundamentals/     # concepts + first eval
module_2_single_turn/      # deterministic + LLM-judge evaluators
module_3_agent_evals/      # trajectory + tool evaluators
module_4_ci/               # pytest gate, aggregate gate, GitHub Actions
# --- Session 2: Evals for Production ---
module_5_online_evals/     # reference-free evals + scoring live traces
module_6_improving_evals/  # annotation queues + few-shot judge alignment
module_7_production_ci/    # scheduled drift monitor + baseline
.github/workflows/evals.yml         # offline gate (reference; manual-only)
.github/workflows/online-evals.yml  # online monitor (reference; manual-only)
config.py                  # model + LangSmith config, experiment metadata
fixtures/cassettes/        # recorded model responses (VCR) for fast local tests
```

---

## Conventions this repo follows

Worth copying into your own repo — they're what keeps an eval suite usable once
it grows past a handful of datasets.

### Dataset naming: `{domain}/{capability}/{version_or_variant}`

```
hr-onboarding/policy-qa/intro             # Module 1
hr-onboarding/policy-qa/v1                # Module 2
hr-onboarding/tool-selection/v1           # Module 3
hr-onboarding/tool-failures/v1            # Module 3 (mocked)
hr-onboarding/policy-qa/from-production   # Module 6 (curated)
```

Slash-separated names sort and filter cleanly in the UI. Prose titles don't.

### Splits: hold out scratch, gate everything else

Every example gets a split. The `train` slice is where you tune prompts and
few-shot judges — tune against your gate and the gate stops measuring anything.

The subtlety is which direction you filter. Gating on `splits=["test"]` looks
equivalent and **fails open**: LangSmith files every example with no explicit
split under the implicit `base` split, so each example a PM adds through the
web UI is silently dropped from the gate. Nothing breaks, no one is told, and
coverage quietly shrinks. Gate on *everything except scratch* instead and a
forgotten split becomes a false failure — loud and fixable — rather than a test
that never ran.

```python
client.create_examples(..., splits=[e["split"] for e in EXAMPLES])

# Fail-open — an example with no split assigned is invisible here:
#   data = list(client.list_examples(dataset_name=NAME, splits=["test"]))

# Fail-safe — unassigned examples get gated:
all_examples = list(client.list_examples(dataset_name=NAME))
held_out = {e.id for e in client.list_examples(dataset_name=NAME, splits=["train"])}
data = [e for e in all_examples if e.id not in held_out]
```

`module_4_ci/ci_gate.py` does this and prints the accounting every run
(total / gated / held out / unassigned), so drift shows up in the CI log.
`--require-splits` turns "someone forgot" into a hard failure.

The names are just strings — nothing in LangSmith enforces `test`/`train`, and
no model is being trained here. `gate`/`scratch` reads better if the ML
vocabulary confuses your team; only the holdout discipline matters.

### Metadata at all three levels

Datasets get `{owner, capability, module, source}`; examples get
`{difficulty, policy_topic, source}`; experiments get commit, branch, model,
and author via `config.experiment_metadata()`.

> **Gotcha:** the `LANGSMITH_EXPERIMENT` env var is read *only* by the langsmith
> pytest plugin. It does **not** tag experiments created by `client.evaluate` —
> those need `metadata=` passed explicitly. That's what `experiment_metadata()`
> is for.

### Response caching: local yes, CI no

`module_4_ci/conftest.py` points `LANGSMITH_TEST_CACHE` at `fixtures/cassettes/`
so local test runs record and replay model calls. Measured on the module 4
suite: **~45 s cold → ~1.4 s fully cached.**

It's **disabled when `$CI` is set**: a gate that replays yesterday's responses
can't detect that today's model regressed. Cassettes are gitignored — no
credentials reach disk, but their filenames are keyed on the LangSmith dataset
UUID, so they'd never replay in another workspace. `WORKSHOP_NO_CACHE=1` forces
real calls locally. See [module 4](module_4_ci/) for the `cached_hosts` trap.

---

## Taking this to a real repo

The module-per-lesson layout here is pedagogical. In a production repo, organize
by *role* instead:

```
evals/
├── datasets/
│   ├── from_production.py       # sample traces -> dataset  (module_6/curate_dataset.py)
│   ├── synthetic_generation.py  # generate test cases
│   └── schemas/                 # pydantic models for inputs/outputs
├── evaluators/
│   ├── __init__.py              # export all evaluators
│   ├── correctness.py           # (module_2/deterministic_evals.py)
│   ├── tool_selection.py        # (module_3/trajectory_evals.py, tool_evals.py)
│   ├── safety.py
│   └── llm_judges/              # (module_2/llm_judge_evals.py)
├── fixtures/
│   ├── tool_mocks/              # (hr_agent/mocking.py + module_3/mock_datasets.py)
│   └── cassettes/               # VCR recordings
├── suites/
│   ├── regression.py            # (module_4/ci_gate.py build_suite + THRESHOLDS)
│   ├── capability_specific.py
│   └── nightly.py               # comprehensive scheduled runs
└── conftest.py                  # (module_4/conftest.py)
```

The rule behind it: **evaluators, dataset-creation scripts, orchestration, and
mocks live in code** (versioned, reviewed, testable). **Dataset storage,
experiment comparison, and human feedback live in LangSmith.**
