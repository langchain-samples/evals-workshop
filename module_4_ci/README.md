# Module 4 — Evaluations in CI

> Goal: turn the evals from Modules 2 & 3 into an automated gate that fails the
> build when quality regresses — so prompt/model/tool changes are caught in the
> PR, not in production.

## Two gating patterns (use both)

| Pattern | File | What it gives you | Best for |
|---------|------|-------------------|----------|
| **Per-example gate** | `test_evals.py` | One pytest case per dataset example. CI shows *exactly which* example broke. | Deterministic, safety-critical checks you want to hard-block on. |
| **Aggregate gate** | `ci_gate.py` | Mean score per metric vs a threshold; non-zero exit fails the build. | Dataset-level quality bars, including fuzzy LLM-judge metrics tracked as trends. |

Three suites are available to the aggregate gate: `single_turn`, `agent`, and
`tool_failures` (the mocked failure scenarios from Module 3 — no real systems
touched, so it's safe and cheap to run on every PR).

A good rule: **hard-assert the deterministic/safety metrics per-example**
(never call a forbidden tool, always use the right employee id), and **gate the
fuzzy quality metrics in aggregate** (correctness ≥ 0.8 across the set), since
a single LLM-judge call can be noisy but the mean is stable.

## Run locally

```bash
# Per-example tests
pytest module_4_ci/test_evals.py -v --langsmith-output

# Aggregate gates (exit non-zero on regression)
python module_4_ci/ci_gate.py --suite single_turn
python module_4_ci/ci_gate.py --suite agent
python module_4_ci/ci_gate.py --suite tool_failures
```

## The GitHub Actions workflow

[`.github/workflows/evals.yml`](../.github/workflows/evals.yml) is the reference
gate: it runs all four checks in one job.

> **It is manual-only in this repo.** The workflow ships as teaching material,
> so its triggers are `workflow_dispatch` (Actions tab → Run workflow) and
> nothing else. Every run makes real, paid model calls and needs secrets — a
> workshop repo that auto-ran on every PR would bill anyone who forked it and
> fail for anyone who hadn't configured keys.
>
> **To make it a real gate in your repo:** add `LANGSMITH_API_KEY` and
> `ANTHROPIC_API_KEY` under **Settings → Secrets and variables → Actions**,
> uncomment the `pull_request:` / `push:` triggers at the top of the file, and
> make it a required status check in branch protection. All three steps matter —
> a workflow that runs but isn't required doesn't gate anything.

Experiments are tagged with the commit, branch, author, and model so a
regression traces straight back to the PR that caused it — but *how* that
happens differs by gate, and it's a common trip-up:

| Gate | How it gets tagged |
|------|--------------------|
| `pytest` (`test_evals.py`) | `LANGSMITH_EXPERIMENT` names it; `LANGSMITH_EXPERIMENT_METADATA` (JSON) and the `experiment_metadata=` mark argument tag it. |
| `ci_gate.py` | `metadata=experiment_metadata()` passed to `client.evaluate`. |

> **`LANGSMITH_EXPERIMENT` is read only by the langsmith pytest plugin.** Setting
> it does nothing for experiments created by `client.evaluate` — those need
> `metadata=` passed explicitly. See `config.experiment_metadata()`.

## Splits: exclude scratch, don't include `test`

Both gates run against **every example except the held-out splits** (`train`).
The `train` slice exists so you can tune prompts and few-shot judges without
tuning against your own gate.

Note the direction — it is the whole point. The obvious version of this is
`--split test`, and it is a **fail-open** gate: LangSmith puts every example
with no explicit split into the implicit `base` split, so each example a
teammate adds through the web UI is silently dropped from the run. The gate
keeps passing while its coverage shrinks, which is the worst way for a gate to
fail, because it is indistinguishable from health.

Excluding scratch instead means a forgotten split shows up as a *false failure*
— loud and fixable — rather than a *missed test*.

```bash
# Default: gate everything except `train`; unassigned examples are gated.
python module_4_ci/ci_gate.py --suite single_turn

# Hard-fail if anyone left an example without a split.
python module_4_ci/ci_gate.py --suite single_turn --require-splits

# More scratch splits.
python module_4_ci/ci_gate.py --suite single_turn --held-out train --held-out wip

# Narrow to one split on purpose (opt-in; skips everything else).
python module_4_ci/ci_gate.py --suite single_turn --split test
```

Every run prints its accounting, so coverage drift is visible in the CI log:

```
Dataset 'hr-onboarding/policy-qa/v1': 8 examples — 7 gated, 1 held out (train),
2 with no split assigned.
  note: 2 example(s) have no split assigned, so LangSmith put them in 'base'.
        They ARE being gated — assign them a split to be explicit.
```

## Response caching: fast locally, off in CI

`conftest.py` points `LANGSMITH_TEST_CACHE` at `fixtures/cassettes/`, so local
`pytest` runs record model API calls (via `vcrpy`) and replay them afterwards.
Measured on this suite:

| Run | Time |
|-----|------|
| Cold (recording) | ~45 s |
| Warm (partial — agent took a new path) | ~15 s |
| Warm (fully cached) | **~1.4 s** |

Roughly a 30× loop speedup, and free. Worth knowing about the middle row:
`record_mode` is `new_episodes` and requests are matched on the **body**, so
until the cassette covers every path the agent takes, some runs still make live
calls (and can still be flaky). It settles after a couple of runs.

**Disabled when `$CI` is set — deliberately.** A gate that replays yesterday's
recorded responses cannot detect that today's model regressed; it would pass
forever and tell you nothing. Caching is a development-speed tool; the gate must
make real calls. `WORKSHOP_NO_CACHE=1` forces real calls locally too.

### The `cached_hosts` trap

The obvious way to cache only model traffic is
`@pytest.mark.langsmith(cached_hosts=["api.anthropic.com"])`. **Check that it
actually records anything.** Model calls don't always leave for the provider's
own domain — a gateway, proxy, or corporate egress in front of the provider
changes the host your process really connects to. The filter then matches
nothing and caching looks enabled while recording zero requests. An empty
`fixtures/cassettes/` after a run is the tell.

Don't guess the host. Record once with no filter, then read it off the cassette:

```bash
grep -m1 'uri:' fixtures/cassettes/*.yaml
```

So the default here passes **no** host filter: the langsmith plugin already
excludes the LangSmith API itself (`ignore_hosts=[client.api_url]`), so traces
and feedback are always written for real while everything else is cached. Set
`WORKSHOP_CACHED_HOSTS=host1,host2` once you've confirmed the real hosts.

Cassettes are **gitignored**. No credentials reach disk (langsmith's VCR config
strips every request header before recording), but responses carry
workspace-identifying headers, and cassette filenames are keyed on the LangSmith
*dataset UUID* — so a cassette recorded in one workspace would never replay in
another anyway. Delete the directory to re-record.

## Setting thresholds (the hard part)

Thresholds live in `THRESHOLDS` in `ci_gate.py`. Guidance:

- **Safety/format metrics → 1.0.** `no_forbidden_tools`, `correct_employee_id`,
  `tool_args_well_formed`, `response_not_empty` should never regress.
- **Quality metrics → a touch below current baseline.** Run the suite a few
  times on `main`, take the stable mean, set the threshold slightly under it.
  Too tight and flaky judges block good PRs; too loose and real regressions slip
  through.
- **Trajectory exact-match → keep low (or off as a gate).** It's a useful
  *signal* but too brittle to block merges on; prefer `required_tools_used`.
- **Watch cost & time.** Evals call models. Keep the CI dataset small and
  curated (10–50 sharp examples), and run the big nightly suite on a schedule,
  not on every PR.

## Common patterns beyond this repo

- **Regression vs. baseline**, not absolute threshold: compare this PR's
  experiment to `main`'s and fail only on a *drop* (LangSmith stores both).
- **Sampling on every PR, full suite nightly** (a scheduled workflow) to balance
  signal against cost.
- **Block on safety, report on quality**: hard-fail safety metrics; post quality
  deltas as a PR comment for human review rather than auto-blocking.

---

## 🗣️ Live discussion (facilitator-led — not in code)

> These are conversation prompts for the session, intentionally left for the
> presenter rather than fabricated here:
>
> - **How we run evals in CI internally** — Eric to walk through the real
>   pipeline: what gates on PR vs. nightly, how thresholds/baselines are set,
>   how flaky-judge noise is handled, and cost controls.
> - **Partner success stories** — examples from other teams: what they gated
>   on, what regressions evals caught before release, and lessons learned.
> - **Open Q&A** — map these patterns onto *your* services and existing CI.
