# Module 7 — Evals in CI/CD (Production Monitoring)

> Goal: gate not just your *pull requests* but your *production behavior*. Run the
> reference-free evals from Module 5 on a schedule, compare to a baseline, and
> alert when quality drifts.

Module 4 put evals in CI as an **offline gate on a PR**: did this change regress my
curated dataset? That's necessary but not sufficient. Plenty of regressions never
show up in a diff — a provider silently updates the model behind your API, a
retrieval source goes stale, real-world questions shift. Your offline suite is
green and production is quietly getting worse.

The fix is a second gate, on the other side of the release: an **online monitor**
that watches live traffic over time.

| | **Offline gate** (Module 4) | **Online monitor** (this module) |
|---|---|---|
| Trigger | every PR / push | a schedule (cron) |
| Shipped here as | manual-only (`workflow_dispatch`) | manual-only (`workflow_dispatch`) |
| Data | curated dataset | recent production traces |
| Compares to | per-metric thresholds | a baseline (drift detection) |
| Catches | bad merges | silent production degradation |
| On failure | blocks the merge | fires an alert |

## What's here

| File | What it teaches |
|------|-----------------|
| `monitor.py` | Pull the last N production traces, score them with Module 5's reference-free evals, and **exit non-zero if any metric drifted below baseline**. |
| `baseline.json` | The committed baseline means + drift `tolerance`. Regenerate after an intentional, reviewed quality change. |
| `../.github/workflows/online-evals.yml` | Runs the monitor on demand. The daily `schedule:` cron is **commented out** — uncomment it to arm the monitor for real. |

## Run it

```bash
# Check the last 50 production traces against baseline.json.
python module_7_production_ci/monitor.py
#   --deterministic-only   skip LLM judges (no model cost)
#   --limit 100            score more traces

# Recapture the baseline after an intentional, reviewed quality change.
python module_7_production_ci/monitor.py --write-baseline
```

(Generate traffic first with `module_5_online_evals/production_traffic.py` if the
project is empty.)

## Setting the baseline & tolerance (the judgment call)

- **Baseline** = the metric means on known-good traffic. Capture it once with
  `--write-baseline`, eyeball the numbers, and commit `baseline.json`. Re-capture
  only after a *deliberate, reviewed* quality change — never to silence an alert.
- **Tolerance** = how far below baseline is "drift, not noise." Reference-free LLM
  judges wobble run-to-run, so a single trace dipping means nothing; the mean over
  many traces is stabler. The shipped baseline uses `0.15` because the demo runs on
  only ~10 traces (where the judges wobble a lot); tighten it toward `0.05–0.1` as
  you average over more traffic and learn each metric's real noise floor.
- **Deterministic metrics deserve tight tolerance.** `response_not_empty` dropping
  at all is a real incident; `professional_tone` drifting 0.05 is probably noise.

## Key ideas to land

1. **Two gates, two jobs.** Offline gate blocks bad *merges*; online monitor
   catches *production* degradation the diff never showed.
2. **Drift vs. baseline, not absolute thresholds.** In production you care about
   *change* — "groundedness fell" matters more than its exact value.
3. **Reuse the online evaluators.** The monitor runs Module 5's reference-free
   checks verbatim — score one trace online, aggregate many on a schedule.
4. **Mind cost and cadence.** LLM judges cost money per trace. Sample, run on a
   schedule (not per-request), and use `--deterministic-only` for a free
   fast-path signal between full runs.

## Doing this for real: LangSmith monitors & automations

The scheduled script is portable and CI-native, but LangSmith also does much of
this server-side, no job to run:

- **Monitors / dashboards** chart feedback scores (the ones `score_traces.py`
  writes) over time, so drift is visible at a glance.
- **Automations / rules** (UI: *Tracing project → Rules*) can run an evaluator on a
  sampled fraction of incoming traces automatically and trigger actions — add to a
  dataset, add to an annotation queue, send a webhook — when a condition is met.
- A common production setup: a **rule** auto-scores 10% of traffic online; a
  **monitor** alerts on score drift; this **scheduled job** is the CI-owned
  backstop and the thing your pipeline can block a release on.

---

This is the end of **Session 2**. You now have the full production loop: offline
experiments (Session 1) → online evals on live traces (Module 5) → human
annotation + judge alignment (Module 6) → a scheduled production monitor (Module
7), all feeding the dataset flywheel.
