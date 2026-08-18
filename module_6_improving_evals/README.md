# Module 6 — Improving Evals (Annotation & Few-Shot)

> Goal: when your LLM judge disagrees with your humans, fix it — without
> fine-tuning. Capture human judgment with an **annotation queue**, then **align**
> the judge with a handful of those labels as few-shot examples.

An LLM judge is only useful if its scores track what your team actually considers
good. Out of the box they often don't: a generic judge grades against a generic
notion of quality and misses your house style, your edge cases, your bar. A judge
you can't trust is worse than no judge — it gives wrong answers *confidently*.

The loop that fixes this:

```
   production traces
        │  queue the ones worth a look
        ▼
   annotation queue  ──►  human labels (ground truth)
        │                      │
        │                      ├──►  measure judge agreement ("evaluate the evaluator")
        │                      └──►  feed back as few-shot examples ──►  aligned judge
        ▼
   curate the best into your offline dataset (the flywheel from Module 5)
```

## What's here


| File                  | What it teaches                                                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `annotation_queue.py` | Create a queue and push the **flagged** production traces to it for human review. (Picks the traces your online evals scored low — `score_traces.py`'s feedback — falling back to the deterministic checks.) |
| `few_shot_judge.py`   | One judge ("does this meet our HR house style?") that runs zero-shot (rubric only) or few-shot (rubric + human-labeled examples).                                                                            |
| `judge_alignment.py`  | **Evaluate the evaluator**: measure how often the judge agrees with humans, zero-shot vs few-shot, on a held-out label set.                                                                                  |
| `curate_dataset.py`   | **Close the flywheel**: promote flagged/labeled production traces into a curated offline dataset, each example linked to its source trace.                                                                    |


## Run it

```bash
# Measure judge↔human agreement, zero-shot vs few-shot (needs a model key).
python module_6_improving_evals/judge_alignment.py

python module_6_improving_evals/few_shot_judge.py
# Push flagged production traces to an annotation queue for labeling.
# (Run module_5_online_evals/production_traffic.py first to create traces.)
python module_6_improving_evals/annotation_queue.py
#   --all     queue recent traces, not just the flagged ones
```

`judge_alignment.py` prints the zero-shot agreement, the few-shot agreement, and
the lift. The few-shot examples it uses are a held-out slice of the same human
labels — exactly what you'd harvest from the annotation queue.

## Closing the flywheel: traces → dataset

The diagram above ends with "curate the best into your offline dataset." That
last arrow is `curate_dataset.py`:

```bash
# Traces your online evals scored below 1 — the failures worth locking in.
python module_6_improving_evals/curate_dataset.py --flagged

# Traces a human labeled in the annotation queue — the gold ones.
python module_6_improving_evals/curate_dataset.py --labeled

# See what would be added without writing anything.
python module_6_improving_evals/curate_dataset.py --all --dry-run
```

It writes to `hr-onboarding/policy-qa/from-production`, and every example
carries **`source_run_id`** — so LangSmith links each dataset row back to the
exact production trace it came from. Click through in the UI and you're looking
at the original run.

Two properties worth noting:

- **Idempotent.** It skips traces already in the dataset (matched on
  `source_run_id`), so you can run it nightly without piling up duplicates.
- **`needs_review: true`.** The production answer becomes the *starting*
  reference, not ground truth. A reviewer edits it in the UI — that editing step
  is what "curation" means. An unreviewed production answer is just the agent
  grading its own homework.

Once curated, that dataset is a permanent regression test: a failure that
happened once in production can never quietly happen again.

### ⚠️ Curation is a data-governance decision, not just a data-plumbing one

Everything above is mechanically easy, which is exactly the risk. Promoting a
trace changes its blast radius: a raw trace lives in one project under that
project's access rules, while a dataset gets shared across a team, copied
between workspaces, and kept indefinitely as a regression suite.

Our system under test is an **HR agent**. In a real deployment its traces carry
employee PII — names, employee IDs, salaries, medical and benefits details. The
workshop's traffic is synthetic, so nothing here is sensitive; your production
traffic will not be.

`curate_dataset.py` **does not redact anything.** It copies the question and
answer verbatim (truncating only) and links each example back to its source run.
That's deliberate — half-working automatic redaction is worse than none, because
it invites trust it hasn't earned. So when you point this at real traffic:

| Practice | Why |
|---|---|
| `--dry-run` first | See exactly what would be copied before it's copied. |
| Prefer `--labeled` over `--flagged` | A human already opened those traces in the annotation queue. |
| Clear the `needs_review` flag deliberately | It marks "a person has not yet checked this example" — including for PII, not just for answer quality. |
| Match the dataset's access controls and retention to the source project | The dataset is not a lower-sensitivity copy just because it's smaller. |

The general rule: **the annotation queue is the right place for a human to see
production content, and curation should follow review rather than replace it.**

## Key ideas to land

1. **Evaluate the evaluator.** Before trusting an LLM judge, measure its agreement
  with human labels. Agreement *is* the judge's accuracy.
2. **Annotation queues are how human judgment scales.** Queue the traces worth a
  human's time (the flagged ones), label them once, reuse the labels everywhere.
3. **Few-shot beats prompt-wrangling for alignment.** Showing the judge real
  labeled examples moves it toward your reviewers faster than rewriting the
   rubric — and needs no fine-tuning.
4. **Never score the judge on its own few-shot examples.** Hold out a separate
  slice, or you're measuring memorization, not alignment.
5. **Labels are reusable.** The same human labels measure the judge, align the
  judge, and seed the offline dataset.
6. **A production failure should only surprise you once.** Curating it into a
  dataset (`curate_dataset.py`) turns an incident into a permanent regression
  test.

## How this connects to LangSmith

- **Annotation Queues** (UI: *Annotation Queues*) are first-class. You can attach a
**rubric** to a queue so reviewers grade against consistent criteria; their
feedback lands on the runs as scores you can query with `client.list_feedback`.
- **Corrections → few-shot, automatically.** When a reviewer corrects a run in a
LangSmith dataset, those corrections can be served back as few-shot examples to
the judge — the productized version of what `few_shot_judge.py` does by hand.
- We build the alignment loop by hand here so the mechanics are clear. In practice,
pair queues + datasets + `[openevals](https://github.com/langchain-ai/openevals)`
judges, which already accept few-shot examples.

Next: **Module 7** — run the reference-free evals from Module 5 on a schedule as a
production monitor that alerts when quality drifts.