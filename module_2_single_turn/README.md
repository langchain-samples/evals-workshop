# Module 2 — Single-Turn Evaluations

> Goal: evaluate one input → one output. Master the two evaluator families —
> deterministic code checks and LLM-as-judge — on the HR agent's policy answers.

A *single-turn* eval ignores how the agent got there; it judges the final
answer to a single question. It's the right starting point because the
techniques here (exact/fuzzy matching, shape validation, reference grading,
groundedness) are the building blocks you'll reuse in the harder agent evals.

## What's here

| File | What it teaches |
|------|-----------------|
| `datasets.py` | A 5-example policy Q&A dataset with rich ground truth (`expected_facts`, `reference_answer`, `policy_topic`). |
| `deterministic_evals.py` | Code evaluators: non-empty, required-facts (fractional), topic-drift guard, and **structured-output shape validation**. |
| `llm_judge_evals.py` | LLM-as-judge evaluators with structured output: correctness, groundedness (anti-hallucination), professional tone. |
| `structured_output.py` | A target that returns a typed object + the "is the output the right shape?" check. |
| `run_eval.py` | The full experiment: deterministic + LLM judges together. |
| `system1_alignment.py`, `system1_judge.py`, `system1_client.py`, `system1_labels.py` | **A working System 1 example**: a phrase-list check and a System 1 model (Jev) scored against 40 labeled replies on a held-out split. Runs offline by default; the System 1 side is opt-in. See [the guide](choosing-an-evaluator.md#try-it-on-this-repo). |
| `multimodal_datasets.py`, `multimodal_fixtures.py`, `multimodal_evals.py`, `multimodal_run_eval.py` | **Multimodal evals.** Expense claims whose receipts are images, with up to three files (PNG + CSV) attached to one example. Deterministic checks on what was read from each image, plus a judge that looks at the images. See [below](#multimodal-evals-images-and-attachments). |
| [`choosing-an-evaluator.md`](choosing-an-evaluator.md) | **When to use which.** A decision ladder from exact lookup to System 1 models (Jev) to LLM judge to human review, with a worked example. Read this before you write a judge. |

## Run it

```bash
# Unit-test the deterministic evaluators first — no API key needed.
python module_2_single_turn/deterministic_evals.py

# The full single-turn experiment (deterministic + LLM judges).
python module_2_single_turn/run_eval.py

# The structured-output shape-validation experiment.
python module_2_single_turn/structured_output.py

# Multimodal: receipt images as attachments.
python module_2_single_turn/multimodal_evals.py      # evaluator self-tests, no API key
python module_2_single_turn/multimodal_fixtures.py   # write the images to fixtures/receipts/ to look at
python module_2_single_turn/multimodal_run_eval.py   # the experiment
```

## Key ideas to land

1. **Deterministic first.** `mentions_required_facts` and
   `structured_answer_is_valid` cost nothing and never flake. Most "is it
   broken?" questions are objective — answer them with code. When you do reach
   for a judge, [`choosing-an-evaluator.md`](choosing-an-evaluator.md) is the
   checklist for whether you should.
2. **Fractional scores beat pass/fail** for partial credit (`2/3 facts
   present` is more actionable than a bare `0`).
3. **LLM judges need structure.** We return a Pydantic object
   (`reasoning` + verdict), never free text. The `reasoning` becomes the
   evaluator comment so a surprising score is always explainable.
4. **Groundedness ≠ correctness.** An answer can match the reference yet add a
   hallucinated detail. We grade them separately, feeding the judge the *real*
   policy text as the source of truth.
5. **Validate your evaluators.** `deterministic_evals.py` has self-tests on
   known good/bad outputs. An untrusted evaluator is worse than none.
6. **Pick the cheapest evaluator that can answer the question.** Code for facts you
   can look up or compute; a System 1 model (Jev) for narrow, typed calls about
   meaning at volume; an LLM judge only for open-ended criteria or when you need a
   rationale; humans for high-stakes calls and for checking the rest. The ladder and a
   working System 1 example are in [`choosing-an-evaluator.md`](choosing-an-evaluator.md).
   Whichever you choose, measure it against labels first.

> Production tip: LangChain's [`openevals`](https://github.com/langchain-ai/openevals)
> ships ready-made correctness/groundedness judges. We build them by hand here
> so the mechanics are clear — reach for `openevals` once they are.

Next: **Module 3**, where we stop judging the final answer alone and start
judging *how the agent got there* — the tool-call trajectory.

---

## Dataset conventions introduced here

`datasets.py` is where the repo's dataset conventions show up in full:

```python
client.create_dataset(
    dataset_name="hr-onboarding/policy-qa/v1",     # {domain}/{capability}/{version}
    metadata={"owner": ..., "capability": ..., "module": 2},
)
client.create_examples(
    inputs=[...], outputs=[...],
    metadata=[{"policy_topic": "vacation", "difficulty": "easy"}, ...],
    splits=["gate", "gate", "gate", "scratch", "scratch"],
)
```

- **Name** — slashes sort and filter cleanly once you have dozens of datasets.
- **Metadata** — dataset-level for discovery, example-level for slicing results
  ("how do we do on `difficulty: hard`?"). Both are filterable in the UI.
- **Splits** — `gate` is what CI gates on; `scratch` is space for tuning
  prompts and few-shot judges. `metadata` and `splits` are *per-example*
  sequences, positionally aligned with `inputs`/`outputs`.

Run a suite against one split:

```python
data = list(client.list_examples(dataset_name=DATASET_NAME, splits=["gate"]))
client.evaluate(target, data=data, ...)
```

`examples_for_split(client, "gate")` in `datasets.py` wraps that.

> Note this is the *fail-open* direction — fine for slicing a dataset by hand,
> wrong for a CI gate, where an example with no split assigned would be
> silently skipped. See [module 4](../module_4_ci/) and the top-level README.

---

## Multimodal evals: images and attachments

Some inputs aren't text. In the HR app, a new hire submits an expense claim: a
one-line note plus **photos of receipts**, sometimes with a **card-statement CSV**.
[`hr_agent/receipts.py`](../hr_agent/receipts.py) reads every file and returns a
typed decision: `approve`, `needs_receipt`, or `reject`. This section evaluates
that step.

### Several files in one example

The text fields go in `inputs` as usual. The files go in `attachments`: a dict of
named files, so **one example (one row in the dataset) can hold as many as it
needs**. From [`multimodal_datasets.py`](multimodal_datasets.py):

```python
client.create_examples(
    dataset_name="hr-onboarding/expense-receipts/v1",
    examples=[{
        "inputs":  {"note": "Conference day expenses...", "submitted_on": "2026-09-25"},
        "outputs": {"decision": "needs_receipt", "receipts": [...], "missing_receipt_keywords": ["parking"]},
        "attachments": {
            "receipt_1":      ("image/png", receipt_1_bytes),
            "receipt_2":      ("image/png", receipt_2_bytes),
            "card_statement": ("text/csv",  statement_bytes),
        },
        "split": "gate",
    }],
)
```

Each value is `(mime_type, bytes)`. A file path also works, but only with
`dangerously_allow_filesystem=True`, so this repo passes bytes. Use names without
dots (`receipt_1`, not `receipt_1.png`).

Why use attachments instead of base64 in `inputs`? LangSmith stores each one as a
file, shows it as an image (or as text) when you open the example, and keeps
`inputs` small enough to read and filter. The dataset has six claims covering the
cases worth testing:

| Example | Attachments | What it tests |
|---|---|---|
| `single-receipt` | 1 PNG | the baseline read |
| `two-receipts` | 2 PNGs | one receipt per file, with no totals swapped between them |
| `statement-missing-receipt` | 2 PNGs + 1 CSV | **reading across files**: the statement lists a $48 parking charge with no receipt |
| `late-submission` | 1 PNG | a date read off the image, compared with the policy's 30 days |
| `wrong-attachment` | 1 PNG + a non-receipt image | ignoring the welcome banner instead of inventing a receipt from it |
| `phone-photo` | 1 degraded PNG | the same read on a rotated, blurred, low-contrast image (`image_quality: degraded`) |

The images are drawn in code by [`multimodal_fixtures.py`](multimodal_fixtures.py),
so nothing is downloaded and nothing real is uploaded. Because the answer key
comes from the same spec that drew the pixels, the expected total always matches
what the image shows. Run the file to write them to `fixtures/receipts/` and see
exactly what the model sees.

### Reading attachments in the target and evaluators

Name a parameter `attachments`, and `evaluate` downloads each example's files and
passes them in. This works for the target and for evaluators:

```python
def target(inputs: dict, attachments: dict) -> dict:
    # attachments["receipt_1"] == {"reader": <file-like>, "mime_type": "image/png", "presigned_url": ...}
    files = files_from_attachments(attachments)       # -> [(name, mime, bytes), ...]
    return review_expense_claim(inputs["note"], inputs["submitted_on"], files).model_dump()
```

Two details that cause bugs:

- **Rewind the reader.** `files_from_attachments` calls `seek(0)` after reading.
  The runner resets readers after the target and after the last evaluator, but
  not between evaluators, so a second evaluator reading the same file would get
  empty bytes.
- **Sort by name.** The order of the dict isn't guaranteed. Label each file with
  its name in the prompt (`hr_agent/receipts.py` does this) so the model can say
  which file a number came from, and so the eval can check that.

### Evaluating what the model saw

The usual rule still applies: **check in code first**. The model looked at an
image, but it answered with numbers and labels:

| Evaluator | Kind | Catches |
|---|---|---|
| `receipts_found` | code | a dropped file, *or* a receipt invented from a non-receipt image |
| `receipt_totals_match` | code | a misread total, matched **by attachment name**, so right numbers attributed to the wrong images score 0 |
| `missing_receipt_flagged` | code | missing the charge that only shows up when you compare the CSV with the images |
| `decision_matches` | code | the wrong final call |
| `reason_grounded_in_attachments` | LLM judge, **shown the same files** | a written reason that describes something the files don't show |

The judge is an **image eval**: it receives the attachments and checks the review
against the pixels, not against the answer key. Because it doesn't need ground
truth, the same evaluator can run on production traces.

> **Validate the judge, including here.** The first version of this judge failed
> a correct review. The review said "submitted on 2026-09-09". That date comes
> from `inputs`, not from any file, so the judge, which only saw the files, called
> it unsupported. The fix was to give the judge the claim's fields too. A judge
> only knows what you show it. Check its failures before you trust its passes.

To slice results, group the experiment by the `image_quality` or `scenario`
metadata. If accuracy holds on `clean` and drops on `degraded`, the problem is
image quality, not reasoning.

> From production: a `@traceable` function can log files as attachments
> (`langsmith.schemas.Attachment`), and `create_examples` accepts
> `source_run_id` + `use_source_run_attachments` to copy a trace's files into a
> dataset example. That is how a misread receipt from production becomes a test
> case. This repo hasn't exercised that path.
