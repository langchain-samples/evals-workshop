# Choosing an evaluator: deterministic first, judge last

LLM-as-judge is the most flexible evaluator and the easiest to overuse. It costs
money on every run, it can be wrong *confidently*, and its scores drift when you
change the judge model or edit its prompt. Reach for it when nothing cheaper can
answer the question — and most of the time something cheaper can.

The pattern behind most misuse: a team needs to know "did the agent get X right?",
and X is something a program can check — but a judge is one prompt away, so the
judge gets written. A common version of this shows up early in agent projects: an
LLM is asked to decide whether an identifier already exists in a system of record,
when a lookup against that system answers it exactly, instantly, and for free.

## The ladder

Start at the top. Stop at the first rung that can answer your question.

| # | Ask yourself | Use | Cost / flake | In this repo |
|---|---|---|---|---|
| 1 | Is there a **system of record** I can look the answer up in? | Exact lookup against it | Free · never flakes | `mentions_required_facts` (`module_2/deterministic_evals.py`) checks the answer against known facts |
| 2 | Is it a **shape** rule — valid JSON, right fields, enum in range? | Schema / type validation | Free · never flakes | `structured_answer_is_valid`, `tool_args_well_formed` |
| 3 | Is it about the **process** — which tools, in what order, with which args? | Trajectory and argument checks | Free · never flakes | `module_3/trajectory_evals.py`, `tool_evals.py` |
| 4 | Can I state it as an **invariant** over what the tools returned? | A relative check ("this arg must equal that output field") | Free · never flakes | `args_follow_tool_outputs` (`module_3/replay_evals.py`) |
| 5 | Can a **heuristic** approximate it well enough — keywords, regex, thresholds? | A rule with a *known* error rate | Free · flakes only where the rule is crude | `not_deflected`, `reports_tool_failure` |
| 6 | Is it a **narrow, typed call about meaning** you can read straight off the text — a yes/no, one of a few labels, a rating on a short scale — that you need to make at **volume**? | **System 1 model** (e.g. Jev) — [see below](#system-1-models-jev-the-rung-between-a-rule-and-an-llm-judge) | Cents per thousand · low variance · no explanation | `system1_judge.py`, `system1_alignment.py` — [try it](#try-it-on-this-repo) |
| 7 | Is it genuinely **open-ended** — tone, groundedness of free text, "was this a reasonable approach" — or does it need a **written rationale**? | LLM judge | $ per run · noisy | `module_2/llm_judge_evals.py`, `llm_trajectory_judge.py` |
| 8 | Is it **high-stakes**, or are you unsure the judge is right? | Human review | Slow · the ground truth | Module 6: annotation queues |

Rungs 1–5 are code. Rungs 6 and 7 are the two that put a model in the loop — a
fast, typed classifier and a slow, reasoning generator — and rung 8 is what tells
you whether either can be trusted.

## Worked example: "did it schedule orientation on the right day?"

| Approach | What happens |
|---|---|
| **Judge** — "Given this transcript, was orientation scheduled on the employee's start date?" | Pays for a model call to re-derive something already sitting in two tool outputs. Can be talked into a wrong answer by a fluent final message. Different score next week if the judge model changes. |
| **Absolute check** — `date == "2026-06-15"` | Free and exact — until HR moves the start date, when it fails a healthy agent. This is the point-in-time problem; see [Module 3](../module_3_agent_evals/). |
| **Invariant** — `schedule_orientation.date == lookup_employee.start_date` | Free, exact, and valid in every world. Needs no judge and no recording. |

The judge isn't wrong here so much as unnecessary. Same shape as the failure-handling
check in Module 3: `reports_tool_failure` is a phrase check, not a judge, because
"did the reply admit the tool failed?" has a small, enumerable set of tells.

## System 1 models (Jev): the rung between a rule and an LLM judge

Kahneman's split: **System 1** is fast, intuitive judgment; **System 2** is slow,
deliberate reasoning. An LLM judge is System 2 — it reads, reasons, writes a
rationale, and emits a verdict, and you pay for every one of those steps. A
**System 1 model** skips the reasoning and the prose: it looks at some text and
answers a fixed set of typed questions, with probabilities.
[Jev](https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals),
from TypeSafe AI (released September 2026), is one of the System 1 models now
available in LangSmith Evals — LangChain calls this a third kind of evaluator,
alongside code checks and LLM-as-judge. (LangSmith's docs call them *decision
models*; you'll see both names.)

### What it is

You give it a **state** (a trace, a message, a JSON blob) and a list of
**questions**. Each question has a type, and the type defines the shape of the
answer:

| Question type | Answers with | Use for |
|---|---|---|
| **Noul** | probability (0–1) that a statement is true | yes/no checks: "does this reply leak another employee's data?" |
| **Choice** | one option from a fixed list (2–255 for Jev), with a probability per option and a confidence | classification: user intent, response type |
| **Score** | a position on 2–10 ordered levels you define | graded ratings — use sparingly |

All questions in a request run in parallel, so adding a criterion barely moves
latency. In LangSmith each question becomes its own **feedback key**, so it can be
filtered, charted, and alerted on like any other score — no output schema to
define, no reasoning to parse. It does **not** generate text and does not explain
its answers.

### Where it fits in this framework

The tell is that you can answer the question by *reading the text*, the answer
set is small and known, and you need to ask it a lot. In the HR agent:

| Question | Type | Today |
|---|---|---|
| Does the reply tell the user the action failed or couldn't be completed? | Noul | `reports_tool_failure` — a phrase list (rung 5) |
| Did the agent answer, ask for clarification, decline as out of scope, or deflect? | Choice | `not_deflected` — a phrase list |
| Does the reply disclose another employee's private data? | Noul | nothing — a rule can't see it |
| What is this production message actually asking for — policy, an onboarding action, something out of scope? | Choice | nothing — useful for slicing [Module 5](../module_5_online_evals/) traffic |

Those are the phrase-list heuristics from rung 5 hitting their ceiling. A phrase
list misses "I wasn't able to get that done"; a System 1 model reads meaning. And
because it's cheap enough to run on **every** trace rather than a sample, it can
back an alert — LangChain's own suggested uses are fast online safety keys (PII
leakage, prompt injection, toxicity) that trigger on the feedback key.

A useful cut: **can the verdict be read off the text, or does it have to be
derived?** System 1 models are for the first kind. If you have to *compute* it
(compare two dates, count tool calls), that's code (rungs 1–4). If you have to
*reason* to it, or you need to know *why*, that's an LLM judge (rung 7).

### Where it doesn't

Per TypeSafe's and HoneyHive's own guidance, not for:

- **Exact calculations, counts, date comparisons.** Don't interpolate between
  Score levels to recover a number. Parse recorded facts (status codes, IDs) in
  code and ask it only about meaning.
- **Multi-step reasoning.** Double negatives and indirect questions reduce
  reliability. Ask one direct question per key.
- **Anything needing a rationale.** LangChain says so plainly: for open-ended
  criteria where you want written reasoning with the verdict, an LLM judge is
  still the better tool. Deterministic conditions still belong in code, which is
  "faster and more reliable".
- **Verifying events that aren't in its input.** "I've transferred you" doesn't
  prove a handoff happened. Check the tool call.
- **Finding failure modes you haven't seen.** It answers the questions you wrote.
  Discovering new problems starts with humans reading traces (and, for this loop,
  [Engine](../module_6_improving_evals/README.md#engine-the-same-loop-automated)).
- **Being an untested security gate.** Text in the input that argues for its own
  classification can steer the answer.

### What the evidence says — and what it doesn't

This is a new model class, so the honest read is "promising, early, mostly
vendor-adjacent".

| Source | Finding | Caveat |
|---|---|---|
| [LangChain, Sept 2026](https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals) | On one agent-eval task, Jev matched a human reviewer on every decision; score variance 92–913× lower than three LLM judges; 0.44 s vs 2.16–2.83 s per call; **$0.34 vs $0.39 / $2.90 / $28.17** for the full set. | LangChain calls it "one test on one agent"; sample size isn't stated. The cost gap is largest against the biggest judge. |
| [Rao & Callison-Burch](https://arxiv.org/abs/2609.29769) (preprint) | Against three flash-tier LLM judges over nine panels from seven benchmarks, Jev's accuracy differed significantly in only **8 of 27** paired comparisons — ahead mostly on **binary** criteria, behind only on **graded** ones. LLM judges cost **29–325×** as much and took **30–220×** as long. | Most of the other comparisons were inconclusive. Compared against *flash-tier* judges, not the strongest models. |
| [Li et al.](https://arxiv.org/abs/2609.26550) (preprint) | Within three points of GPT-6 "wherever a verdict can be read off the text", at 0.36% of the fee and 0.15 s median latency; **behind where the verdict must be derived** (math, code, logic). | Confidence routing weakens on style-adversarial pairs and reference-free prose. |

Two things line up across all three: **binary questions are where it's strongest,
graded ones are where it's weakest**, and **"read off the text" vs "derived" is
the boundary**. Treat the headline ratios as directional — they depend on which
LLM you compare against — and the accuracy claims as things to re-measure on your
own labels, not assume.

### Cascades: accept when confident, escalate when unsure

Because it returns probabilities, you can route on them: accept a confident verdict,
send the unsure ones to something stronger. The two preprints disagree on how much
that buys, and the disagreement is instructive:

- Li et al. report the cascade (with a threshold frozen in advance) **0.9 points more
  accurate than GPT-6 at 41% of the fee**, matching it on two new workloads.
- Rao & Callison-Burch report that LLM judges **repeat nearly all of Jev's most
  confident errors**, so escalating to an LLM lowers cost but gains at most 1.5
  points over the best single judge.

Our read, and it is an inference rather than a finding from either paper: a cascade
is mainly a **cost** tool. When the System 1 model is confidently wrong, an LLM tends
to be wrong in the same place, so a bigger model isn't the safety net. For the
verdicts that matter, escalate the *uncertain* ones to a **human** (rung 8) — for
example a Noul probability in the 0.4–0.6 band, or an explicit `unclear` option on a
Choice, routed to an [annotation queue](../module_6_improving_evals/) — and tune the
band on labeled examples. Use confidence for routing, not as proof the answer is
right.

### Using one here

1. **Prefer Noul.** "For failure detection, a binary criterion is easier to define
   and check against human labels than a vague five-level rating." If you need a
   grade, ask several binary questions and combine them in code.
2. **One direct criterion per question**, phrased so high probability means yes.
   Put grading criteria in the *questions*, not in the state.
3. **Send only what the question needs.** Limits are 64k tokens per request and
   32k for state plus the longest question; irrelevant context also hurts accuracy
   inside those limits.
4. **Measure it against human labels before you trust it** — same "evaluate the
   evaluator" step as any judge ([Module 6](../module_6_improving_evals/)), with
   partial-answer and adversarial cases included, and a held-out slice you never
   tuned on. `system1_alignment.py` does this for a decision model; Module 6's `judge_alignment.py`
   does it for the few-shot LLM judge. The procedure is the same: agreement with humans *is* its accuracy.
5. **Pin the version.** `jev-latest` moves when TypeSafe ships a release. Select the
   versioned ID you validated (HoneyHive's example: `jev-1.13.0`) and re-check
   against your labels before upgrading.
6. **Don't carry a threshold across question types.** A Noul and a yes/no Choice can
   return different probabilities for the same question, and a question and its
   negation needn't sum to 1.

### What to know about running it in LangSmith

- **Creating an evaluator is UI-only; calling the model is not.** The docs say
  decision-model *evaluators* can be created "only in the LangSmith UI" — SDK support
  isn't there yet. But the model itself is a plain HTTP endpoint (`POST /v1/systemone`),
  so you can call it from code, which is what this repo's example does. The UI-authored
  kind is [config, not code](../README.md#in-code-is-about-authorship-not-about-where-it-runs):
  edit a question and every historical score changes meaning. Keep the state mapping
  and question text in the repo as documentation, and pin the model.
- **Access and price.** TypeSafe describes Jev as early access (a waitlist at launch),
  priced at $0.042 per million *input* tokens, output free — check the current terms.
  LangChain measured $0.00035 per call. That's far below the strongest LLM judges it
  compared, but the gap to the cheapest one was small ($0.34 vs $0.39 for the full
  set) — so the speed and low variance may matter more than the price — and it's a
  separate bill from LangSmith when you bring your own key.
- **Setup.** Store a TypeSafe API key as a workspace secret (`TYPESAFE_API_KEY`),
  then Tracing project → **Evaluators** → **+ Evaluator** → LLM-as-a-Judge, and
  pick the decision model. The docs list support for both tracing projects and
  datasets, but only walk through the online path — check the offline path yourself.
- **Data retention — decide before you connect.** TypeSafe does *not* offer zero
  data retention: prompts and outputs sent to Jev may be retained by the provider.
  This repo's traces are HR-agent traces; a real one carries employee PII. Same gate
  as [curation](../module_6_improving_evals/) and Engine. Separately, running an
  online evaluator on any run upgrades that trace to extended retention, which
  changes trace pricing.
- **An alternative provider.** The docs also list **SemIf**, which runs through the
  LangSmith LLM Gateway with no provider key (enabled for US organizations on Free,
  Developer and Plus plans at the time of writing) — so it can avoid the extra
  third-party hop. The free period was announced as running through
  September 28, 2026; verify what it costs now.

### Try it on this repo

The question is one this repo already answers badly: *did the reply admit the tool
failed?* Module 3's `reports_tool_failure` is a phrase list. It passes "no problem at
all!" (it contains "problem") and fails "that didn't go through". The files next to
this guide score it, and a System 1 model, against 40 labeled replies:

```bash
python module_2_single_turn/system1_alignment.py                     # phrase-list baseline — no keys
python module_2_single_turn/system1_alignment.py --classifier both   # + System 1 (needs a key; sends text out)
python module_3_agent_evals/mocked_eval.py --system1                 # the same, as an evaluator in an experiment
```

The baseline, run for real on the held-out `gate` split:

```
phrase  (tool_evals.reports_tool_failure signal words)  (27 examples: 27 decided, 0 in review band)
  accuracy  0.48   precision  0.45   recall  0.38   f1  0.42   coverage  1.00
  errors by category: B: 6 (FN), E: 6 (FP), F: 2 (FN)
```

The System 1 side asks one Noul question. Note where the split falls: *code* decides
which tool results failed (a recorded fact), and the model only judges whether the
reply *says so* (meaning). The state carries no grading instructions.

```python
noul("Does the assistant_reply tell the user that at least one of the "
     "failed_tool_results did not complete, or is delayed?",
     true="The reply says or clearly implies that an action failed, was blocked, "
          "or is delayed or unavailable.",
     false="The reply says or implies everything succeeded, or never mentions the problem.")

state = {"failed_tool_results": [...], "assistant_reply": "..."}
```

An answer inside the review band (default 0.4–0.6) isn't scored as pass or fail; it's
flagged `system1_needs_review`, ready for an annotation queue. Tune the band on the
`scratch` split and report on `gate`.

| File | Role |
|---|---|
| [`system1_alignment.py`](system1_alignment.py) | scores each classifier against the labels; `--self-test` runs offline |
| [`system1_judge.py`](system1_judge.py) | the question above, as an evaluator |
| [`system1_client.py`](system1_client.py) | a small `httpx` client: https-only host allowlist, keys from the environment only, strict response validation, no vendor SDK |
| [`system1_labels.py`](system1_labels.py) | 40 synthetic labeled replies, `gate`/`scratch` split |

Two honest limits. **We haven't run it against the live API** — the client is tested
against a mock server, so request and response handling is verified but Jev's accuracy
on your data is not, and there is no System 1 number here to quote. And the label set is
author-written and built to break phrase lists (the B and E errors above are by
construction), so it shows phrase lists *have* failure modes, not how yours does. Bring
50 of your own labeled replies; that is the exercise. The System 1 side **sends text to
a third party**, so it is off unless you ask for it and prints the host first.

Everything else in this section comes from LangChain's docs and blog, TypeSafe's docs,
HoneyHive's guide, and the two preprints' abstracts (we did not read the papers' full
text).

## Where an LLM judge earns its cost

- **Tone and register** — "warm, professional, concise". No rule captures it.
- **Groundedness of free text** — does this paragraph assert anything the source
  doesn't support? A rule can catch a missing fact; only a model can catch an
  invented one.
- **Holistic process quality** where many trajectories are acceptable and you
  can't enumerate them (`trajectory_is_reasonable`).
- **Open-ended helpfulness** on questions with no single right answer.

## Making a judge worth trusting

If a question really does need a judge, the rest of the workshop is how to keep it
honest:

1. **One criterion per judge.** "Is it correct, grounded, polite, and concise?" is
   four judges wearing a trench coat. Split them (Module 2 does).
2. **Structured output with reasoning first**, so a surprising score is
   explainable rather than a bare number.
3. **Gate on the cheap checks, judge what survives.** If the deterministic checks
   already failed, the judge's opinion of the same answer adds cost, not
   information.
4. **Measure agreement with humans before you rely on it** — the "evaluate the
   evaluator" step in [Module 6](../module_6_improving_evals/). A judge you haven't
   measured is a guess with a decimal point.
5. **Never grade the judge on its own few-shot examples.**
6. **Version the prompt.** Edit a judge and every historical score quietly
   changes meaning.

## A test for "should this be a judge?"

Before writing a judge, try to write the deterministic check. If you can't say
*why* it's impossible — "there's no source of truth to look up", "the criterion is
about style" — you probably haven't found the rung yet.

And a reverse test for a judge you already have: replace it with a lookup or an
invariant on a sample of 50 traces. If they agree, delete the judge.

## Trajectories are not always enough

Trajectory evals grade the *path*. They're strong for safety ("never called
`provision_equipment` on a read-only question") and weak when:

- **many paths are valid** — exact match fails healthy runs, while a
  required-tools check passes runs that did the work badly (Module 3 covers each);
- **the outcome is what matters, not the route** — check the *result* (what state
  the world ended in, what the final answer commits to) rather than the steps;
- **the answer depends on data that changes** — an absolute check goes stale; use
  an invariant, or replay the recorded tool outputs.

The strongest suites layer these: a deterministic outcome/invariant check as the
gate, a trajectory check as the diagnosis when it fails, and a judge only for
what's left.
