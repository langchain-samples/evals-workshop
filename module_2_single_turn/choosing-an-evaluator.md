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
| 6 | Is it genuinely **subjective or semantic** — tone, groundedness of free text, "was this a reasonable approach"? | LLM judge | $ per run · noisy | `module_2/llm_judge_evals.py`, `llm_trajectory_judge.py` |
| 7 | Is it **high-stakes**, or are you unsure the judge is right? | Human review | Slow · the ground truth | Module 6: annotation queues |

Rungs 1–5 are code. Rung 6 is the only one that needs a model in the loop, and
rung 7 is what tells you whether rung 6 can be trusted.

## Worked example: "did it schedule orientation on the right day?"

| Approach | What happens |
|---|---|
| **Judge** — "Given this transcript, was orientation scheduled on the employee's start date?" | Pays for a model call to re-derive something already sitting in two tool outputs. Can be talked into a wrong answer by a fluent final message. Different score next week if the judge model changes. |
| **Absolute check** — `date == "2026-06-15"` | Free and exact — until HR moves the start date, when it fails a healthy agent. This is the point-in-time problem; see [Module 3](../module_3_agent_evals/). |
| **Invariant** — `schedule_orientation.date == lookup_employee.start_date` | Free, exact, and valid in every world. Needs no judge and no recording. |

The judge isn't wrong here so much as unnecessary. Same shape as the failure-handling
check in Module 3: `reports_tool_failure` is a phrase check, not a judge, because
"did the reply admit the tool failed?" has a small, enumerable set of tells.

## Where a judge earns its cost

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
