# Module 4 — Multi-Turn Evaluations

> Goal: evaluate conversations, not single messages. Grade one turn with the
> history held fixed, then grade whole threads driven by a simulated user. Do
> both for text and for conversations where files arrive across turns.

Modules 2 and 3 send the agent one message. Real users don't stop there. They say
"them" and "the same", correct themselves, answer a question the agent asked, and
send the missing receipt two messages later. Each of those is a way to fail that a
single-message eval can't reach.

There are two shapes of multi-turn eval, and they answer different questions:

| | **Turn-level** | **Thread-level (simulated user)** |
|---|---|---|
| Example holds | the conversation so far + the next user message | a persona and a goal |
| You run | one agent turn | a whole conversation, until the user is done |
| You grade | that one reply and its tool calls | the whole thread: did it get there, how many turns, how smoothly |
| Repeatable? | yes: same history every run | no: the simulated user varies, so run it several times |
| Use it for | specific skills: references, corrections, context bleed | end-to-end task success and friction |
| Files | `datasets.py`, `turn_evals.py`, `run_eval.py` | `simulated_eval.py` |

Start with turn-level. It is cheap, deterministic, and points at the exact turn
that broke. Add simulation when you need to know whether the whole conversation
works.

## What's here

| File | What it teaches |
|------|-----------------|
| `datasets.py` | 6 conversations, each ending in the turn under test: a pronoun, "do the same for Sam", a correction, an answer to the agent's own question, a policy follow-up, and one the policy can't answer. |
| `turn_evals.py` | Scores **only this turn's** tool calls (expected, forbidden), the facts in the reply, and an LLM judge that reads the whole conversation. |
| `run_eval.py` | The turn-level experiment. |
| `simulated_eval.py` | A simulated user (persona + goal) talks to the agent for up to 6 turns. Thread-level evaluators: goal completed, no forbidden calls, facts stated, user turns taken, and a conversation-quality judge. |
| `multimodal_datasets.py`, `multimodal_run_eval.py` | **Multimodal multi-turn.** Expense claims where receipts arrive in different turns: the missing receipt sent later, a charge withdrawn in words, the wrong file sent, a receipt replaced. Graded by Module 2's evaluators, unchanged. |

## Run it

```bash
# Evaluator self-tests — no API key needed.
python module_4_multi_turn/turn_evals.py

python module_4_multi_turn/run_eval.py                       # turn-level, text
python module_4_multi_turn/simulated_eval.py --repetitions 3 # thread-level, simulated user
python module_4_multi_turn/multimodal_run_eval.py            # turn-level, files across turns
```

---

## Turn-level: freeze the history, grade the next turn

Each example's input is a list of messages ending with the user's new one:

```python
{"messages": [
    {"role": "user",      "content": "Set up an email account for Jordan Lee."},
    {"role": "assistant", "content": "Done! Jordan Lee (E1007) now has an email account..."},
    {"role": "user",      "content": "Great. Now do the same for Sam Rivera."},
]}
```

The reference says what *this* turn must do, and what it must not:

```python
{"expected_calls":  [{"name": "create_it_account", "args": {"employee_id": "E1008", "system": "email"}}],
 "forbidden_calls": [{"name": "create_it_account", "args": {"employee_id": "E1007"}}]}
```

That forbidden call is the **context-bleed** check. Jordan's ID is the one sitting
in the conversation, and reusing it is the typical failure. The `correction`
example works the same way: the user swaps the monitor for a headset, so ordering
a monitor is forbidden.

**Grade only this turn.** The agent's result holds the input history *plus* what
it added. `run_eval.py` keeps only the new part:

```python
result = build_agent().invoke({"messages": [(m["role"], m["content"]) for m in history]})
new = {"messages": result["messages"][len(history):]}
```

Skip that slice and turn 3 gets credit for a tool call made in turn 1. This is the
problem Module 3's [scope table](../module_3_agent_evals/README.md#scope-a-trajectory-here-is-one-agent-run-not-a-conversation)
warns about. Making each example *be a turn* is the fix that table points to.

**Where the history comes from.** Here it's hand-written, so the assistant turns
are plain text. Real threads also hold tool calls and tool results. To test
against a real conversation, snapshot one turn of a production thread with
`module_3_agent_evals/snapshot_trace.py --run-id <uuid> --last-turn`.

**Pass history as messages, not as one pasted string.** Each turn stays its own
message with its own role. Pasting the transcript into one prompt changes what
the model sees, and it lets anything the user wrote in an earlier turn read like
instructions.

> **Check the judge's passes, too.** On our run, `unanswerable-follow-up`
> correctly said the policy doesn't cover carry-over, then told the user to email
> `hr@company.com`. That address appears nowhere in the policy. `context_handling`
> scored the reply 3/3 anyway. The judge was told not to invent policy facts, and
> it didn't treat a contact address as one. That kind of miss is what aligning a
> judge to human labels catches ([Module 7](../module_7_improving_evals/)).

## Thread-level: let a simulated user drive

Some questions only exist across a whole conversation: does the agent finish the
job, how many turns does it take, does it make the user repeat themselves? To
answer them, `simulated_eval.py` runs the conversation. A second model plays the
user from a persona:

```python
"You are an onboarding coordinator. You need Sam Rivera set up with an email account,
 a Slack account, and a laptop. Start vague ('I need to get a new hire set up') and
 only give the name and the list when asked."
```

Each turn, the simulator returns `{message, done}`. The agent answers with a
checkpointer and a shared `thread_id`, the same way `hr_agent.run_conversation`
does. The loop stops when the user says it's done or after 6 user turns.

Thread-level evaluators look at everything that happened:

| Evaluator | Kind | Question |
|---|---|---|
| `goal_completed` | code | did every expected action happen, in any turn? |
| `no_forbidden_calls_in_thread` | code | did a wrong action happen anywhere? |
| `facts_stated` | code | did the assistant say the facts at some point? |
| `user_turns` | code, a count | how many turns did it take? Lower is better; it's not pass/fail |
| `conversation_quality` | LLM judge, reference-free | did it keep track, avoid asking twice, recover from mistakes? |

"Did it happen in any turn" is the **right** question here. Module 3 warns that
a set-membership check over a thread is weak, but that warning is about
*preconditions* ("look it up before every claim"). For *task completion*, "did it
happen at all" is exactly what you want to know.

**Simulated conversations vary.** The same scenario produces a different
conversation each run. On ours, `benefits-follow-ups` took 4 user turns in one
repetition and 3 in the other. Run with `--repetitions` and compare averages,
not single runs. Two more caveats:

- The simulator is a model too. If the agent fails, read the transcript before
  blaming it: sometimes the simulated user went off-script.
- A cooperative simulator makes things easy. Ours gave the full list as soon as
  it was asked. To stress the agent, write personas that are vague, change
  their minds, or make mistakes (like `misspelled-name`).

[`openevals`](https://github.com/langchain-ai/openevals) ships a ready-made
simulated user and multi-turn runner. We build ours by hand so the loop is visible.

## Multimodal multi-turn: files arrive across turns

An expense claim rarely fits in one message. The employee sends receipts, the
reviewer says one is missing, and the employee sends it, or withdraws the charge,
or sends the wrong file.

**Every file from every turn goes in the one example's `attachments`, and each
turn lists the names of the files it carried:**

```python
inputs = {"submitted_on": "2026-09-25", "turns": [
    {"role": "user", "text": "Conference day expenses...", "attachments": ["receipt_1", "receipt_2", "card_statement"]},
    {"role": "assistant", "text": "The $48.00 GATE 12 PARKING charge needs a receipt."},
    {"role": "user", "text": "Here's the parking receipt.", "attachments": ["receipt_3"]},
]}
attachments = {"receipt_1": ..., "receipt_2": ..., "card_statement": ..., "receipt_3": ...}
```

The target (`build_turns` in `multimodal_run_eval.py`) puts each file back into
the message it came with. A name that isn't in the example's attachments raises
an error rather than being skipped, so a typo in the dataset can't make a test
quietly easier.

| Example | Turn 3 | Correct state after turn 3 |
|---|---|---|
| `receipt-arrives-later` | sends the parking receipt | `approve`, 3 receipts |
| `charge-withdrawn` | "the parking was personal, drop it". No file: words override the CSV from turn 1 | `approve`, 2 receipts |
| `wrong-file-sent` | sends the welcome banner instead of the receipt | still `needs_receipt` |
| `receipt-replaced` | sends a corrected hotel receipt | `approve`, with the *replacement's* total ($167.00, not $189.00) |

The answer key is the claim's state after the last turn, in the same shape as
Module 2's. So Module 2's evaluators (`receipts_found`, `receipt_totals_match`,
`missing_receipt_flagged`, `decision_matches`) grade it unchanged. Only the input
got longer. `receipts_found` now has a second job: on `receipt-replaced`,
counting the superseded receipt as well gives 3 instead of 2.

## Key ideas to land

1. **Make each example be a turn.** History as input, one reply graded. The run
   boundary and the assertion boundary are then the same thing.
2. **Slice to this turn.** Whatever the agent returns includes the history; score
   only what it added.
3. **Test the multi-turn skills directly:** reference resolution, corrections,
   answers to the agent's own question, and context bleed. Each needs its own
   example, with a forbidden call where there's a tempting wrong action.
4. **Simulate when you need the whole thread**, and treat one simulated run as an
   anecdote. Compare across repetitions.
5. **Files are part of the conversation.** Keep them in one example's attachments,
   referenced by name from the turn that sent them. Later words can override an
   earlier file.

Next: **Module 5**, where these suites become a CI gate.
