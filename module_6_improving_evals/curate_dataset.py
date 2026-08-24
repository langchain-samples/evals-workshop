"""Module 6 — close the flywheel: turn production traces into a dataset.

Module 5 scored live traces. This step is what makes that worth doing: the
interesting traces — the ones your online evals flagged, or a human labeled in
the annotation queue — get promoted into a **curated offline dataset**. From
then on they're a permanent regression test. A failure that happened once in
production can never quietly happen again.

```
production traces ──► online evals flag ──► human labels ──► THIS SCRIPT ──► dataset
                                                                                │
                                    Modules 1-4 run against it forever ◄────────┘
```

Every created example carries `source_run_id`, so LangSmith links each dataset
row back to the exact production trace it came from — you can always answer
"where did this test case come from?"

⚠️  **Before you point this at real production traffic, read this.**

Curation moves production content into an artifact with a *longer life and a
wider audience* than a raw trace: datasets get shared across a team, copied into
other workspaces, and kept indefinitely as regression suites. Our system under
test is an HR agent, so real traces can carry employee PII — names, employee
IDs, salaries, medical and benefits details.

**This script does not redact anything.** It copies the question and the answer
verbatim (only truncating them) and links each example back to its source run.
Treat that as deliberate: automated redaction that half-works is worse than none,
because it invites trust. So, for real traffic:

  - Review and redact examples before, or immediately after, they land — the
    `needs_review: True` metadata flag exists to mark that as outstanding.
  - Prefer ``--labeled``, i.e. traces a human already opened in an annotation
    queue. Someone has looked at that content.
  - Apply the same access controls and retention rules to the dataset that you
    apply to the production project it came from.
  - Use ``--dry-run`` first to see exactly what would be copied.

Selection modes:
  --flagged   (default) traces with a low online-eval score, i.e. what
              score_traces.py wrote feedback on. The failures.
  --labeled   traces a human scored in the annotation queue. The gold ones.
  --all       every recent root trace.

Run:
    python module_6_improving_evals/curate_dataset.py --flagged
    python module_6_improving_evals/curate_dataset.py --labeled --limit 50
    python module_6_improving_evals/curate_dataset.py --all --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

from config import require_langsmith
from module_5_online_evals.project import PRODUCTION_PROJECT
from module_5_online_evals.score_traces import _answer_from_run, _question_from_run

# {domain}/{capability}/{version_or_variant} — see the top-level README.
DATASET_NAME = "hr-onboarding/policy-qa/from-production"

# Trace content is remote data written by whoever used the app. We copy only the
# two plain strings we need (question, answer) into examples, never the raw run
# object, and never deserialize it into live Python objects. Long strings are
# truncated so one runaway trace can't bloat the dataset.
MAX_FIELD_CHARS = 4_000


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + " …[truncated]"


def _scores_by_run(client: Client, runs: list) -> dict[str, dict[str, float]]:
    """{run_id: {feedback_key: score}} for the given runs — one API call."""
    scores: dict[str, dict[str, float]] = {}
    if not runs:
        return scores
    for fb in client.list_feedback(run_ids=[r.id for r in runs]):
        if fb.score is None:
            continue
        scores.setdefault(str(fb.run_id), {})[fb.key] = float(fb.score)
    return scores


# Feedback written by our own scripts / LLM judges carries these source types.
# Anything else — and in particular anything with a user attached — came from a
# person in the UI or an annotation queue. (There is no server-side filter for
# "human"; FeedbackSourceType only knows 'api' and 'model'.)
_MACHINE_SOURCE_TYPES = {"api", "model", "auto", "evaluator"}


def _is_human(client: Client, run_ids: list) -> set[str]:
    """Run ids carrying human (annotation-queue / UI) feedback."""
    human: set[str] = set()
    if not run_ids:
        return human
    for fb in client.list_feedback(run_ids=run_ids):
        source = getattr(fb, "feedback_source", None)
        if source is None:
            continue
        # A named user is the strongest signal; an unknown source type is the
        # fallback (LangSmith labels UI feedback 'app').
        if getattr(source, "user_id", None) or getattr(source, "user_name", None):
            human.add(str(fb.run_id))
        elif str(getattr(source, "type", "")).lower() not in _MACHINE_SOURCE_TYPES:
            human.add(str(fb.run_id))
    return human


def select_runs(client: Client, runs: list, mode: str) -> tuple[list, dict]:
    """Pick the runs worth promoting, and return them with their feedback scores."""
    scores = _scores_by_run(client, runs)
    if mode == "all":
        return runs, scores
    if mode == "labeled":
        labeled = _is_human(client, [r.id for r in runs])
        return [r for r in runs if str(r.id) in labeled], scores
    # flagged: any online-eval score below 1
    return [
        r for r in runs
        if any(v < 1 for v in scores.get(str(r.id), {}).values())
    ], scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote production traces into a curated LangSmith dataset.",
        epilog=(
            "WARNING: copies production question/answer text VERBATIM — nothing is "
            "redacted. Real HR traffic can contain employee PII, and a dataset is "
            "shared more widely and kept longer than a raw trace. Use --dry-run "
            "first; prefer --labeled (human-reviewed) over --flagged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", default=PRODUCTION_PROJECT)
    parser.add_argument("--dataset", default=DATASET_NAME, help="Target dataset name.")
    parser.add_argument("--limit", type=int, default=50, help="Recent traces to consider.")
    parser.add_argument("--split", default="gate",
                        help="Split to file the new examples under. Defaults to the gated "
                             "split: a trace your online evals flagged is a real failure, "
                             "so it should regression-test forever.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--flagged", action="store_const", const="flagged", dest="mode",
                       help="(default) traces an online eval scored below 1.")
    group.add_argument("--labeled", action="store_const", const="labeled", dest="mode",
                       help="Traces a human labeled in the annotation queue.")
    group.add_argument("--all", action="store_const", const="all", dest="mode",
                       help="Every recent root trace.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be added; write nothing.")
    parser.set_defaults(mode="flagged")
    args = parser.parse_args()

    require_langsmith()
    client = Client()

    runs = list(client.list_runs(project_name=args.project, is_root=True, limit=args.limit))
    if not runs:
        raise SystemExit(
            f"No traces in project '{args.project}'. Generate some with "
            "module_5_online_evals/production_traffic.py."
        )

    selected, scores = select_runs(client, runs, args.mode)
    if not selected:
        raise SystemExit(
            f"No traces matched mode '{args.mode}' out of {len(runs)} considered. "
            "Run score_traces.py (for --flagged) or label some traces (for --labeled), "
            "or use --all."
        )

    # Don't re-add a trace that's already in the dataset. source_run_id is what
    # makes this idempotent — re-run it daily without piling up duplicates.
    already: set[str] = set()
    if client.has_dataset(dataset_name=args.dataset):
        for ex in client.list_examples(dataset_name=args.dataset):
            if ex.source_run_id:
                already.add(str(ex.source_run_id))
    fresh = [r for r in selected if str(r.id) not in already]

    print(f"Project '{args.project}': {len(runs)} traces considered, "
          f"{len(selected)} matched --{args.mode}, {len(fresh)} new "
          f"({len(selected) - len(fresh)} already curated).")

    if not fresh:
        print("Nothing new to add.")
        return

    inputs, outputs, metadata, source_run_ids = [], [], [], []
    for run in fresh:
        run_scores = scores.get(str(run.id), {})
        inputs.append({"question": _clip(_question_from_run(run))})
        outputs.append({
            # The production answer becomes the *starting* reference, not gospel.
            # A reviewer edits it in the UI; that's the point of curation.
            "reference_answer": _clip(_answer_from_run(run)),
        })
        metadata.append({
            "source": "production",
            "source_project": args.project,
            "source_run_id": str(run.id),
            "selection_mode": args.mode,
            "online_scores": run_scores,
            # Marks "no person has looked at this example yet" — covering both
            # the unverified reference_answer AND any PII in the copied text.
            "needs_review": True,
        })
        source_run_ids.append(run.id)

    if args.dry_run:
        print("\n--dry-run; nothing written. Would add:")
        for i, m in zip(inputs, metadata):
            print(f"  - {i['question'][:70]!r}  scores={m['online_scores']}")
        return

    if not client.has_dataset(dataset_name=args.dataset):
        dataset = client.create_dataset(
            dataset_name=args.dataset,
            description="Curated from production traces. Each example links back "
                        "to its source run.",
            metadata={
                "owner": "evals-workshop",
                "capability": "policy-qa",
                "source": "production-sampling",
                "source_project": args.project,
            },
        )
        dataset_id = dataset.id
    else:
        dataset_id = client.read_dataset(dataset_name=args.dataset).id

    client.create_examples(
        dataset_id=dataset_id,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        splits=[args.split] * len(inputs),
        # The link back to the originating trace.
        source_run_ids=source_run_ids,
    )

    print(f"\nAdded {len(fresh)} example(s) to '{args.dataset}' (split: {args.split}).")
    print("REMINDER: these are verbatim production strings — nothing was redacted. "
          "Review them for PII before sharing this dataset.")
    print("Every example carries source_run_id — click through in LangSmith to see "
          "the production trace it came from.")
    print("\nNext: review the reference_answer on each (they're raw production output, "
          "flagged needs_review), then run your offline suite against this dataset.")


if __name__ == "__main__":
    main()
