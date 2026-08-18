"""Reset the Session 2 demo state — delete the production project, the
annotation queue, and the production-curated dataset so a live run starts from
a clean slate. The hand-written module datasets are left intact.

Run this between workshop cohorts (or before a fresh demo) so
`production_traffic.py` creates the project anew and the annotation queue starts
empty. Safe to run when nothing exists yet — missing targets are skipped.

Run:  python reset_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langsmith import Client

from config import require_langsmith
from module_5_online_evals.project import PRODUCTION_PROJECT
from module_6_improving_evals.annotation_queue import QUEUE_NAME
from module_6_improving_evals.curate_dataset import DATASET_NAME as CURATED_DATASET


def main() -> None:
    require_langsmith()
    client = Client()

    # 1) Delete the production tracing project (by name — removes traces + feedback).
    try:
        client.delete_project(project_name=PRODUCTION_PROJECT)
        print(f"Deleted project '{PRODUCTION_PROJECT}'.")
    except Exception as e:  # not found / already gone
        print(f"Project '{PRODUCTION_PROJECT}' not deleted ({type(e).__name__}: {e}).")

    # 2) Delete the annotation queue (look up its id by name first).
    try:
        queues = list(client.list_annotation_queues(name=QUEUE_NAME, limit=5))
        if not queues:
            print(f"Annotation queue '{QUEUE_NAME}' not found (already clean).")
        for q in queues:
            client.delete_annotation_queue(q.id)
            print(f"Deleted annotation queue '{QUEUE_NAME}'.")
    except Exception as e:
        print(f"Annotation queue not deleted ({type(e).__name__}: {e}).")

    # 3) Delete the dataset curated from production traces. The hand-written
    # module datasets are left alone — they're part of the workshop material.
    try:
        if client.has_dataset(dataset_name=CURATED_DATASET):
            client.delete_dataset(dataset_name=CURATED_DATASET)
            print(f"Deleted curated dataset '{CURATED_DATASET}'.")
        else:
            print(f"Curated dataset '{CURATED_DATASET}' not found (already clean).")
    except Exception as e:
        print(f"Curated dataset not deleted ({type(e).__name__}: {e}).")

    print("\nReset complete. Re-run module_5_online_evals/production_traffic.py to repopulate.")


if __name__ == "__main__":
    main()
