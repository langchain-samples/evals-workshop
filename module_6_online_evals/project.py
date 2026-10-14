"""The production project name — one source of truth for Session 2.

`production_traffic.py` sends traces *to* this project; `score_traces.py`,
`annotation_queue.py`, and `monitor.py` read traces *from* it. We keep it as its
own constant (not the ambient `LANGSMITH_PROJECT`, which your `.env` may point at
the offline experiment project) so "production traffic" always lands in, and is
read from, the same dedicated place. Every consumer also takes a `--project`
override for pointing at your real production project.
"""

from __future__ import annotations

# The dedicated project that stands in for "production" in this workshop.
PRODUCTION_PROJECT = "hr-agent-production"
