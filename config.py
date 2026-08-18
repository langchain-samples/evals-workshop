"""Shared configuration for the evals workshop.

One place to control which model powers the agent and the LLM judges, so the
whole workshop can swap providers by changing a single env var.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load .env once, on import, for every script in the workshop.
# override=True makes the workshop's .env authoritative over ambient shell/IDE
# environment variables. This matters because some shells/IDEs export keys like
# ANTHROPIC_API_KEY as an EMPTY string; plain load_dotenv() treats an empty
# ambient value as "already set" and silently skips the real key in .env,
# which then surfaces as a confusing 403 from the model provider/gateway.
load_dotenv(override=True)

# init_chat_model string form: "<provider>:<model>".
DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"

# The agent under test. Override with WORKSHOP_MODEL.
AGENT_MODEL = os.getenv("WORKSHOP_MODEL", DEFAULT_MODEL)

# The model that grades outputs in LLM-as-judge evaluators. Judges can be a
# different (often cheaper) model than the agent. Override with
# WORKSHOP_JUDGE_MODEL, else falls back to the agent model.
JUDGE_MODEL = os.getenv("WORKSHOP_JUDGE_MODEL", AGENT_MODEL)


@lru_cache(maxsize=None)
def get_judge(model: str = JUDGE_MODEL):
    """Return a chat model for use in LLM-as-judge evaluators.

    Temperature 0 for the most deterministic grading we can get. Cached so
    repeated evaluator calls reuse one client.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, temperature=0)


def require_langsmith() -> None:
    """Fail fast with a friendly message if LangSmith isn't configured."""
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit(
            "LANGSMITH_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or `export LANGSMITH_API_KEY=...`. Get one at "
            "https://smith.langchain.com → Settings → API Keys."
        )


# --- Experiment metadata -------------------------------------------------
# Best practice: every experiment carries the code + config that produced it,
# so a regression in LangSmith can be traced back to a commit, a branch, and a
# model. `experiment_prefix` alone only gives you a name.
#
# NOTE: the LANGSMITH_EXPERIMENT env var is read ONLY by the langsmith pytest
# plugin (langsmith/testing/_internal.py). It does NOT name or tag experiments
# created by `client.evaluate(...)` — those need this metadata passed explicitly.

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _git(*args: str) -> str | None:
    """Run a read-only git command; return its output, or None if unavailable.

    Fixed argv (never shell=True) and a short timeout, so this can't hang a
    workshop script or interpolate anything into a shell.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else None


@lru_cache(maxsize=1)
def git_context() -> dict[str, str]:
    """Commit + branch for the current checkout.

    In CI we prefer the values the runner already knows (GitHub sets these on
    every job); locally we ask git. Either way the experiment gets tagged.
    """
    commit = os.getenv("GITHUB_SHA") or _git("rev-parse", "HEAD")
    branch = os.getenv("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD")
    ctx: dict[str, str] = {}
    if commit:
        ctx["commit"] = commit[:12]
    if branch:
        ctx["branch"] = branch
    if _git("status", "--porcelain"):
        ctx["dirty"] = "true"  # uncommitted changes — results aren't reproducible
    return ctx


def experiment_metadata(**extra) -> dict:
    """Metadata to attach to a LangSmith experiment.

    Pass to ``client.evaluate(..., metadata=experiment_metadata(suite="agent"))``.
    Everything here is searchable/filterable in the LangSmith UI, which is what
    turns a pile of experiments into something you can actually compare.
    """
    meta = {
        "model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "ci": bool(os.getenv("CI")),
        **git_context(),
    }
    if author := os.getenv("GITHUB_ACTOR"):
        meta["author"] = author
    meta.update(extra)
    return meta
