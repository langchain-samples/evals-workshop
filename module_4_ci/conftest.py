"""Pytest fixtures/config for the CI module.

Done at import, before tests are collected (decorators on the test functions
read this state, so it has to be set first):

  1. Put the repo root on sys.path so `import hr_agent`, `import config`, and the
     `module_*` packages resolve no matter where pytest is invoked from.
  2. Enable LangSmith tracing by default. The langsmith pytest plugin's
     `t.log_inputs/log_outputs/log_feedback` helpers raise unless
     LANGSMITH_TRACING is 'true', so we set it here (without clobbering an
     explicit value the user already exported).
  3. Turn on VCR-based HTTP caching for **local** runs only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Repo root importable.
sys.path.insert(0, str(REPO_ROOT))

# 2. Tracing on by default for the langsmith pytest integration.
os.environ.setdefault("LANGSMITH_TRACING", "true")

# 3. Response caching (vcrpy "cassettes"): record model API calls on the first
# run, replay them after. Turns a 60-second, few-cents test loop into a
# sub-second, free one while you're iterating on evaluators.
#
# LOCAL ONLY, on purpose. A CI gate that replays yesterday's recorded model
# responses cannot detect that today's model regressed — it would pass forever
# and tell you nothing. Caching is a development-speed tool; the gate must make
# real calls. GitHub Actions (and most CI) sets CI=true.
#
# No credentials reach disk: langsmith's VCR config clears *all* request headers
# before recording (langsmith/utils.py `filter_request_headers`). We still
# gitignore the directory — responses carry workspace-identifying headers, and
# cassette filenames are keyed on the LangSmith dataset UUID, so they'd never
# replay in someone else's workspace. Delete the directory to re-record.
CASSETTE_DIR = REPO_ROOT / "fixtures" / "cassettes"

if not os.getenv("CI") and not os.getenv("LANGSMITH_TEST_CACHE"):
    if os.getenv("WORKSHOP_NO_CACHE"):
        pass  # escape hatch: force real calls locally
    else:
        CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["LANGSMITH_TEST_CACHE"] = str(CASSETTE_DIR)

# By default we cache every host EXCEPT the LangSmith API — the langsmith plugin
# already excludes that one (`ignore_hosts=[client.api_url]`), so traces and
# feedback are always written for real while model calls get recorded.
#
# We deliberately do NOT hardcode `cached_hosts=["api.anthropic.com", ...]`.
# Model traffic doesn't always leave for the provider's own domain — a gateway,
# proxy, or corporate egress in front of the provider changes the host your
# process actually connects to. A filter naming the provider then matches
# nothing, and caching looks enabled while recording zero requests. Leaving the
# filter off is simpler and correct under any routing.
#
# To be surgical anyway, set WORKSHOP_CACHED_HOSTS to a comma-separated list
# (e.g. "host-a.example.com,host-b.example.com"). Don't guess the hosts — read
# them off a recorded cassette (the `uri:` field in fixtures/cassettes/*.yaml)
# so you filter on where your calls really go. Then verify: an empty
# fixtures/cassettes/ after a run means nothing matched.
_HOSTS = [h.strip() for h in os.getenv("WORKSHOP_CACHED_HOSTS", "").split(",") if h.strip()]

# `cached_hosts` raises if caching isn't enabled, so only pass it when it is.
# Import this in test modules: `@pytest.mark.langsmith(**CACHE_MARK)`.
CACHE_MARK: dict = (
    {"cached_hosts": _HOSTS} if (_HOSTS and os.getenv("LANGSMITH_TEST_CACHE")) else {}
)

# Load .env (LANGSMITH_API_KEY etc.) the same way the rest of the workshop does.
import config  # noqa: E402,F401  (import side effect: load_dotenv)
