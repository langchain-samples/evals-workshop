"""A minimal client for System 1 ("decision") models like Jev.

A System 1 model doesn't write anything. You send it a **state** (some text or
JSON) and named, typed **questions**; it answers each with a probability:

    noul    yes/no      -> a probability (0-1) that the statement is true
    choice  pick one    -> the top option, plus a probability per option
    score   rate 0..N   -> a probability-weighted position on ordered levels

This wraps the one HTTP endpoint (`POST /v1/systemone`) with `httpx` — no vendor
SDK, so there is nothing new to vet or pin. It is deliberately small: enough to
run the alignment example in `system1_alignment.py` and to back an evaluator
(`system1_judge.py`). The contract is TypeSafe's, from
https://docs.typesafe.ai/ and LangSmith's LLM Gateway docs; check them for
anything newer.

Where it can point:

  typesafe  https://api.typesafe.ai            key: TYPESAFE_API_KEY
            model: a versioned id, e.g. "jev-1.13.0"
  gateway   https://gateway.smith.langchain.com  key: LANGSMITH_API_KEY
            model: "typesafe/jev-1.13.0" (needs the TypeSafe key stored as a
            LangSmith provider secret) or a hosted model such as "semif-…"

**Pin the model id.** ``jev-latest`` is an alias that moves when a release ships;
a moving judge silently changes what every historical score means.

Security posture — this client sends a bearer token and untrusted text:

- **Keys come from the environment only** and never appear in logs, exceptions or
  ``repr``. (An API key in an error message ends up in CI logs.)
- **The host is allowlisted.** A bearer token goes wherever the base URL points, so
  an attacker who can set one env var could otherwise walk your key to their
  server. Only https, and only the hosts below.
- **Nothing is sent unless you call it.** No import-time requests, no telemetry.
- **The response is untrusted.** Parsed with ``json`` only and validated field by
  field; a malformed answer raises rather than being coerced into a score.
- **Payloads are capped**, so one huge trace can't become a huge bill.

Data note: text you send leaves your machine. TypeSafe documents that zero data
retention is an enterprise-only option, so treat the state as *shared with a third
party*. Don't send real employee data through this without deciding that's OK.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx

# --- Where requests may go -----------------------------------------------------

PROVIDERS = {
    "typesafe": {
        "base_url": "https://api.typesafe.ai",
        "key_env": "TYPESAFE_API_KEY",
        "default_model": "jev-1.13.0",
    },
    "gateway": {
        "base_url": "https://gateway.smith.langchain.com",
        "key_env": "LANGSMITH_API_KEY",
        "default_model": "typesafe/jev-1.13.0",
    },
}

ALLOWED_HOSTS = frozenset({"api.typesafe.ai", "gateway.smith.langchain.com"})
ALLOWED_HOST_SUFFIXES = (".smith.langchain.com",)  # regional LangSmith gateways

ENDPOINT_PATH = "/v1/systemone"

# The docs cap state + longest question at 32k tokens. Characters are a crude
# proxy, but a crude cap that fails loudly beats none.
MAX_STATE_CHARS = 60_000
MAX_QUESTIONS = 32  # LangSmith's stated cap; the direct API states none.

RETRYABLE = {429, 529, 500, 502, 503, 504}
MAX_RETRIES = 3
MAX_BACKOFF_S = 10.0


class System1Error(RuntimeError):
    """Any failure talking to (or understanding) a System 1 model."""


def _check_base_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise System1Error(f"Refusing non-https System 1 base URL: {parsed.scheme or '(none)'}://…")
    if host not in ALLOWED_HOSTS and not host.endswith(ALLOWED_HOST_SUFFIXES):
        raise System1Error(
            f"Refusing to send credentials to {host!r}: not an allowlisted System 1 host "
            f"({sorted(ALLOWED_HOSTS)} or *{ALLOWED_HOST_SUFFIXES[0]})."
        )
    return url.rstrip("/")


# --- Building questions --------------------------------------------------------

def noul(instructions: str, *, true: str | None = None, false: str | None = None) -> dict:
    """A yes/no question. Phrase it so a *high* probability means yes."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def choice(instructions: str, options: dict[str, str | None]) -> dict:
    """Pick one of ``options`` ({name: description}). Include an 'unclear' option
    if abstaining is acceptable — it's how you route uncertainty to a human."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict:
    """Rate on ordered ``levels`` (2-10), lowest first. Use sparingly: binary
    questions are easier to define and to check against human labels."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# --- Request / response --------------------------------------------------------

def build_request(state: Any, questions: dict[str, dict], model: str) -> dict:
    """The JSON body for ``POST /v1/systemone``. Pure — no network."""
    if not questions or len(questions) > MAX_QUESTIONS:
        raise System1Error(f"Send 1-{MAX_QUESTIONS} questions, got {len(questions or {})}.")
    if not model:
        raise System1Error("A model id is required (pin a versioned one, e.g. 'jev-1.13.0').")
    size = len(state) if isinstance(state, str) else len(json.dumps(state, default=str))
    if size > MAX_STATE_CHARS:
        raise System1Error(
            f"State is {size} chars, over the {MAX_STATE_CHARS} limit. Send only the part "
            "of the trace the question needs."
        )
    return {"state": state, "model": model, "questions": questions}


def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def parse_response(body: Any, questions: dict[str, dict]) -> dict[str, dict]:
    """Validate a response against the questions asked; return ``answers``.

    Strict on purpose: a missing question, a wrong type, or a probability outside
    [0, 1] raises ``System1Error`` instead of turning into a quiet 0 or 1.
    """
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise System1Error("Response has no 'answers' object.")
    answers = body["answers"]
    for name, q in questions.items():
        a = answers.get(name)
        if not isinstance(a, dict):
            raise System1Error(f"Response is missing an answer for question {name!r}.")
        kind = q["type"]
        if a.get("type") != kind:
            raise System1Error(f"Answer {name!r} has type {a.get('type')!r}, expected {kind!r}.")
        if kind == "noul":
            if not _is_prob(a.get("noul")):
                raise System1Error(f"Answer {name!r}: 'noul' must be a number in [0, 1].")
        elif kind == "choice":
            if a.get("choice") not in q["criteria"]:
                raise System1Error(f"Answer {name!r}: {a.get('choice')!r} is not a declared option.")
            probs = a.get("probabilities")
            if not isinstance(probs, dict) or not all(_is_prob(v) for v in probs.values()):
                raise System1Error(f"Answer {name!r}: 'probabilities' must map options to [0, 1].")
            if not _is_prob(a.get("confidence")):
                raise System1Error(f"Answer {name!r}: 'confidence' must be a number in [0, 1].")
        elif kind == "score":
            levels = len(q["criteria"])
            s = a.get("score")
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s <= levels - 1:
                raise System1Error(f"Answer {name!r}: 'score' must be within 0..{levels - 1}.")
    return answers


class System1Client:
    """Call a System 1 model.

    >>> client = System1Client("typesafe")                 # reads TYPESAFE_API_KEY
    >>> answers = client.ask(
    ...     {"reply": "That didn't go through — the system is offline."},
    ...     {"acknowledges_failure": noul("Does the reply say the action failed?")},
    ... )
    >>> answers["acknowledges_failure"]["noul"]            # e.g. 0.93
    """

    def __init__(
        self,
        provider: str = "typesafe",
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if provider not in PROVIDERS:
            raise System1Error(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}.")
        spec = PROVIDERS[provider]
        self.provider = provider
        self.model = model or spec["default_model"]
        self.base_url = _check_base_url(base_url or os.getenv("SYSTEM1_BASE_URL") or spec["base_url"])
        self._key_env = spec["key_env"]
        # Kept off the instance's public surface and out of __repr__.
        self.__api_key = api_key or os.getenv(self._key_env)
        if not self.__api_key:
            raise System1Error(
                f"No API key: set {self._key_env} in your environment "
                f"(provider {provider!r})."
            )
        self._http = httpx.Client(transport=transport, timeout=timeout)

    def __repr__(self) -> str:  # never include the key
        return f"System1Client(provider={self.provider!r}, model={self.model!r}, host={urlparse(self.base_url).hostname!r})"

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or ""

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]:
        """Ask ``questions`` about ``state``; return validated answers keyed by question."""
        body = build_request(state, questions, self.model)
        url = self.base_url + ENDPOINT_PATH
        headers = {"Authorization": f"Bearer {self.__api_key}", "Content-Type": "application/json"}

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._http.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                if attempt == MAX_RETRIES:
                    raise System1Error(f"Network error calling {self.host}: {type(e).__name__}") from None
                time.sleep(min(2 ** attempt * 0.5, MAX_BACKOFF_S))
                continue
            if resp.status_code in RETRYABLE and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 2 ** attempt * 0.5
                except ValueError:
                    wait = 2 ** attempt * 0.5
                time.sleep(min(wait, MAX_BACKOFF_S))
                continue
            break

        if resp.status_code != 200:
            request_id = resp.headers.get("x-typesafe-request-id", "")
            # Truncated, and the Authorization header is never part of a response.
            raise System1Error(
                f"{self.host} returned HTTP {resp.status_code}"
                + (f" (request id {request_id})" if request_id else "")
                + f": {resp.text[:200]!r}"
            )
        try:
            payload = resp.json()
        except ValueError:
            raise System1Error("Response was not valid JSON.") from None
        return parse_response(payload, questions)


if __name__ == "__main__":
    # Self-tests — no network, no key. httpx.MockTransport stands in for the server.
    SECRET = "sk-test-DO-NOT-LEAK"

    # 1. Question builders produce the documented shapes.
    assert noul("Is it urgent?") == {"type": "noul", "instructions": "Is it urgent?"}
    assert noul("Q", true="yes", false="no")["criteria"] == {"true": "yes", "false": "no"}
    assert choice("Team?", {"a": "A", "b": None})["criteria"] == {"a": "A", "b": None}
    assert score("How mad?", ["calm", "angry"])["criteria"] == ["calm", "angry"]

    # 2. Request building enforces the caps.
    def _fails(fn, *a, **k):
        try:
            fn(*a, **k)
        except System1Error:
            return True
        return False

    q1 = {"ack": noul("Did it admit failure?")}
    assert build_request("hi", q1, "jev-1.13.0") == {"state": "hi", "model": "jev-1.13.0", "questions": q1}
    assert _fails(build_request, "x" * (MAX_STATE_CHARS + 1), q1, "jev-1.13.0")
    assert _fails(build_request, "hi", {}, "jev-1.13.0")
    assert _fails(build_request, "hi", {f"q{i}": noul("?") for i in range(MAX_QUESTIONS + 1)}, "m")
    assert _fails(build_request, "hi", q1, "")

    # 3. Host allowlist: https + known hosts only. This is the exfiltration guard.
    assert _check_base_url("https://api.typesafe.ai/") == "https://api.typesafe.ai"
    assert _check_base_url("https://gateway.smith.langchain.com")
    assert _check_base_url("https://eu.gateway.smith.langchain.com")
    for evil in ("http://api.typesafe.ai", "https://evil.example.com",
                 "https://api.typesafe.ai.evil.example.com", "https://smith.langchain.com.evil.io",
                 "ftp://api.typesafe.ai"):
        assert _fails(_check_base_url, evil), evil

    # 4. Response validation is strict.
    qs = {
        "ack": noul("?"),
        "team": choice("?", {"tech": None, "billing": None}),
        "mad": score("?", ["calm", "so-so", "angry"]),
    }
    good = {"model": "jev-1.13.0", "answers": {
        "ack": {"type": "noul", "noul": 0.93},
        "team": {"type": "choice", "choice": "tech", "confidence": 0.8,
                 "probabilities": {"tech": 0.9, "billing": 0.1}},
        "mad": {"type": "score", "score": 1.4, "confidence": 0.7, "legend": {}, "probabilities": {}},
    }}
    assert parse_response(good, qs)["ack"]["noul"] == 0.93

    def _mut(path, value):
        import copy
        b = copy.deepcopy(good)
        d = b["answers"]
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        return b

    for bad in (
        {}, {"answers": []},
        _mut(["ack", "noul"], 1.5), _mut(["ack", "noul"], True), _mut(["ack", "noul"], "0.9"),
        _mut(["ack", "type"], "choice"),
        _mut(["team", "choice"], "sales"),
        _mut(["team", "probabilities"], {"tech": 2}),
        _mut(["mad", "score"], 3),
        {"answers": {k: v for k, v in good["answers"].items() if k != "ack"}},
    ):
        assert _fails(parse_response, bad, qs), bad

    # 5. End to end against a mock server: request shape, auth, parsing.
    seen: dict = {}

    def ok_handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "ack": {"type": "noul", "noul": 0.07}}})

    c = System1Client("typesafe", api_key=SECRET, transport=httpx.MockTransport(ok_handler))
    out = c.ask({"reply": "All done!"}, q1)
    assert out["ack"]["noul"] == 0.07
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["model"] == "jev-1.13.0" and seen["body"]["questions"] == q1

    # 6. The key never leaks: not via repr, not via an HTTP error.
    assert SECRET not in repr(c) and SECRET not in str(c)
    err_client = System1Client("typesafe", api_key=SECRET,
                               transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    try:
        err_client.ask("x", q1)
        raise AssertionError("401 should raise")
    except System1Error as e:
        assert "401" in str(e) and SECRET not in str(e)

    # 7. Retries: 429 (honouring retry-after) then success; gives up after MAX_RETRIES.
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"answers": {"ack": {"type": "noul", "noul": 0.5}}})

    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        c2 = System1Client("typesafe", api_key=SECRET, transport=httpx.MockTransport(flaky))
        assert c2.ask("x", q1)["ack"]["noul"] == 0.5 and calls["n"] == 3
        always429 = System1Client("typesafe", api_key=SECRET,
                                  transport=httpx.MockTransport(lambda r: httpx.Response(429)))
        assert _fails(always429.ask, "x", q1)
        badjson = System1Client("typesafe", api_key=SECRET,
                                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")))
        assert _fails(badjson.ask, "x", q1)
    finally:
        time.sleep = real_sleep

    # 8. Construction guards: unknown provider, missing key, hostile SYSTEM1_BASE_URL.
    assert _fails(System1Client, "nope", api_key=SECRET)
    os.environ.pop("TYPESAFE_API_KEY", None)
    assert _fails(System1Client, "typesafe")
    assert _fails(System1Client, "typesafe", api_key=SECRET, base_url="https://evil.example.com")
    os.environ["SYSTEM1_BASE_URL"] = "https://evil.example.com"
    try:
        assert _fails(System1Client, "typesafe", api_key=SECRET)
    finally:
        os.environ.pop("SYSTEM1_BASE_URL")

    print("All System 1 client self-tests passed.")
