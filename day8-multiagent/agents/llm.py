"""Shared LLM-call plumbing used by all three nodes (researcher/writer/
critic) -- one retry/backoff wrapper and one cost-accounting pair, so the
three roles can't silently drift into three different retry policies or
three different pricing formulas.

Same retry shape as day4-rag/rag/agent.py's _call_llm, and wider than that
module's own single-agent numbers, not narrower: a multi-agent run makes
MORE Groq calls per question than Day 7's single loop did (researcher's own
tool-loop steps, plus a writer call, plus a critic call, times however many
revision cycles happen), so it trips Groq's shared "on_demand" per-minute
budget at least as easily. Day 7's own history (module docstring there)
shows these numbers only got wide enough after live 429 storms, not by
guessing correctly up front -- this project should expect the same and
re-tune from real eval_multiagent.py runs, not assume these starting
values are final.

RETRY WAIT TIME TAKES THE HEADER AS A FLOOR, NOT AS THE ANSWER: a live eval
run (RESULTS.md) showed the fixed 15s*2^n exponential schedule burning
700-950+ seconds of backoff sleep on a single rate-limited question, which
looked like pure waste next to Groq's 429 response stating exactly how
long until the exhausted budget refills, via `x-ratelimit-reset-tokens`.
Trusting that header ALONE was tried first and made things measurably
WORSE, not better, on a live rerun (RESULTS.md): the header reports time
until the token bucket has SOME room again (a continuous trickle-refill),
not time until enough room exists for the specific, often multi-thousand-
token request that's actually pending -- so honoring it directly caused
retries to fire too soon, fail again immediately, and exhaust all 5
retries in a few seconds without the budget ever genuinely recovering
(0/12 questions completed, all rate_limited). _retry_after_seconds now
takes max(header-derived wait, the same exponential schedule as before)
-- the header can only make a wait LONGER than the blind guess (when it
knows something the blind schedule doesn't), never shorter (since the
blind schedule already empirically works, just slowly).
"""

import re
import time

from groq import APIConnectionError, APIStatusError
from langchain_core.messages import AIMessage, BaseMessage

from agents.config import GROQ_PRICE_PER_1M_COMPLETION_TOKENS, GROQ_PRICE_PER_1M_PROMPT_TOKENS

RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_BACKOFF_SECONDS = 15

# Upper bound on a single header-derived wait -- a sanity clamp, not a
# tuned value: protects against trusting a pathological/malformed header
# value outright, while still allowing legitimate multi-second waits
# (observed live: several seconds, not milliseconds, once a heavy
# multi-thousand-token prompt was the thing that tripped the limit).
_MAX_HEADER_WAIT_SECONDS = 120

# Go-style duration strings, exactly as Groq's rate-limit headers return
# them (confirmed live: "15s", "1m26.4s", "547ms"). The negative lookahead
# on the minutes group specifically disambiguates "547ms" (547
# milliseconds) from being misparsed as "547" minutes followed by a
# dangling "s" -- both start with digits-then-'m', and without the
# lookahead the regex greedily (and wrongly) consumes the 'm' as minutes.
_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?$")


def _parse_duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    match = _DURATION_RE.match(value.strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = match.groups()
    total = 0.0
    if hours:
        total += int(hours) * 3600
    if minutes:
        total += int(minutes) * 60
    if seconds:
        total += float(seconds)
    if millis:
        total += int(millis) / 1000
    return total


def _retry_after_seconds(exc: APIStatusError, attempt: int) -> float:
    """x-ratelimit-reset-tokens checked first (the confirmed binding
    constraint for this workload), then the standard retry-after header,
    then Groq's separate request-count reset -- whichever is present and
    parses cleanly wins. The result is floored at the same exponential
    schedule call_llm always used (see module docstring for why: a header
    that under-promises must never make the wait SHORTER than the
    already-proven blind schedule, only ever longer when it knows more).
    """
    fallback = RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**attempt)
    headers = getattr(exc.response, "headers", None) or {}
    for header_name in ("x-ratelimit-reset-tokens", "retry-after", "x-ratelimit-reset-requests"):
        seconds = _parse_duration_seconds(headers.get(header_name))
        if seconds is not None:
            # +0.5s buffer -- a reset boundary is exact from Groq's side,
            # not from the clock this process is running on.
            return min(max(seconds + 0.5, fallback), _MAX_HEADER_WAIT_SECONDS)
    return fallback


def call_llm(llm, messages: list[BaseMessage]) -> AIMessage:
    """llm is a node's own bound ChatGroq instance (researcher/writer/critic
    each bind a different toolset -- see their own _get_llm()), not a
    shared one -- only the retry/backoff behavior around invoke() is
    shared, not the model binding itself.
    """
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except APIStatusError as exc:
            if exc.status_code != 429 or attempt == RATE_LIMIT_RETRIES:
                raise
            time.sleep(_retry_after_seconds(exc, attempt))
        except APIConnectionError:
            # No HTTP response at all -- distinct from APIStatusError, same
            # as agent.py's _call_llm. Retried the same way (transient
            # network issues are exactly what retries are for), own
            # attempt budget, still bounded by RATE_LIMIT_RETRIES. No
            # response means no headers to read, so this always uses the
            # blind exponential schedule -- there's nothing else to go on.
            if attempt == RATE_LIMIT_RETRIES:
                raise
            time.sleep(RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**attempt))
    raise AssertionError("unreachable")


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * GROQ_PRICE_PER_1M_PROMPT_TOKENS
        + completion_tokens / 1_000_000 * GROQ_PRICE_PER_1M_COMPLETION_TOKENS
    )


def usage_from(message: AIMessage) -> tuple[int, int]:
    """Returns (prompt_tokens, completion_tokens) -- same fallback shape as
    agent.py's _usage_from (usage_metadata first, raw response_metadata as
    a fallback for langchain-groq versions that don't populate the former
    yet)."""
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    raw = (message.response_metadata or {}).get("token_usage", {})
    return raw.get("prompt_tokens", 0), raw.get("completion_tokens", 0)
