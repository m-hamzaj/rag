import httpx
from groq import APIConnectionError

from agents import llm as llm_module
from conftest import _fake_api_error, _text_message


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_count = 0

    def invoke(self, messages):
        self.invoke_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_call_llm_retries_429_and_eventually_succeeds(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    llm = _FakeLLM([_fake_api_error(429), _fake_api_error(429), _text_message("ok")])

    result = llm_module.call_llm(llm, [])

    assert result.content == "ok"
    assert llm.invoke_count == 3


def test_call_llm_raises_after_exhausting_retries_on_sustained_429(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_module, "RATE_LIMIT_RETRIES", 2)
    llm = _FakeLLM([_fake_api_error(429)] * 10)

    try:
        llm_module.call_llm(llm, [])
        assert False, "expected APIStatusError"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 429
    assert llm.invoke_count == 3  # initial attempt + 2 retries


def test_call_llm_does_not_retry_a_non_429_status_error():
    llm = _FakeLLM([_fake_api_error(400, "tool_use_failed")])

    try:
        llm_module.call_llm(llm, [])
        assert False, "expected APIStatusError"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    assert llm.invoke_count == 1  # no retry -- a 400 isn't transient


def test_call_llm_retries_connection_errors(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm_module, "RATE_LIMIT_RETRIES", 1)
    conn_error = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))
    llm = _FakeLLM([conn_error, _text_message("ok")])

    result = llm_module.call_llm(llm, [])

    assert result.content == "ok"


def test_cost_usd_matches_groq_pricing():
    # 1M prompt tokens alone == GROQ_PRICE_PER_1M_PROMPT_TOKENS.
    from agents.config import GROQ_PRICE_PER_1M_PROMPT_TOKENS

    assert llm_module.cost_usd(1_000_000, 0) == GROQ_PRICE_PER_1M_PROMPT_TOKENS


def test_usage_from_reads_usage_metadata():
    message = _text_message("hi", prompt_tokens=42, completion_tokens=7)

    assert llm_module.usage_from(message) == (42, 7)


def test_call_llm_uses_the_header_when_it_asks_for_longer_than_the_fallback(monkeypatch):
    # A live eval run showed the blind exponential schedule burning
    # 700-950+ seconds of backoff sleep on a single rate-limited question.
    # A header wait LONGER than the blind schedule's current attempt is
    # exactly the case where trusting it helps -- Groq knows more than the
    # blind guess does here.
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: slept.append(seconds))
    llm = _FakeLLM([_fake_api_error(429, headers={"x-ratelimit-reset-tokens": "45s"}), _text_message("ok")])

    llm_module.call_llm(llm, [])

    assert slept == [45.5]  # 45s + the 0.5s buffer -- longer than the attempt-0 fallback (15s)


def test_call_llm_ignores_a_header_that_asks_for_less_than_the_fallback(monkeypatch):
    # Found live: trusting a SHORT header value outright (rather than
    # flooring it at the blind schedule) made an eval run measurably
    # WORSE -- x-ratelimit-reset-tokens reports time until the token
    # bucket has SOME room again, not until a large pending request's full
    # requirement is available, so honoring a short value caused retries
    # to fire too soon and exhaust the retry budget in seconds without the
    # rate limit ever actually clearing (RESULTS.md: 0/12 questions
    # completed on that run). The wait must never go BELOW what the
    # already-proven exponential schedule would have used.
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: slept.append(seconds))
    llm = _FakeLLM([_fake_api_error(429, headers={"x-ratelimit-reset-tokens": "1s"}), _text_message("ok")])

    llm_module.call_llm(llm, [])

    assert slept == [llm_module.RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**0)]  # 15s, not 1.5s


def test_call_llm_prefers_reset_tokens_header_over_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: slept.append(seconds))
    llm = _FakeLLM(
        [
            _fake_api_error(429, headers={"x-ratelimit-reset-tokens": "20s", "retry-after": "99"}),
            _text_message("ok"),
        ]
    )

    llm_module.call_llm(llm, [])

    assert slept == [20.5]  # the smaller of the two headers wins, not retry-after's larger 99


def test_call_llm_falls_back_to_exponential_backoff_without_a_usable_header(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: slept.append(seconds))
    llm = _FakeLLM([_fake_api_error(429), _text_message("ok")])  # no headers at all

    llm_module.call_llm(llm, [])

    assert slept == [llm_module.RATE_LIMIT_BASE_BACKOFF_SECONDS * (2**0)]


def test_parse_duration_seconds_handles_grocs_actual_formats():
    assert llm_module._parse_duration_seconds("547ms") == 0.547
    assert llm_module._parse_duration_seconds("15s") == 15.0
    assert llm_module._parse_duration_seconds("1m26.4s") == 86.4
    assert llm_module._parse_duration_seconds(None) is None
    assert llm_module._parse_duration_seconds("garbage") is None


def test_retry_after_seconds_clamps_a_pathologically_large_header(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: slept.append(seconds))
    llm = _FakeLLM([_fake_api_error(429, headers={"x-ratelimit-reset-tokens": "9999s"}), _text_message("ok")])

    llm_module.call_llm(llm, [])

    assert slept == [llm_module._MAX_HEADER_WAIT_SECONDS]
