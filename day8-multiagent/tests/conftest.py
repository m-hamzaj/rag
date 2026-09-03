"""Shared test helpers/fixtures for the node- and graph-level tests.

Mocking seam: agents/llm.py's call_llm is patched ONCE, at
agents.llm.call_llm itself -- researcher.py/writer.py/critic.py all invoke
it as `llm_module.call_llm(...)` (a module-attribute lookup at call time,
not a name bound at import time -- see agents/researcher.py's import
comment for why that distinction matters), so one patch here covers all
three nodes' LLM calls, in the exact order they actually fire across a
whole graph run. This mirrors day4-rag/tests/test_agent.py's
_patch_llm_responses seam one level up, for the same reason: tests should
exercise the real retry/error-handling code in agents/llm.py, not bypass it.
"""

import httpx
import pytest
from groq import APIStatusError
from langchain_core.messages import AIMessage

from agents import critic as critic_module
from agents import llm as llm_module
from agents import researcher as researcher_module
from agents import writer as writer_module

_ARTICLE_A = {
    "document_url": "https://x/a",
    "document_title": "Article A",
    "chunk_index": 0,
    "text": "Article A chunk text",
    "similarity": 0.6,
}
_ARTICLE_B = {
    "document_url": "https://x/b",
    "document_title": "Article B",
    "chunk_index": 0,
    "text": "Article B chunk text",
    "similarity": 0.5,
}


@pytest.fixture(autouse=True)
def _fake_groq_api_key(monkeypatch):
    """Every node's _get_llm() raises RuntimeError on a falsy GROQ_API_KEY
    -- give all three a fake key by default so tests don't need to repeat
    this, and reset each node's cached _llm/_llm_key_model so a fake
    ChatGroq installed by one test (see test_*_llm_is_bound_with_*) never
    leaks into another. A test that specifically wants to exercise the
    missing-key path overrides this by setting GROQ_API_KEY back to None
    itself.
    """
    for mod in (researcher_module, writer_module, critic_module):
        monkeypatch.setattr(mod, "GROQ_API_KEY", "fake-key")
        monkeypatch.setattr(mod, "_llm", None)
        monkeypatch.setattr(mod, "_llm_key_model", None)


def _tool_call_message(
    name: str, arguments: dict, call_id: str = "call_1", prompt_tokens: int = 100, completion_tokens: int = 20
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def _text_message(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def _fake_api_error(status_code: int, message: str = "error", headers: dict | None = None) -> APIStatusError:
    fake_response = httpx.Response(
        status_code, headers=headers or {}, request=httpx.Request("POST", "https://api.groq.com/x")
    )
    return APIStatusError(message, response=fake_response, body=None)


def _patch_llm_calls(monkeypatch, responses: list):
    """Each call to llm_module.call_llm (from ANY node) returns the next
    item in `responses`, in order -- an AIMessage to return, or an
    exception instance to raise. calls["seen_messages"] records the exact
    message list each call was invoked with, calls["seen_llm"] the bound
    llm object each call received (useful for asserting which node's own
    bound tools were used).
    """
    calls = {"count": 0, "seen_messages": [], "seen_llm": []}

    def fake_call_llm(llm, messages):
        calls["seen_llm"].append(llm)
        calls["seen_messages"].append(list(messages))
        i = calls["count"]
        calls["count"] += 1
        item = responses[i]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm_module, "call_llm", fake_call_llm)
    return calls


def _patch_corpus(monkeypatch, search_result=None, article_chunks=None):
    """Patches the corpus functions as researcher.py itself references them
    (`from agents.db import ...` / `from agents.embed import ...` binds
    names into researcher.py's own namespace at import time), same pattern
    day4-rag/tests/test_agent.py already uses for agent.py.
    """
    monkeypatch.setattr(researcher_module, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        researcher_module, "search_similar_chunks", lambda embedding, top_k: search_result or [_ARTICLE_A, _ARTICLE_B]
    )
    monkeypatch.setattr(researcher_module, "get_chunks_by_document", lambda url: article_chunks or [])
