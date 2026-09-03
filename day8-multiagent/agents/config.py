"""Configuration for day8-multiagent.

Deliberately duplicated from day4-rag/rag/config.py, not imported across
project folders -- every project boundary in this curriculum (Day 1<->Day
3<->Day 4) already talks over HTTP rather than sharing Python code, and
importing rag.config here would pull day4-rag's entire dependency set
(chromadb, fastembed, langchain, groq) into this project as a runtime
coupling, not just a design one. Only the values a READER of day4-rag's
corpus needs are kept -- e.g. no CHUNK_SIZE_WORDS/CHUNK_OVERLAP_WORDS,
since this project never chunks or ingests anything itself.
"""

import os

# --- Retrieval (values reused by agents/researcher.py's tools) -----------
# No separate TOP_K here -- the researcher's search_articles tool has its
# own constants (_SEARCH_TOP_N, _SEARCH_RAW_POOL) in researcher.py, matching
# how agent.py keeps them local to the tool rather than in shared config.

# --- Embedding -------------------------------------------------------------
# MUST match day4-rag's EMBEDDING_MODEL_NAME exactly -- the vectors already
# stored in day4-rag's Chroma collection were produced by this model; a
# different one would silently produce vectors that don't compare
# meaningfully against them (see agents/embed.py's docstring).
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
FASTEMBED_CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR", "/app/.fastembed_cache")

# --- Generation -------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Same pricing constants as day4-rag/rag/config.py (console.groq.com/docs/models,
# confirmed 2026-08-20) -- copied, not imported, same reasoning as the rest
# of this module. Used by agents/llm.py's cost accounting and
# baseline/single_agent.py's frozen copy of Day 7's cost enforcement.
GROQ_PRICE_PER_1M_PROMPT_TOKENS = 0.15
GROQ_PRICE_PER_1M_COMPLETION_TOKENS = 0.60

# --- Storage (day4-rag's already-running Chroma, reused read-only) --------
# Host-side defaults (localhost:8001) match day4-rag's own host-side
# defaults for the exact same reason: both projects' `db` access, in or out
# of Docker, land on the same already-running Chroma instance. Inside this
# project's own docker-compose, these are overridden to
# host.docker.internal:8001 (see docker-compose.yml's header comment for
# why it's host.docker.internal and not a service name -- day4-rag's `db`
# container lives in a different compose project's network).
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8001))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "chunks")
