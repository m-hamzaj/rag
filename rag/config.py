"""Static configuration for the RAG pipeline. Every tunable the brief calls
out (chunk size, top-k) -- plus everything else worth changing without
editing code -- lives here as an env-overridable value, not a literal
buried in chunk.py/retrieve.py.
"""

import os

# --- Chunking ---------------------------------------------------------
# Word-based sliding window (see rag/chunk.py) -- both required to be
# config values by the brief, since they're the first things worth
# sweeping once real retrieval quality is in front of you.
CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", 200))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", 40))

# --- Retrieval ----------------------------------------------------------
# How many chunks come back from a similarity search.
TOP_K = int(os.environ.get("TOP_K", 5))

# Cosine similarity floor a chunk must clear to be considered "good
# enough" evidence. Below this, the question gets refused with "I don't
# know" instead of being answered from weak/irrelevant matches -- this is
# the mechanism, not the LLM's judgment, that makes the refusal real
# rather than a prompt-engineering suggestion the model can ignore.
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))

# --- Embedding ------------------------------------------------------------
# Local, free, no API key -- fastembed runs a small ONNX model entirely on
# CPU (no torch, unlike sentence-transformers directly -- meaningfully
# smaller Docker image and faster cold start for the same model weights).
# The model downloads once, baked into the image. 384-dim output;
# EMBEDDING_DIM must match whatever model this actually is, since the
# pgvector column is declared with a fixed dimension at schema-creation
# time. Changing EMBEDDING_MODEL_NAME to a different-dimension model means
# also updating EMBEDDING_DIM and rebuilding the vector table from scratch.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 384))

# Fixed, explicit cache path rather than trusting fastembed's own default
# (which resolves relative to the package/home directory) -- the Dockerfile
# downloads the model into this exact path at build time, as root, then
# chowns it to the non-root runtime user. An implicit default risks the
# build-time download landing somewhere the runtime user can't read, or
# somewhere that isn't actually inside the image layer that gets kept.
FASTEMBED_CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR", "/app/.fastembed_cache")

# --- Generation -----------------------------------------------------------
# Groq for the actual answer-writing step -- embeddings and retrieval are
# both hand-rolled (see the module docstrings in chunk.py/retrieve.py),
# but generating fluent prose from retrieved chunks is a job for an LLM,
# not something worth reimplementing.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Storage ----------------------------------------------------------
# A separate Postgres from Day 1's -- Day 1's schema has no vector column
# and Day 3 already established the pattern of talking to Day 1 over HTTP
# rather than touching its database directly. Default port 5433, not
# 5432, so both projects' Postgres containers can run at the same time.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

# --- Day 1 API (source of the articles to ingest) -------------------------
DOCUMENTS_API_BASE_URL = os.environ.get("DOCUMENTS_API_BASE_URL", "http://localhost:8000")
