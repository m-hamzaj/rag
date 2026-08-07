"""A small Streamlit interface for asking the RAG system questions and
seeing the answer, its citations, and -- since the whole point of this
project is a hand-written, inspectable retrieval loop, not a framework
black box -- the actual retrieved chunks and similarity scores behind it.

Kept deliberately narrow: this is a way to *use* rag/ask.py, not a
reimplementation of it. All retrieval/generation logic still lives in
rag/ (chunk.py, embed.py, db.py, retrieve.py, generate.py, ask.py); this
file only calls it and renders the result.
"""

import os
import sys
from urllib.parse import urlparse

# `streamlit run ui/app.py` adds this script's own folder (ui/) to
# sys.path, not the project root -- unlike `python -m rag.cli`, which adds
# the current working directory. rag/ lives one level up from here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from rag.ask import ask
from rag.config import GROQ_API_KEY, RELATED_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD, TOP_K
from rag.db import count_chunks, list_documents
from rag.ingest import ingest_new_documents
from rag.retrieve import retrieve

st.set_page_config(page_title="Day 4 RAG", page_icon="🌿", layout="wide")

st.title("Day 4 — RAG over the Day 3 nature/wildlife corpus")
st.caption(
    "Hand-written retrieval: embed the question, pull the top matching chunks, "
    "answer from those chunks only -- or refuse if nothing is a good enough match."
)

try:
    n_chunks = count_chunks()
    st.caption(f"Index: {n_chunks:,} chunks currently stored.")
except Exception as exc:
    st.error(f"Could not reach the vector database: {exc}")
    st.stop()

with st.expander("Add new articles to ask about"):
    st.markdown(
        "Two steps, since this project only *answers* questions -- it doesn't scrape:\n\n"
        "1. Add the article to Day 1 first, via Day 3's crawler dashboard "
        "([localhost:8080](http://localhost:8080), \"Crawl a URL\") or "
        "`docker compose run --rm crawler` in `day3-crawler/`.\n"
        "2. Click below to index whatever's new in Day 1 -- only articles not "
        "already indexed here get chunked and embedded, so this stays fast "
        "regardless of how large the corpus already is."
    )
    if st.button("Index new articles"):
        with st.spinner("Checking Day 1 for articles not yet indexed..."):
            result = ingest_new_documents(quiet=True)
        if result["documents"]:
            st.success(f"Indexed {result['documents']} new article(s) into {result['chunks']} chunks.")
            st.rerun()  # refresh the chunk count and sidebar list above
        else:
            st.info("Nothing new -- every Day 1 article is already indexed.")

with st.sidebar:
    st.header("What's in the corpus?")
    st.caption(
        "Answers only come from these articles -- questions about anything else "
        "get refused. Browse or search before asking."
    )
    try:
        documents = list_documents()
    except Exception as exc:
        documents = []
        st.error(f"Could not list documents: {exc}")

    search = st.text_input("Filter by title", placeholder="tiger, hummingbird, kale...")
    filtered = [d for d in documents if search.lower() in d["title"].lower()] if search else documents
    st.caption(f"{len(filtered)} of {len(documents)} articles shown")

    by_source = {}
    for d in filtered:
        by_source.setdefault(urlparse(d["url"]).netloc, []).append(d)
    for source in sorted(by_source):
        with st.expander(f"{source} ({len(by_source[source])})"):
            for d in sorted(by_source[source], key=lambda x: x["title"]):
                st.markdown(f"- [{d['title']}]({d['url']})")

if not GROQ_API_KEY:
    st.warning(
        "GROQ_API_KEY is not set -- retrieval below will still work, but generating "
        "an actual answer will fail. Set it in .env and restart this container."
    )

question = st.text_input("Ask a question about the corpus", placeholder="How do you make hummingbird nectar?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Embedding question, searching, generating..."):
        result = ask(question)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["citations"]:
        st.subheader("Sources")
        for c in result["citations"]:
            st.markdown(f"- [{c['title']}]({c['url']})")
    else:
        st.caption("No citations -- refused, or the answer used none of the retrieved chunks.")

    with st.expander("Why this answer -- retrieved chunks and similarity scores"):
        st.caption(
            f"TOP_K={TOP_K}, SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}, "
            f"RELATED_SIMILARITY_THRESHOLD={RELATED_SIMILARITY_THRESHOLD} -- chunks below the "
            "related floor never reach the LLM at all; chunks between the two floors get a "
            "caveated related-background answer instead of a direct one."
        )
        retrieved = retrieve(question)
        if retrieved["accepted"]:
            st.markdown("**Accepted -- used for a direct answer:**")
            for c in retrieved["accepted"]:
                st.markdown(f"**{c['similarity']:.3f}** — {c['document_title']}")
                st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        if retrieved["related"]:
            st.markdown("**Related -- used for a caveated background answer, if accepted was empty:**")
            for c in retrieved["related"]:
                st.markdown(f"**{c['similarity']:.3f}** — {c['document_title']}")
                st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
        if not retrieved["accepted"] and not retrieved["related"]:
            st.write("Nothing cleared even the related-topic floor -- this is why it refused, if it did.")
elif ask_clicked:
    st.warning("Enter a question first.")
