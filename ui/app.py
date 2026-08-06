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

# `streamlit run ui/app.py` adds this script's own folder (ui/) to
# sys.path, not the project root -- unlike `python -m rag.cli`, which adds
# the current working directory. rag/ lives one level up from here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from rag.ask import ask
from rag.config import GROQ_API_KEY, SIMILARITY_THRESHOLD, TOP_K
from rag.db import count_chunks
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
            f"TOP_K={TOP_K}, SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD} -- chunks below "
            "the threshold are dropped before the LLM ever sees them (the refusal mechanism)."
        )
        chunks = retrieve(question)
        if not chunks:
            st.write("Nothing cleared the similarity threshold -- this is why it refused, if it did.")
        else:
            for c in chunks:
                st.markdown(f"**{c['similarity']:.3f}** — {c['document_title']}")
                st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
elif ask_clicked:
    st.warning("Enter a question first.")
