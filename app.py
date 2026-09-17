import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# WISDOM-AI
# Quran-grounded Retrieval-Augmented Generation
# ============================================================

st.set_page_config(
    page_title="WISDOM-AI",
    page_icon="📖",
    layout="centered",
)

APP_NAME = "WISDOM-AI"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

INDEX_PATH = Path("data/quran.index")
METADATA_PATH = Path("data/quran_chunks.json")

TOP_K = 5


# ------------------------------------------------------------
# Page styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 28px;
    }

    .source-card {
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    """Load FAISS index, metadata, and embedding model once per app process."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INDEX_PATH}. Run preprocess.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METADATA_PATH}. Run preprocess.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    return index, chunks, model


def retrieve(query, index, chunks, model, top_k=TOP_K):
    """Retrieve the most semantically similar Quran passages."""

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def build_context(results):
    """Create a compact, source-labelled context for the LLM."""

    blocks = []

    for i, item in enumerate(results, start=1):
        reference = item.get("reference", "Unknown reference")
        text = item.get("text", "").strip()

        blocks.append(
            f"[SOURCE {i} | {reference}]\n{text}"
        )

    return "\n\n".join(blocks)


def generate_answer(question, results):
    """Generate a grounded answer with Groq."""

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it in Streamlit "
            "Community Cloud → App settings → Secrets."
        )
        st.stop()

    client = Groq(api_key=api_key)

    context = build_context(results)

    system_prompt = """
You are WISDOM-AI, a Quran-grounded retrieval-augmented
assistant.

Your knowledge source for this answer is ONLY the retrieved
Quran passages supplied in the user message.

Follow these rules strictly:

1. Answer the user's question using the retrieved passages.
2. Do not invent Quranic verses, references, quotations,
   or religious claims.
3. Never fabricate a Surah or Ayah number.
4. If the retrieved passages are insufficient, say so clearly.
5. Distinguish explanation from what the Quran passage itself
   states.
6. When relevant, cite the Surah and Ayah reference in the
   answer using the supplied references.
7. Do not pretend to issue a religious ruling or replace a
   qualified scholar.
8. If the question is unrelated to the Quran resource, explain
   that WISDOM-AI is designed for Quran-related questions.
9. Keep the answer respectful, clear, and reasonably concise.
10. Do not cite material that is not present in the retrieved
    context.
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED QURAN PASSAGES:
{context}

Write a grounded answer to the question.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content


def main():
    st.markdown(
        '<div class="title">📖 WISDOM-AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'A Quran-grounded Retrieval-Augmented AI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Ask a question related to the Quran. "
        "WISDOM-AI retrieves relevant passages from "
        "the supplied Quran resource before generating "
        "an answer."
    )

    try:
        index, chunks, model = load_resources()
    except Exception as exc:
        st.error("The WISDOM-AI knowledge base is not ready.")
        st.code(str(exc))
        st.info(
            "Run preprocess.py with the supplied Quran PDF, "
            "then place quran.index and quran_chunks.json "
            "inside the data/ folder."
        )
        st.stop()

    with st.sidebar:
        st.header("WISDOM-AI")
        st.caption("Quran-grounded RAG assistant")
        st.divider()
        st.write(f"📚 Indexed passages: {len(chunks):,}")
        st.write(f"🔎 Retrieved per query: {TOP_K}")
        st.write(f"🧠 Embeddings: {MODEL_NAME}")
        st.write(f"🤖 LLM: {GROQ_MODEL}")
        st.divider()
        st.caption(
            "Responses are AI-generated and should be checked "
            "against the original source and qualified scholarship "
            "when a matter requires interpretation or a ruling."
        )

    question = st.text_input(
        "Ask WISDOM-AI",
        placeholder="Example: What does the Quran say about patience?",
    )

    if not question.strip():
        return

    with st.spinner("Searching the Quran..."):
        results = retrieve(
            question.strip(),
            index,
            chunks,
            model,
        )

    if not results:
        st.warning("No relevant passage was retrieved.")
        return

    with st.spinner("Generating a grounded answer..."):
        answer = generate_answer(question.strip(), results)

    st.markdown("### Answer")
    st.markdown(answer)

    st.markdown("### Retrieved Quran passages")

    for i, item in enumerate(results, start=1):
        reference = item.get("reference", "Unknown")
        score = item.get("score", 0.0)

        with st.expander(
            f"{i}. {reference}  •  similarity {score:.3f}"
        ):
            st.write(item.get("text", ""))


if __name__ == "__main__":
    main()
