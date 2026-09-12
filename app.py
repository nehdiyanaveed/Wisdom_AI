import os
import re
import tempfile

import streamlit as st
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ============================================================
# WISDOM-AI
# Quran-based Retrieval-Augmented Generation Application
# ============================================================

st.set_page_config(
    page_title="WISDOM-AI",
    page_icon="📖",
    layout="centered"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

APP_NAME = "WISDOM-AI"

# Google Drive file ID from your provided URL
GOOGLE_DRIVE_FILE_ID = "1AKrKXmXKtfuWeFNkuTMY4xt3ilvubIr-"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 5

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f5f5;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #fafafa;
        border-left: 4px solid #777777;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Download Quran resource
# ------------------------------------------------------------

@st.cache_resource
def download_resource():

    try:
        import gdown

        output_file = os.path.join(
            tempfile.gettempdir(),
            "quran_resource.pdf"
        )

        if not os.path.exists(output_file):

            url = (
                f"https://drive.google.com/uc?id="
                f"{GOOGLE_DRIVE_FILE_ID}"
            )

            gdown.download(
                url,
                output_file,
                quiet=True
            )

        return output_file

    except Exception as e:

        st.error(
            "Unable to download the Quran resource from Google Drive."
        )

        st.exception(e)

        return None


# ------------------------------------------------------------
# Extract PDF text
# ------------------------------------------------------------

@st.cache_resource
def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        try:
            text = page.extract_text()

            if text:
                pages.append(
                    {
                        "page": page_number + 1,
                        "text": text
                    }
                )

        except Exception:
            continue

    return pages


# ------------------------------------------------------------
# Clean text
# ------------------------------------------------------------

def clean_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------------------
# Create chunks
# ------------------------------------------------------------

def create_chunks(pages):

    chunks = []

    for page in pages:

        text = clean_text(page["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page": page["page"]
                    }
                )

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


# ------------------------------------------------------------
# Build FAISS index
# ------------------------------------------------------------

@st.cache_resource
def build_faiss_index(chunks):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ------------------------------------------------------------
# Retrieve relevant Quran passages
# ------------------------------------------------------------

def retrieve_documents(
    query,
    index,
    chunks,
    model,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        retrieved.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            }
        )

    return retrieved


# ------------------------------------------------------------
# Generate answer using Groq
# ------------------------------------------------------------

def generate_answer(
    query,
    retrieved_documents
):

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY")
    )

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured."
        )

        st.stop()

    client = Groq(
        api_key=api_key
    )

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Page: {document['page']}

{document['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are WISDOM-AI, a Quran-focused
retrieval-augmented assistant.

Your job is to answer questions using
ONLY the provided Quran resource.

Rules:

1. Do not invent Quranic verses.
2. Do not fabricate references.
3. Do not claim that a passage exists in the
   Quran if it is not present in the supplied context.
4. If the retrieved context does not contain
   enough information to answer the question,
   clearly say that the provided resource does
   not contain enough information.
5. Distinguish between what is directly stated
   in the supplied Quran resource and your own
   explanation.
6. Keep answers respectful and clear.
7. Do not present personal opinions as Quranic
   teachings.
8. When possible, mention the relevant source
   page.
9. Do not answer unrelated questions.

The purpose of WISDOM-AI is retrieval and
explanation, not replacing qualified scholars.
"""

    user_prompt = f"""
Question:

{query}

Retrieved Quran context:

{context}

Answer the question using the retrieved context.
"""

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_tokens=1000
    )

    return response.choices[0].message.content


# ------------------------------------------------------------
# Main application
# ------------------------------------------------------------

def main():

    st.markdown(
        '<div class="main-title">📖 WISDOM-AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        A Quran-grounded Retrieval-Augmented AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Ask a question related to the Quran."
    )

    st.info(
        "WISDOM-AI retrieves relevant passages "
        "from the supplied Quran resource before "
        "generating an answer."
    )

    # --------------------------------------------------------
    # Load resource
    # --------------------------------------------------------

    with st.spinner(
        "Loading Quran resource..."
    ):

        pdf_path = download_resource()

        if pdf_path is None:
            st.stop()

        pages = extract_text(pdf_path)

        if not pages:

            st.error(
                "No readable text was found in the resource."
            )

            st.stop()

        chunks = create_chunks(pages)

        model = load_embedding_model()

        index = build_faiss_index(chunks)

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        st.header("WISDOM-AI")

        st.write(
            "Quran-based RAG assistant"
        )

        st.divider()

        st.write(
            f"📄 Pages indexed: {len(pages)}"
        )

        st.write(
            f"🧩 Chunks indexed: {len(chunks)}"
        )

        st.write(
            f"🔎 Retrieved passages: {TOP_K}"
        )

        st.divider()

        st.caption(
            "AI-generated responses should be "
            "verified against the original source "
            "and qualified scholarship where appropriate."
        )

    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    query = st.text_input(
        "Ask WISDOM-AI",
        placeholder=(
            "Example: What does the Quran say about patience?"
        )
    )

    if query:

        with st.spinner(
            "Searching the Quran..."
        ):

            retrieved_documents = retrieve_documents(
                query=query,
                index=index,
                chunks=chunks,
                model=model
            )

        if not retrieved_documents:

            st.warning(
                "No relevant passage was retrieved."
            )

            st.stop()

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                query,
                retrieved_documents
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            "### Answer"
        )

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Retrieved sources
        # ----------------------------------------------------

        st.markdown(
            "### Retrieved Sources"
        )

        for i, document in enumerate(
            retrieved_documents,
            start=1
        ):

            with st.expander(
                f"Source {i} • Page {document['page']}"
            ):

                st.write(
                    document["text"]
                )

                st.caption(
                    f"Similarity score: "
                    f"{document['score']:.3f}"
                )


if __name__ == "__main__":
    main()
