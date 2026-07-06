"""
Simple RAG (Retrieval-Augmented Generation) Chatbot
----------------------------------------------------
A minimal, beginner-friendly RAG pipeline for GenAI students.

Pipeline:
  1. Upload a PDF
  2. Extract text
  3. Split text into chunks
  4. Convert chunks into embeddings (Sentence Transformers)
  5. Store embeddings in a vector database (FAISS)
  6. On a question: embed it, do a similarity search, retrieve top chunks
  7. Feed the retrieved chunks + question into a local LLM (FLAN-T5)
  8. Show the generated answer

No API key required — everything runs locally, free of cost.
"""

import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Simple RAG Chatbot", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Cached model loaders (so models load only once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading language model...")
def load_generator():
    # flan-t5-base is small, free, and runs on CPU — good for a classroom demo.
    return pipeline("text2text-generation", model="google/flan-t5-base")


# ---------------------------------------------------------------------------
# Step 1-2: Extract text from PDF
# ---------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


# ---------------------------------------------------------------------------
# Step 3: Chunk the text
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap keeps some context between chunks
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Step 4-5: Create embeddings and store them in a FAISS vector database
# ---------------------------------------------------------------------------
def build_faiss_index(chunks: list[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 = Euclidean distance search
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Step 6: Embed the question and retrieve the most similar chunks
# ---------------------------------------------------------------------------
def retrieve_relevant_chunks(question, chunks, index, embedder, top_k=3):
    q_embedding = embedder.encode([question]).astype("float32")
    _, top_indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in top_indices[0]]


# ---------------------------------------------------------------------------
# Step 7: Generate the final answer using the retrieved chunks as context
# ---------------------------------------------------------------------------
def generate_answer(question, context_chunks, generator) -> str:
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]["generated_text"]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("📄 Simple RAG Chatbot")
st.caption("Upload a PDF and ask questions — answers are grounded in your document, not guessed.")

embedder = load_embedder()
generator = load_generator()

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset / New PDF"):
        st.session_state.chunks = None
        st.session_state.index = None
        st.session_state.file_name = None
        st.rerun()

if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
    with st.spinner("Extracting text, chunking, and building the vector index..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        index = build_faiss_index(chunks, embedder)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.file_name = uploaded_file.name

    st.success(f"'{uploaded_file.name}' processed into {len(chunks)} chunks. Ask a question below!")

if st.session_state.chunks:
    question = st.text_input("Ask a question about the document:", placeholder="e.g. What is the leave policy?")

    if question:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            relevant_chunks = retrieve_relevant_chunks(
                question, st.session_state.chunks, st.session_state.index, embedder
            )
            answer = generate_answer(question, relevant_chunks, generator)

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("🔍 See the retrieved chunks used as context"):
            for i, chunk in enumerate(relevant_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
                st.divider()
else:
    st.info("👆 Upload a PDF to get started.")
