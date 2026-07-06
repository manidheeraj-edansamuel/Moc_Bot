import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Simple RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Simple RAG Chatbot")
st.caption("Upload a PDF, ask questions, and get answers grounded in your document.")

# ----------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    llm_backend = st.selectbox(
        "Answer generation backend",
        ["Local (FLAN-T5, no API key needed)", "OpenAI (needs API key)"],
        help="Local mode runs a small free model on your machine. "
             "OpenAI mode gives better answers but needs an API key.",
    )

    openai_api_key = ""
    if llm_backend.startswith("OpenAI"):
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 300, 100, step=50)
    top_k = st.slider("Number of chunks to retrieve (top-k)", 1, 8, 3)

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Upload a PDF\n"
        "2. Text is split into chunks\n"
        "3. Chunks are embedded and stored in a FAISS vector index\n"
        "4. Your question is embedded and matched against the chunks\n"
        "5. The most relevant chunks are sent to the LLM to generate an answer"
    )

# ----------------------------------------------------------------------------
# CACHED MODEL LOADERS (loaded once, reused across reruns)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    # A small, fast, free sentence-embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading local language model...")
def load_local_llm():
    # A small, free, local text-generation model (no API key required)
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")


embedder = load_embedding_model()

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def chunk_text(text, size=800, overlap=100):
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_faiss_index(chunks: list[str]):
    """Embed chunks and build a FAISS similarity search index."""
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product = cosine similarity (since normalized)
    index.add(embeddings)
    return index, embeddings


def retrieve_relevant_chunks(question: str, index, chunks: list[str], k: int) -> list[str]:
    """Embed the question and retrieve the top-k most similar chunks."""
    q_embedding = embedder.encode([question], normalize_embeddings=True)
    q_embedding = np.array(q_embedding, dtype="float32")

    scores, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Combine retrieved chunks and the question into a single LLM prompt."""
    context = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer_local(prompt):
    llm = load_local_llm()

    response = llm(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        truncation=True,
    )

    return response[0]["generated_text"]


def generate_answer_openai(prompt: str, api_key: str) -> str:
    """Generate an answer using the OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using the provided document context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.file_name = None

# ----------------------------------------------------------------------------
# STEP 1 — UPLOAD & PROCESS PDF
# ----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
    with st.spinner("Extracting text and building the vector index..."):
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text.strip():
            st.error("Couldn't extract any text from this PDF. It may be a scanned/image-only PDF.")
        else:
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            index, _ = build_faiss_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name

    st.success(f"Processed **{uploaded_file.name}** into {len(st.session_state.chunks)} chunks. Ready for questions!")

# ----------------------------------------------------------------------------
# STEP 2 — ASK QUESTIONS
# ----------------------------------------------------------------------------
if st.session_state.chunks:
    st.markdown("---")
    st.subheader("💬 Ask a question about the document")

    question = st.text_input("Your question", placeholder="e.g. What is the leave policy?")
    ask_button = st.button("Get Answer", type="primary")

    if ask_button and question.strip():
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            relevant_chunks = retrieve_relevant_chunks(
                question, st.session_state.index, st.session_state.chunks, top_k
            )
            prompt = build_prompt(question, relevant_chunks)

            if llm_backend.startswith("OpenAI"):
                if not openai_api_key:
                    st.error("Please enter your OpenAI API key in the sidebar.")
                    st.stop()
                answer = generate_answer_openai(prompt, openai_api_key)
            else:
                answer = generate_answer_local(prompt)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("🔍 Show retrieved chunks (the evidence used)"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(chunk)
else:
    st.info("👆 Upload a PDF to get started.")
