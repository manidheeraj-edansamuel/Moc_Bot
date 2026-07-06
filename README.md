# Simple RAG Chatbot (Free & Local)

A minimal Retrieval-Augmented Generation (RAG) app built for GenAI students.
No API key needed — everything runs locally and for free.

## How it works

```
PDF Upload → Extract Text → Chunk Text → Embed Chunks → Store in FAISS
                                                              │
User Question → Embed Question → Similarity Search ──────────┘
                                        │
                                Retrieve Top Chunks
                                        │
                              Local LLM (FLAN-T5) generates the answer
```

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** `FAISS`
- **LLM:** `google/flan-t5-base` (via Hugging Face `transformers`, runs on CPU)

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The first run will download the embedding model (~90MB) and the FLAN-T5
model (~250MB) from Hugging Face — this requires an internet connection
once, then everything works offline.

## Using the app

1. Open the app in your browser (Streamlit will print a local URL).
2. Upload any PDF (e.g. a college notes file, a policy document).
3. Type a question about the document's content.
4. The app retrieves the most relevant chunks and shows a grounded answer,
   along with the exact source chunks it used (expand "See retrieved chunks").

## Notes for students

- `flan-t5-base` is a small model — good for demos, but answers may be
  simpler than GPT-4/Gemini. Swap in a larger model (e.g. `flan-t5-large`)
  or a hosted API in `load_generator()` for better quality.
- `chunk_size` and `overlap` in `chunk_text()` control how documents are
  split — try tuning these for longer/shorter documents.
- `top_k` in `retrieve_relevant_chunks()` controls how many chunks are
  retrieved per question.
