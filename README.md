# Simple RAG Chatbot (Streamlit)

A minimal Retrieval-Augmented Generation (RAG) app for learning purposes.
Upload a PDF, ask questions, and get answers grounded in the document —
no fine-tuning required.

## How it works

```
PDF Documents
      │
      ▼
Text Chunking
      │
      ▼
Embedding Model  (all-MiniLM-L6-v2)
      │
      ▼
Vector Database  (FAISS)
      │
User Question ──► Embedding
      │
      ▼
Similarity Search
      │
      ▼
Relevant Chunks
      │
      ▼
Large Language Model  (Local FLAN-T5 or OpenAI GPT)
      │
      ▼
Final Answer
```

## 1. Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> Installing `torch` and `sentence-transformers` may take a few minutes the first time.

## 2. Run the app

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

## 3. Using the app

1. In the sidebar, choose your **answer generation backend**:
   - **Local (FLAN-T5)** — free, runs on your machine, no API key needed. Good for testing and offline demos.
   - **OpenAI** — better quality answers, requires your own OpenAI API key (get one at https://platform.openai.com/api-keys).
2. Upload a PDF (e.g. an HR policy document, lecture notes, a manual).
3. Type a question about the document and click **Get Answer**.
4. Expand **"Show retrieved chunks"** to see exactly which parts of the document were used to generate the answer — this is what makes RAG transparent and reduces hallucination.

## Project structure

```
.
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Ideas for extending this project (great for assignments!)

- Swap FAISS for **ChromaDB** to see a different vector store in action.
- Support multiple PDFs at once (multi-document RAG).
- Add a **chat history** so follow-up questions retain context.
- Try a different embedding model (e.g. `all-mpnet-base-v2` for higher quality).
- Add source page numbers to the retrieved chunks.
- Deploy it on **Streamlit Community Cloud** to share with others.

## Troubleshooting

- **"Couldn't extract any text from this PDF"** — the PDF is likely scanned images rather than real text. You'd need OCR (e.g. `pytesseract`) to handle those.
- **Slow first run** — the embedding/local LLM models are downloaded once and cached; subsequent runs are faster.
- **Local model answers are short/simple** — FLAN-T5-base is a small model. Switch to the OpenAI backend for noticeably better answers.

