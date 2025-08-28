import os
import uuid
import textwrap
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import pdfplumber
import chromadb
from chromadb.config import Settings

# Optional: use Ollama both for embeddings (nomic-embed-text) and for QA generation
import ollama

# =====================
# App Config & Styling
# =====================
st.set_page_config(page_title="RAG: Chroma + Ollama", layout="wide")

st.markdown(
    """
<style>
body, .main { background: linear-gradient(135deg, #ede7f6 0%, #c8e6c9 100%) !important; color: #232323 !important; }
[data-testid="stSidebar"] { background: linear-gradient(160deg, #512da8 0%, #00897b 100%); color: #fff !important; }
[data-testid="stSidebar"] .css-1v0mbdj { color: #fff !important; font-weight: 600; }
h1, h2, h3 { color: #512da8 !important; font-family: 'Montserrat', 'Segoe UI', sans-serif; }
.stButton>button { background: linear-gradient(90deg, #00897b 0%, #512da8 100%); color: #fff !important; border-radius: 14px; font-weight: bold; padding: 0.7em 1.5em; border: none; box-shadow: 0 4px 12px #00897b44; transition: all 0.2s; }
.stButton>button:hover { background: linear-gradient(90deg, #512da8 0%, #00897b 100%); transform: translateY(-2px) scale(1.03); }
.stTextArea textarea, .stTextInput input { border-radius: 15px; background: #f3e5f5 !important; color: #222 !important; box-shadow: 0 2px 8px #512da811; font-size: 1.05em; }
.block-container { background: rgba(255,255,255,0.88) !important; border-radius: 18px; padding: 2rem 2.5rem; margin-top: 1.5rem; box-shadow: 0 6px 24px #512da818; }
.small { opacity: .8; font-size: .9em; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚖️ Local Legal Research AI — ChromaDB RAG (Single Uploader)")
st.caption("Upload PDF / CSV / XLSX once → stored in Chroma → ask questions → get exact snippets (and optional LLM answer).")

# =====================
# Embeddings via Ollama
# =====================
class OllamaEmbeddingFunction:
    """Embedding function wrapper for Chroma that calls Ollama locally.

    Provides the interface Chroma expects: name(), __call__, embed_documents, embed_query.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self._model = model

    # Chroma may call this as a *function*, so expose it as a method
    def name(self) -> str:
        return f"ollama-{self._model}"

    def _embed(self, texts: List[str]):
        vectors = []
        for t in texts:
            t = t if isinstance(t, str) else str(t)
            resp = ollama.embeddings(model=self._model, prompt=t)
            vectors.append(resp["embedding"])  # list[float]
        return vectors

    # Newer Chroma calls the object; older utilities may call embed_* directly
    def __call__(self, texts: List[str]):
        return self._embed(texts)

    def embed_documents(self, texts: List[str]):
        return self._embed(texts)

    def embed_query(self, text: str):
        return self._embed([text])[0]

    def embed_query(self, text: str):
        resp = ollama.embeddings(model=self.model, prompt=text)
        return resp["embedding"]

# =====================
# Chroma Setup (Persistent)
# =====================
CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "legal_docs"

if "embed_model" not in st.session_state:
    st.session_state.embed_model = "nomic-embed-text"  # Good default

embed_model = st.sidebar.selectbox(
    "Embedding model (Ollama)",
    options=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
    index=0,
    help="This should be an Ollama embedding model available on your machine.",
)
st.session_state.embed_model = embed_model

embedding_fn = OllamaEmbeddingFunction(model=st.session_state.embed_model)

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(name=DEFAULT_COLLECTION, embedding_function=embedding_fn)

# =====================
# Utilities
# =====================

def chunk_text(text: str, max_chars: int = 1200, min_chars: int = 80) -> List[str]:
    """Simple paragraph-aware chunker by character length."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur += (p + "\n\n")
        else:
            if len(cur.strip()) >= min_chars:
                chunks.append(cur.strip())
            cur = p + "\n\n"
    if len(cur.strip()) >= min_chars:
        chunks.append(cur.strip())
    return chunks


# (rest of the code remains unchanged)
