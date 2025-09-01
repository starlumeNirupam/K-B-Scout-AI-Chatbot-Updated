import os
import uuid
import pickle
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd

# PDF
from pypdf import PdfReader

# Vector DB Alternative - FAISS
import faiss

# Tokenization & chunking
import tiktoken

# OpenAI SDK v1
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_openai_client():
    import openai
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    openai.api_key = key
    return openai

def new_uuid() -> str:
    return str(uuid.uuid4())

def make_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def chunk_text(
    text: str,
    tokenizer,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150
) -> List[str]:
    if not text or not text.strip():
        return []
    tokens = tokenizer.encode(text)
    if len(tokens) == 0:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(tokens):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks

def read_pdf(file) -> list[tuple[str, dict]]:
    """Extract text from a PDF."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": file.name, "type": "pdf", "page": i+1}))
    return pages

def read_csv(file) -> List[Tuple[str, Dict]]:
    """Returns (row_text, metadata) per row."""
    try:
        df = pd.read_csv(file)
        rows = []
        for idx, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:
                row_text = " | ".join(row_values)
                rows.append((row_text, {"source": file.name, "type": "csv", "row": int(idx) + 1}))
        return rows
    except Exception as e:
        st.error(f"Error reading CSV {file.name}: {e}")
        return []

def read_xlsx(file) -> List[Tuple[str, Dict]]:
    try:
        df = pd.read_excel(file)
        rows = []
        for idx, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:
                row_text = " | ".join(row_values)
                rows.append((row_text, {"source": file.name, "type": "xlsx", "row": int(idx) + 1}))
        return rows
    except Exception as e:
        st.error(f"Error reading Excel {file.name}: {e}")
        return []

def safe_clean(s: str) -> str:
    if not s:
        return ""
    cleaned = s.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

@dataclass
class RAGChunk:
    id: str
    text: str
    metadata: Dict

# -----------------------------
# ------ Vector Store ---------
# -----------------------------

class SimpleVectorStore:
    """Simple in-memory vector store using FAISS"""
    
    def __init__(self):
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.dimension = 1536  # OpenAI embedding dimension
        
    def add_documents(self, client: OpenAI, chunks: List[RAGChunk]):
        """Add documents to the vector store"""
        if not chunks:
            return False
            
        try:
            # Extract texts
            texts = [chunk.text for chunk in chunks]
            
            # Create embeddings
            st.write("Creating embeddings...")
            embeddings = self.create_embeddings(client, texts)
            
            if not embeddings:
                return False
                
            # Initialize FAISS index if needed
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                
            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store metadata
            self.documents.extend([chunk.text for chunk in chunks])
            self.metadatas.extend([chunk.metadata for chunk in chunks])
            self.ids.extend([chunk.id for chunk in chunks])
            
            st.success(f"Added {len(chunks)} documents to vector store!")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents: {e}")
            return False
    
    def create_embeddings(self, client: OpenAI, texts: List[str], batch_size: int = 100):
        """Create embeddings for texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                st.error(f"Error creating embeddings: {e}")
                return []
                
        return all_embeddings
    
    def search(self, client: OpenAI, query: str, top_k: int = 6):
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
            
        try:
            # Create query embedding
            query_embedding = self.create_embeddings(client, [query])
            if not query_embedding:
                return []
                
            # Search
            query_vector = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        self.metadatas[idx],
                        1 - score  # Convert similarity to distance
                    ))
            
            return results
            
        except Exception as e:
            st.error(f"Error during search: {e}")
            return []
    
    def count(self):
        """Return number of documents"""
        return len(self.documents)
    
    def get_files(self):
        """Get unique files in the store"""
        files = set()
        for meta in self.metadatas:
            if "source" in meta and "type" in meta:
                files.add((meta["source"], meta["type"]))
        return list(files)

def format_context(snippets: List[Tuple[str, Dict, float]]) -> str:
    blocks = []
    for i, (doc, meta, dist) in enumerate(snippets, 1):
        src = meta.get("source", "unknown")
        if meta.get("type") == "pdf":
            loc = f"page {meta.get('page', 'unknown')}"
        else:
            loc = f"row {meta.get('row', 'unknown')}"
        blocks.append(f"[{i}] Source: {src} ({meta.get('type','')}, {loc})\n{doc}")
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """You are K&B Scout AI, a helpful enterprise document assistant.
Follow these rules:
- Use only the information in <context> ... </context>.
- If the answer cannot be found in the context, say you do not have enough information.
- Be concise and cite sources as [#] using the bracket numbers that appear in the context.
- Be friendly and professional in your responses.
"""

def answer_with_rag(client: OpenAI, question: str, context_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<context>\n{context_text}\n</context>\n\nQuestion: {question}\nAnswer:"
        }
    ]

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        stream=True
    )

# -----------------------------
# --------- UI Layer ----------
# -----------------------------

st.set_page_config(
    page_title="K&B Scout AI Enterprise Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .main-container {
            background-color: white;
            border-radius: 12px;
            padding: 0;
            margin: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .app-title {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        
        .app-subtitle {
            font-size: 14px;
            opacity: 0.9;
            margin: 0;
        }
        
        .upload-section {
            padding: 30px;
            border-right: 1px solid #e9ecef;
            background-color: #fafbfc;
        }
        
        .file-item {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 10px 15px;
            margin: 5px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .file-icon {
            color: #667eea;
            font-size: 16px;
        }
        
        .chat-section {
            padding: 30px;
            background-color: white;
        }
        
        .chat-header {
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status-ready {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-waiting {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 500;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .stChatInput > div > div > textarea {
            border-radius: 20px;
            border: 1px solid #e9ecef;
            padding: 12px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize OpenAI client
client = get_openai_client()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="app-header">
        <div style="font-size: 28px;">🤖</div>
        <div>
            <div class="app-title">K&B Scout AI</div>
            <div class="app-subtitle">Enterprise Assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main content area
col1, col2 = st.columns([1, 1.2])

# Left column - File Upload
with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    st.markdown("### Upload your files")
    st.markdown("Drag & drop or click to browse")
    
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("**Supports:** .txt, .xls, .xlsx, .csv, .pdf")
    
    if uploaded_files:
        st.markdown(f"**Selected files ({len(uploaded_files)}):**")
        for file in uploaded_files:
            file_type_icon = {
                "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", "txt": "📝"
            }.get(file.name.split('.')[-1].lower(), "📎")
            
            st.markdown(
                f"""
                <div class="file-item">
                    <span class="file-icon">{file_type_icon}</span>
                    <span>{file.name}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    if uploaded_files:
        if st.button("🚀 Process Files"):
            tokenizer = make_tokenizer()
            rag_chunks: List[RAGChunk] = []
            
            with st.status("Processing your files…", expanded=True) as status:
                total_files = len(uploaded_files)
                
                for file_idx, file in enumerate(uploaded_files, 1):
                    st.write(f"Reading **{file.name}** ({file_idx}/{total_files})...")
                    
                    try:
                        if file.name.lower().endswith(".pdf"):
                            units = read_pdf(file)
                        elif file.name.lower().endswith(".csv"):
                            units = read_csv(file)
                        elif file.name.lower().endswith((".xlsx", ".xls")):
                            units = read_xlsx(file)
                        else:
                            # Handle txt files
                            content = str(file.read(), "utf-8")
                            units = [(content, {"source": file.name, "type": "txt", "page": 1})]
                        
                        st.write(f"Extracted {len(units)} units from {file.name}")
                        
                        # Chunk each unit
                        for unit_idx, (unit_text, meta) in enumerate(units):
                            unit_text = safe_clean(unit_text)
                            if not unit_text:
                                continue
                            
                            chunks = chunk_text(unit_text, tokenizer)
                            
                            for chunk_idx, ch in enumerate(chunks):
                                if ch.strip():
                                    chunk_meta = meta.copy()
                                    chunk_meta["chunk_id"] = chunk_idx + 1
                                    rag_chunks.append(RAGChunk(id=new_uuid(), text=ch, metadata=chunk_meta))
                    
                    except Exception as e:
                        st.error(f"Failed to read {file.name}: {e}")
                        continue
                
                if rag_chunks:
                    st.write(f"Processing {len(rag_chunks)} chunks...")
                    success = st.session_state.vector_store.add_documents(client, rag_chunks)
                    
                    if success:
                        status.update(label="✅ Files processed successfully", state="complete")
                        st.rerun()
                    else:
                        status.update(label="❌ Failed to process files", state="error")
                else:
                    status.update(label="ℹ️ No content to add", state="complete")
    
    # Show uploaded files
    st.markdown("---")
    st.markdown("### Uploaded Files")
    
    uploaded_files_list = st.session_state.vector_store.get_files()
    
    if uploaded_files_list:
        st.markdown(f"**{len(uploaded_files_list)} file(s) processed:**")
        for filename, filetype in uploaded_files_list:
            file_icon = {
                "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", "txt": "📝"
            }.get(filetype, "📎")
            
            st.markdown(
                f"""
                <div class="file-item">
                    <span class="file-icon">{file_icon}</span>
                    <span>{filename}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No files uploaded yet")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right column - Chat Interface
with col2:
    st.markdown('<div class="chat-section">', unsafe_allow_html=True)
    
    # Chat header
    st.markdown('<div class="chat-header">', unsafe_allow_html=True)
    st.markdown("### Chat with K&B Scout AI")
    st.markdown("Ask questions about your uploaded documents")
    
    # Status indicator
    doc_count = st.session_state.vector_store.count()
    if doc_count > 0:
        st.markdown(
            f"""
            <div class="status-indicator status-ready">
                🟢 Ready • {doc_count} documents indexed
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-indicator status-waiting">
                🟡 Upload files to get started
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initial greeting
    if not st.session_state.history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("👋 Hello! I'm **K&B Scout AI**, your enterprise document assistant.")
            st.markdown("I can help you find information from your uploaded files. What would you like to know?")
    
    # Chat history
    for msg in st.session_state.history:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input (must be outside columns)
prompt = st.chat_input("Ask K&B Scout AI about your documents...")

if prompt:
if prompt:
    doc_count = st.session_state.vector_store.count()
    if doc_count == 0:
        # Show message even without documents
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            response = "I'd be happy to help, but I don't have any documents to search through yet. Please upload some files first, and then I can answer questions about their content!"
            st.markdown(response)
            st.session_state.history.append({"role": "assistant", "content": response})
    else:
        # Process question with RAG
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            
            # Search for relevant documents
            retrieved = st.session_state.vector_store.search(client, prompt)
            
            if not retrieved:
                answer = "I couldn't find any relevant information in your uploaded documents for this question."
                placeholder.markdown(answer)
                st.session_state.history.append({"role": "assistant", "content": answer})
            else:
                context_text = format_context(retrieved)
                
                # Stream response
                try:
                    stream = answer_with_rag(client, prompt, context_text)
                    answer_accum = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        answer_accum += delta
                        placeholder.markdown(answer_accum)
                    st.session_state.history.append({"role": "assistant", "content": answer_accum})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# Chat controls (also outside columns)
if st.session_state.history:
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("🔄 Clear Chat"):
            st.session_state.history = []
            st.rerun()
    with col_reset:
        if st.button("🗑️ Clear All Data"):
            st.session_state.vector_store = SimpleVectorStore()
            st.session_state.history = []
            st.success("All data cleared successfully!")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; font-size: 12px; padding: 10px;">
        💡 <strong>Tip:</strong> Upload your documents on the left, then ask questions about them on the right!
    </div>
    """,
    unsafe_allow_html=True
)
