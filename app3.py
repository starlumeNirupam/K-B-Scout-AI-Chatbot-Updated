import sys
import importlib

try:
    import pysqlite3  # installed via pysqlite3-binary
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # fallback: if not installed or something odd, try to import sqlite3 anyway
    pass

# (optional) sanity check: print the sqlite version to logs
try:
    import sqlite3
    print("SQLite version:", sqlite3.sqlite_version)
except Exception:
    pass
# ----------------------------------------------------------------

import os
import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

import streamlit as st
import pandas as pd

# PDF
from pypdf import PdfReader

# Vector DB
import chromadb
from chromadb.config import Settings

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

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    print("key",key)
    if not key:
        st.error("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    return OpenAI(api_key=key)

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
    """Extract text from a PDF with OCR fallback."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": file.name, "type": "pdf", "page": i+1}))
        else:
            # OCR fallback
            try:
                images = convert_from_path(file.name, first_page=i+1, last_page=i+1, dpi=300)
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img)
                pages.append((ocr_text, {"source": file.name, "type": "pdf", "page": i+1}))
            except:
                # If OCR fails, add empty page
                pages.append(("", {"source": file.name, "type": "pdf", "page": i+1}))
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

def get_chroma_client():
    """Creates a persistent ChromaDB client."""
    persist_dir = "./chromadb_storage"
    try:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        return client
    except Exception as e:
        st.error(f"Could not create persistent client: {e}")
        return None

def get_or_create_collection(chroma_client, collection_name: str = "kb_scout_documents"):
    """Get existing collection or create new one."""
    try:
        # Try to get existing collection first
        collection = chroma_client.get_collection(name=collection_name)
        return collection
    except:
        # Create new collection if it doesn't exist
        try:
            return chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Error creating collection: {e}")
            return None

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[List[float]]:
    """Batches embeddings to avoid hitting request-size limits."""
    if not texts:
        return []
    
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        st.write(f"Processing embedding batch {batch_num}/{total_batches}...")
        
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"Error creating embeddings for batch {batch_num}: {e}")
            return []
    
    return all_embeddings

def add_chunks_to_collection(collection, client: OpenAI, rag_chunks: List[RAGChunk]):
    """Add chunks to persistent collection."""
    if not rag_chunks or not collection:
        return False
    
    valid_chunks = [c for c in rag_chunks if c.text and c.text.strip()]
    if not valid_chunks:
        return False
    
    documents = [c.text for c in valid_chunks]
    metadatas = [c.metadata for c in valid_chunks]
    ids = [c.id for c in valid_chunks]

    embeddings = embed_texts(client, documents)
    
    if not embeddings or len(embeddings) != len(documents):
        return False

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        return True
    except Exception as e:
        st.error(f"Error adding to collection: {e}")
        return False

def retrieve(collection, client: OpenAI, query: str, top_k: int = 6) -> List[Tuple[str, Dict, float]]:
    if not collection:
        return []
    
    count = collection.count()
    if count == 0:
        return []
    
    try:
        q_emb = embed_texts(client, [query])[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        
        scored = list(zip(docs, metas, dists))
        scored.sort(key=lambda x: x[2])
        return scored
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

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

def get_uploaded_files_from_collection(collection):
    """Get list of unique files that have been uploaded to the collection."""
    if not collection:
        return []
    
    try:
        # Get all metadata to find unique source files
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        files = set()
        for meta in metadatas:
            if "source" in meta and "type" in meta:
                files.add((meta["source"], meta["type"]))
        
        return list(files)
    except:
        return []

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

# Custom CSS matching the design
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
            padding: 0 !important;
        }
        
        /* Hide default streamlit elements and spacing */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }
        
        /* Remove all default margins and padding */
        .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Main container styling */
        .main-container {
            background-color: white;
            margin: 0;
            padding: 0;
            overflow: hidden;
            min-height: 10vh;
        }
        
        /* Header styling */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 0;
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
        
        /* Content area */
        .content-area {
            display: flex;
            min-height: calc(100vh - 80px);
            margin: 0;
            padding: 0;
        }
        
        /* Upload section */
        .upload-section {
            flex: 1;
            padding: 30px;
            border-right: 1px solid #e9ecef;
            background-color: #fafbfc;
        }
        
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            background-color: white;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
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
        
        /* Chat section */
        .chat-section {
            flex: 1.2;
            padding: 30px;
            background-color: white;
            display: flex;
            flex-direction: column;
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
        
        /* Button styling */
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
        
        /* Chat input styling */
        .stChatInput > div > div > textarea {
            border-radius: 20px;
            border: 1px solid #e9ecef;
            padding: 12px 20px;
        }
        
        /* Section headers */
        .section-header {
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
            margin: 0 0 15px 0;
        }
        
        /* Remove streamlit spacing */
        .element-container {
            margin: 0 !important;
        }
        
        /* Hide streamlit branding */
        .css-1rs6os, .css-17eq0hr {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize OpenAI client
client = get_openai_client()

# Initialize persistent ChromaDB
ch_client = get_chroma_client()
if not ch_client:
    st.error("Failed to initialize database. Please check your setup.")
    st.stop()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "collection" not in st.session_state:
    # Always use the same collection for persistence
    st.session_state.collection = get_or_create_collection(ch_client, "kb_scout_documents")

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
    
    # File uploader with custom styling
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "csv", "xlsx", "xls", "txt", "doc", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("**Supports:** .txt, .doc, .docx, .xls, .xlsx, .csv, .pdf")
    
    # Show uploaded files count
    if uploaded_files:
        st.markdown(f"**Selected files ({len(uploaded_files)}):**")
        for file in uploaded_files:
            file_type_icon = {
                "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                "txt": "📝", "doc": "📝", "docx": "📝"
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
    
    # Process button
    if uploaded_files:
        if st.button("🚀 Process Files"):
            tokenizer = make_tokenizer()
            rag_chunks: List[RAGChunk] = []
            
            with st.status("Processing your files…", expanded=True) as status:
                total_files = len(uploaded_files)
                
                for file_idx, file in enumerate(uploaded_files, 1):
                    st.write(f"Reading **{file.name}** ({file_idx}/{total_files})...")
                    
                    try:
                        # Check if file already exists in collection
                        existing_files = get_uploaded_files_from_collection(st.session_state.collection)
                        file_already_exists = any(existing_file[0] == file.name for existing_file in existing_files)
                        
                        if file_already_exists:
                            st.info(f"📁 {file.name} already in database, skipping...")
                            continue
                        
                        if file.name.lower().endswith(".pdf"):
                            units = read_pdf(file)
                        elif file.name.lower().endswith(".csv"):
                            units = read_csv(file)
                        elif file.name.lower().endswith((".xlsx", ".xls")):
                            units = read_xlsx(file)
                        else:
                            # Handle txt, doc, docx files
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
                    st.write(f"Adding {len(rag_chunks)} new chunks to permanent database...")
                    success = add_chunks_to_collection(st.session_state.collection, client, rag_chunks)
                    
                    if success:
                        status.update(label="✅ Files processed and permanently stored", state="complete")
                        st.rerun()  # Refresh to show new files
                    else:
                        status.update(label="❌ Failed to process files", state="error")
                else:
                    status.update(label="ℹ️ No new content to add", state="complete")
    
    # Show permanently stored files
    st.markdown("---")
    st.markdown("### Uploaded Files")
    
    uploaded_files_list = get_uploaded_files_from_collection(st.session_state.collection)
    
    if uploaded_files_list:
        st.markdown(f"**{len(uploaded_files_list)} file(s) in database:**")
        for filename, filetype in uploaded_files_list:
            file_icon = {
                "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                "txt": "📝", "doc": "📝", "docx": "📝"
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
    if st.session_state.collection:
        try:
            count = st.session_state.collection.count()
            if count > 0:
                st.markdown(
                    f"""
                    <div class="status-indicator status-ready">
                        🟢 Ready • {count} documents indexed
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
        except:
            st.markdown(
                """
                <div class="status-indicator status-waiting">
                    🟡 Database not ready
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

    # Chat input
    prompt = st.chat_input("Ask K&B Scout AI about your documents...")
    
    if prompt:
        # Check if we have any documents
        if not st.session_state.collection:
            st.warning("Database not initialized. Please refresh the page.")
        else:
            try:
                doc_count = st.session_state.collection.count()
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
                        
                        # Retrieve relevant documents
                        retrieved = retrieve(st.session_state.collection, client, prompt)
                        
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
            except Exception as e:
                st.error(f"Error processing question: {e}")

    # Chat controls at bottom
    if st.session_state.history:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Clear Chat"):
                st.session_state.history = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear All Data"):
                if st.session_state.collection:
                    try:
                        ch_client.delete_collection(name="kb_scout_documents")
                        st.session_state.collection = get_or_create_collection(ch_client, "kb_scout_documents")
                        st.session_state.history = []
                        st.success("All data cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; font-size: 12px; padding: 10px;">
        💡 <strong>Tip:</strong> Upload your documents on the left, then ask questions about them on the right!<br>
        Your files are permanently stored and will be available in future sessions.
    </div>
    """,
    unsafe_allow_html=True
)
