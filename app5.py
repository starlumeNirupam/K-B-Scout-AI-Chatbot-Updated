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

# Only load .env file if it exists (for local development)
if os.path.exists('.env'):
    load_dotenv()

# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_openai_client() -> OpenAI:
    """Get OpenAI client with Railway environment support."""
    
    # Multiple methods to get API key for Railway compatibility
    key = None
    
    # Method 1: Direct environment variable (Railway's preferred method)
    key = os.environ.get("OPENAI_API_KEY")
    
    # Method 2: Try os.getenv as fallback
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    
    # Method 3: Force reload dotenv for local development
    if not key and os.path.exists('.env'):
        load_dotenv(override=True)
        key = os.getenv("OPENAI_API_KEY")
    
    # Method 4: Alternative variable names
    if not key:
        key = os.environ.get("OPENAI_KEY") or os.environ.get("OPEN_AI_API_KEY")
    
    # Debug output for Railway (this will show in Railway logs)
    print(f"API Key found: {bool(key)}")
    if key:
        print(f"API Key starts with: {key[:10]}...")
        print(f"API Key length: {len(key)}")
    else:
        print("Available environment variables:")
        env_keys = [k for k in os.environ.keys() if 'API' in k.upper() or 'OPENAI' in k.upper()]
        print(f"API-related env vars: {env_keys}")
        print("All environment variables:")
        print(list(os.environ.keys()))
    
    if not key:
        st.error("❌ No OpenAI API key found in environment variables.")
        st.error("Please set OPENAI_API_KEY in your Railway service environment variables.")
        
        with st.expander("Debug Information"):
            st.write("Railway Environment Variables Check:")
            st.write(f"RAILWAY_ENVIRONMENT: {os.getenv('RAILWAY_ENVIRONMENT', 'Not detected')}")
            st.write("Available environment variables containing 'API' or 'OPENAI':")
            env_vars = [k for k in os.environ.keys() if 'API' in k.upper() or 'OPENAI' in k.upper()]
            if env_vars:
                st.write(env_vars)
            else:
                st.write("No API-related environment variables found")
                st.write("This suggests the OPENAI_API_KEY environment variable is not set in Railway")
        
        st.stop()
    
    # Validate key format
    if not key.startswith(('sk-', 'sk-proj-')):
        st.error("❌ Invalid OpenAI API key format. Key should start with 'sk-' or 'sk-proj-'")
        st.error(f"Current key starts with: {key[:10]}")
        st.stop()
    
    # Check key length
    if len(key) < 40:
        st.error("❌ OpenAI API key appears to be truncated or incomplete.")
        st.error(f"Current key length: {len(key)} characters (should be 40+ characters)")
        st.stop()
    
    # Test the API key with Railway-friendly timeout and retry
    try:
        print("Testing OpenAI API connection...")
        test_client = OpenAI(api_key=key, timeout=30.0)  # Railway-friendly timeout
        
        # Test API connection
        test_response = test_client.models.list()
        print("OpenAI API connection successful")
        print(f"Number of models available: {len(test_response.data)}")
        
        return test_client
        
    except Exception as e:
        error_msg = str(e)
        print(f"OpenAI API test failed: {error_msg}")
        
        st.error(f"❌ OpenAI API key validation failed: {error_msg}")
        
        # Specific error handling
        if "401" in error_msg or "Incorrect API key" in error_msg:
            st.error("The API key is invalid or has been revoked. Please check your OpenAI account.")
        elif "429" in error_msg:
            st.error("Rate limit exceeded. Please try again later.")
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            st.error("Network timeout or connection issue. This is common on Railway.")
            st.info("Try redeploying your Railway service or check Railway's status page.")
        else:
            st.error("There might be a network connectivity issue.")
        
        with st.expander("Troubleshooting Steps"):
            st.write("1. Verify your API key is correctly set in Railway service variables (not project variables)")
            st.write("2. Ensure your OpenAI account has available credits")
            st.write("3. Try redeploying your Railway service")
            st.write("4. Check if Railway is experiencing network issues")
            st.write("5. Verify the API key works in a local environment")
        
        st.stop()

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
    try:
        reader = PdfReader(file)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, {"source": file.name, "type": "pdf", "page": i+1}))
            else:
                # OCR fallback (may not work on Railway due to system dependencies)
                try:
                    images = convert_from_path(file.name, first_page=i+1, last_page=i+1, dpi=300)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img)
                    pages.append((ocr_text, {"source": file.name, "type": "pdf", "page": i+1}))
                except Exception as ocr_error:
                    print(f"OCR failed for page {i+1} of {file.name}: {ocr_error}")
                    # If OCR fails, add empty page
                    pages.append(("", {"source": file.name, "type": "pdf", "page": i+1}))
        return pages
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return []

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
    """Creates a persistent ChromaDB client with Railway compatibility."""
    # Use Railway-appropriate storage path
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
        persist_dir = "/tmp/chromadb_storage"  # Railway ephemeral storage
        print("Using Railway ephemeral storage for ChromaDB")
    else:
        persist_dir = "./chromadb_storage"  # Local development
        print("Using local storage for ChromaDB")
    
    try:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        print(f"ChromaDB client created successfully at {persist_dir}")
        return client
    except Exception as e:
        print(f"Could not create persistent client: {e}")
        st.error(f"Could not create persistent client: {e}")
        
        # Fallback to in-memory client
        try:
            print("Falling back to in-memory ChromaDB client")
            st.warning("Using in-memory database (data will not persist between deployments)")
            return chromadb.Client()
        except Exception as fallback_error:
            print(f"Complete database initialization failed: {fallback_error}")
            st.error(f"Complete database initialization failed: {fallback_error}")
            return None

def get_or_create_collection(chroma_client, collection_name: str = "kb_scout_documents"):
    """Get existing collection or create new one."""
    if not chroma_client:
        return None
        
    try:
        # Try to get existing collection first
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
        return collection
    except:
        # Create new collection if it doesn't exist
        try:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
            return collection
        except Exception as e:
            print(f"Error creating collection: {e}")
            st.error(f"Error creating collection: {e}")
            return None

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 50) -> List[List[float]]:
    """Batches embeddings with Railway-friendly settings."""
    if not texts:
        return []
    
    print(f"Creating embeddings for {len(texts)} texts using {model}")
    
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Progress tracking for Railway logs
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        print(f"Processing embedding batch {batch_num}/{total_batches}...")
        st.write(f"Processing embedding batch {batch_num}/{total_batches}...")
        
        batch = texts[i:i + batch_size]
        try:
            # Railway-friendly API call with timeout and retry logic
            resp = client.embeddings.create(
                input=batch, 
                model=model,
                timeout=60.0  # Increased timeout for Railway
            )
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
            print(f"Successfully processed batch {batch_num}")
            
        except Exception as e:
            print(f"Error creating embeddings for batch {batch_num}: {e}")
            st.error(f"Error creating embeddings for batch {batch_num}: {e}")
            
            if "timeout" in str(e).lower():
                st.error("Network timeout occurred. Try uploading fewer files at once.")
            elif "rate_limit" in str(e).lower() or "429" in str(e):
                st.error("Rate limit hit. Please wait a moment and try again.")
            
            return []
    
    print(f"Successfully created {len(all_embeddings)} embeddings")
    return all_embeddings

def add_chunks_to_collection(collection, client: OpenAI, rag_chunks: List[RAGChunk]):
    """Add chunks to persistent collection."""
    if not rag_chunks or not collection:
        return False
    
    valid_chunks = [c for c in rag_chunks if c.text and c.text.strip()]
    if not valid_chunks:
        return False
    
    print(f"Adding {len(valid_chunks)} chunks to collection")
    
    documents = [c.text for c in valid_chunks]
    metadatas = [c.metadata for c in valid_chunks]
    ids = [c.id for c in valid_chunks]

    embeddings = embed_texts(client, documents)
    
    if not embeddings or len(embeddings) != len(documents):
        print("Failed to create embeddings or count mismatch")
        return False

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print("Successfully added chunks to collection")
        return True
    except Exception as e:
        print(f"Error adding to collection: {e}")
        st.error(f"Error adding to collection: {e}")
        return False

def retrieve(collection, client: OpenAI, query: str, top_k: int = 6) -> List[Tuple[str, Dict, float]]:
    if not collection:
        return []
    
    try:
        count = collection.count()
        print(f"Collection has {count} documents")
        if count == 0:
            return []
    except Exception as e:
        print(f"Error getting collection count: {e}")
        return []
    
    try:
        print(f"Creating query embedding for: {query[:100]}...")
        q_emb = embed_texts(client, [query])[0]
        
        print(f"Querying collection for top {min(top_k, count)} results...")
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
        
        print(f"Retrieved {len(scored)} relevant documents")
        return scored
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
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
    except Exception as e:
        print(f"Error getting uploaded files list: {e}")
        return []

SYSTEM_PROMPT = """You are K&B Scout AI, a helpful enterprise document assistant.
Follow these rules:
- Use only the information in <context> ... </context>.
- If the answer cannot be found in the context, say you do not have enough information.
- Be concise and cite sources as [#] using the bracket numbers that appear in the context.
- Be friendly and professional in your responses.
"""

def answer_with_rag(client: OpenAI, question: str, context_text: str):
    print(f"Generating response for question: {question[:100]}...")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<context>\n{context_text}\n</context>\n\nQuestion: {question}\nAnswer:"
        }
    ]

    try:
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            stream=True,
            timeout=60.0  # Railway-friendly timeout
        )
    except Exception as e:
        print(f"Error generating response: {e}")
        st.error(f"Error generating response: {e}")
        return None

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
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
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

# Display Railway environment info
if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
    print("Running on Railway environment")
    with st.sidebar:
        st.info("🚂 Running on Railway")
        with st.expander("Environment Info"):
            st.code(f"""
Railway Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'Detected')}
Project ID: {os.getenv('RAILWAY_PROJECT_ID', 'N/A')}
Service ID: {os.getenv('RAILWAY_SERVICE_ID', 'N/A')}
API Key Present: {bool(os.getenv('OPENAI_API_KEY'))}
            """)

# Initialize OpenAI client with Railway-optimized error handling
print("Initializing OpenAI client...")
try:
    client = get_openai_client()
    print("OpenAI client initialized successfully")
except:
    print("Failed to initialize OpenAI client")
    client = None

# Initialize persistent ChromaDB
print("Initializing ChromaDB client...")
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
    
    # Show API status
    if not client:
        st.error("⚠️ OpenAI API not available. File processing disabled.")
        st.markdown("Please check your OPENAI_API_KEY environment variable.")
    
    # File uploader with custom styling
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "csv", "xlsx", "xls", "txt", "doc", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=(client is None)
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
    if uploaded_files and client:
        if st.button("🚀 Process Files"):
            print(f"Processing {len(uploaded_files)} files...")
            
            tokenizer = make_tokenizer()
            rag_chunks: List[RAGChunk] = []
            
            with st.status("Processing your files…", expanded=True) as status:
                total_files = len(uploaded_files)
                
                for file_idx, file in enumerate(uploaded_files, 1):
                    st.write(f"Reading **{file.name}** ({file_idx}/{total_files})...")
                    print(f"Processing file {file_idx}/{total_files}: {file.name}")
                    
                    try:
                        # Check if file already exists in collection
                        existing_files = get_uploaded_files_from_collection(st.session_state.collection)
                        file_already_exists = any(existing_file[0] == file.name for existing_file in existing_files)
                        
                        if file_already_exists:
                            st.info(f"📁 {file.name} already in database, skipping...")
                            print(f"File {file.name} already exists, skipping...")
                            continue
                        
                        # Process different file types
                        units = []
                        file_ext = file.name.lower().split('.')[-1]
                        
                        if file_ext == "pdf":
                            units = read_pdf(file)
                        elif file_ext == "csv":
                            units = read_csv(file)
                        elif file_ext in ["xlsx", "xls"]:
                            units = read_xlsx(file)
                        elif file_ext in ["txt", "doc", "docx"]:
                            try:
                                content = str(file.read(), "utf-8", errors='ignore')
                                if content.strip():
                                    units = [(content, {"source": file.name, "type": file_ext, "page": 1})]
                            except Exception as txt_error:
                                print(f"Error reading text file {file.name}: {txt_error}")
                                st.warning(f"Could not read {file.name}: {txt_error}")
                                continue
                        
                        if not units:
                            print(f"No content extracted from {file.name}")
                            st.warning(f"No content extracted from {file.name
