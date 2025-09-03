import sys
import importlib
import os
import uuid
import json  # Added for chat history persistence
import datetime  # Added for chat history timestamps
import tempfile # Added for robust PDF OCR handling
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Note: pytesseract and pdf2image are for OCR fallback.
# These imports are here, but the OCR logic in read_pdf is commented out by default
# as it requires external installations (Tesseract OCR, poppler utilities for pdf2image).
# If you need OCR, uncomment the relevant lines in read_pdf and ensure these are installed.
from pdf2image import convert_from_path # Uncommented
from PIL import Image # Uncommented
import pytesseract # Uncommented

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

# This is often used to fix issues with sqlite3 in some environments,
# especially when using chromadb with older python/pip versions or specific builds.
# Keep it as it's in the original requirements.txt.
# It should be at the very top for effect, moving it here.
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# (optional) sanity check: print the sqlite version to logs
try:
    import sqlite3
    # print("SQLite version:", sqlite3.sqlite_version) # Commented for cleaner output
except Exception:
    pass
# ----------------------------------------------------------------


# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_openai_client() -> OpenAI:
    """Initializes OpenAI client using API key from .env or secrets."""
    # load_dotenv(override=True) # Already called globally at the top
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Fallback to Streamlit secrets if not in .env
        key = st.secrets.get("OPENAI_API_KEY")
    
    if not key:
        st.error("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=key.strip()) # Ensure no leading/trailing whitespace

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
    """Splits text into token-based chunks with overlap."""
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
    
    # Create a temporary file to save the uploaded PDF content for pdf2image (OCR)
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, {"source": file.name, "type": "pdf", "page": i+1}))
            else:
                # OCR fallback for pages with no extractable text
                try:
                    # Use the temporary file path for pdf2image
                    images = convert_from_path(tmp_file_path, first_page=i+1, last_page=i+1, dpi=300)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img)
                    
                    if ocr_text.strip():
                        pages.append((ocr_text, {"source": file.name, "type": "pdf", "page": i+1}))
                    else:
                        # Add empty page if OCR also yields nothing
                        pages.append(("", {"source": file.name, "type": "pdf", "page": i+1}))
                except Exception as ocr_e:
                    st.warning(f"OCR failed for page {i+1} of {file.name}: {ocr_e}. Page will be empty.")
                    pages.append(("", {"source": file.name, "type": "pdf", "page": i+1})) # Add empty page
    finally:
        # Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return pages

def read_csv(file) -> List[Tuple[str, Dict]]:
    """Returns (row_text, metadata) per row for CSV files."""
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
    """Returns (row_text, metadata) per row for XLSX/XLS files."""
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

def read_text(file) -> List[Tuple[str, Dict]]:
    """Reads content from a text-based file."""
    try:
        content = file.getvalue().decode("utf-8")
        return [(content, {"source": file.name, "type": "txt", "page": 1})]
    except Exception as e:
        st.error(f"Error reading text file {file.name}: {e}")
        return []

def safe_clean(s: str) -> str:
    """Removes problematic characters and excessive whitespace."""
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
            ids=ids[:len(embeddings)], # Ensure IDs match embeddings length
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
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
        q_emb = embed_texts(client, [query])
        if not q_emb: # Ensure embedding was successful
            return []
        
        res = collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        
        scored = list(zip(docs, metas, dists))
        scored.sort(key=lambda x: x[2]) # Sort by distance (lower is better for cosine)
        return scored
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

def format_context(snippets: List[Tuple[str, Dict, float]]) -> str:
    """Formats retrieved snippets into a string for LLM context."""
    blocks = []
    for i, (doc, meta, dist) in enumerate(snippets, 1):
        src = meta.get("source", "unknown")
        loc_parts = []
        if meta.get("type") == "pdf":
            loc_parts.append(f"page {meta.get('page', 'unknown')}")
        elif "row" in meta: # For CSV/XLSX
            loc_parts.append(f"row {meta.get('row', 'unknown')}")
        else: # For TXT or other general types
            loc_parts.append(meta.get('type', 'document'))

        location_str = ", ".join(loc_parts)
        blocks.append(f"[{i}] Source: {src} ({location_str})\n{doc}")
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
# ------ Chat History Utils ----
# -----------------------------

CHAT_SESSIONS_DIR = "./chat_sessions"

def _get_chat_filepath(chat_id: str) -> str:
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    return os.path.join(CHAT_SESSIONS_DIR, f"{chat_id}.json")

def _load_all_chat_metas() -> List[Dict]:
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
    chat_metas = []
    for filename in os.listdir(CHAT_SESSIONS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHAT_SESSIONS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chat_metas.append({
                        "id": data.get("id", filename.replace(".json", "")),
                        "name": data.get("name", "Unnamed Chat"),
                        "timestamp": data.get("timestamp", datetime.datetime.now().isoformat())
                    })
            except Exception as e:
                st.warning(f"Could not load chat metadata from {filename}: {e}")
    # Sort by timestamp, newest first
    chat_metas.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return chat_metas

def _save_chat_session(chat_id: str, chat_name: str, chat_history: List[Dict]):
    filepath = _get_chat_filepath(chat_id)
    session_data = {
        "id": chat_id,
        "name": chat_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "messages": chat_history
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

def _load_chat_session_by_id(chat_id: str) -> Tuple[str, List[Dict]]:
    filepath = _get_chat_filepath(chat_id)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("name", "Unnamed Chat"), data.get("messages", [])
    except FileNotFoundError:
        st.error(f"Chat session {chat_id} not found.")
        return "New Chat", []
    except Exception as e:
        st.error(f"Error loading chat session {chat_id}: {e}")
        return "New Chat", []

def _delete_chat_session_by_id(chat_id: str):
    filepath = _get_chat_filepath(chat_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        st.success(f"Chat session '{chat_id}' deleted.")
    else:
        st.warning(f"Chat session {chat_id} not found on disk.")

def _generate_intelligent_name(client: OpenAI, first_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Generate a concise, 3-5 word title for the following chat based on the user's first message. Do not include quotes."},
                {"role": "user", "content": f"First message: '{first_message}'\nTitle:"}
            ],
            max_tokens=15,
            temperature=0.3
        )
        name = response.choices[0].message.content.strip()
        if name:
            # Remove any leading/trailing quotes if the model added them
            name = name.strip('"')
            return name
    except Exception as e:
        st.warning(f"Could not generate intelligent chat name: {e}")
    # Fallback to truncated first message + timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"{first_message[:30]}... ({timestamp})"


# Global client instance from cache
@st.cache_resource
def get_cached_openai_client():
    return get_openai_client()
client = get_cached_openai_client()

def _new_chat_session_and_refresh(client: OpenAI, first_message: str = None):
    new_id = new_uuid()
    new_name = "New Chat"
    if first_message:
        with st.spinner("Generating chat title..."):
            new_name = _generate_intelligent_name(client, first_message)
    
    st.session_state.current_chat_id = new_id
    st.session_state.current_chat_name = new_name
    st.session_state.history = []
    _save_chat_session(new_id, new_name, []) # Save empty session immediately
    st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list
    st.rerun() # Refresh UI

def _load_chat_and_refresh(chat_id: str):
    name, history = _load_chat_session_by_id(chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.current_chat_name = name
    st.session_state.history = history
    st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list
    st.rerun()

def _delete_chat_and_refresh(chat_id: str):
    _delete_chat_session_by_id(chat_id)
    st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list
    # If the deleted chat was the current one, create a new one
    if st.session_state.current_chat_id == chat_id:
        _new_chat_session_and_refresh(client) # Automatically creates and reloads
    st.rerun()

def _rename_current_chat(new_name: str):
    st.session_state.current_chat_name = new_name
    _save_chat_session(st.session_state.current_chat_id, new_name, st.session_state.history)
    st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list
    st.success("Chat renamed!")
    # No rerun needed here, simple UI update is fine

# -----------------------------
# --------- UI Layer ----------
# -----------------------------

st.set_page_config(
    page_title="K&B Scout AI Enterprise Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for adaptive light/dark theme styling and visual appeal
st.markdown(
    """
    <style>
        /* Base styles for the Streamlit app. Ensure the overall app container
           is a flex column to properly distribute vertical space. */
        div[data-testid="stAppViewContainer"] {
            display: flex;
            flex-direction: column;
            min-height: 100vh; /* Make the app take full viewport height */
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Define custom CSS variables for consistent theming */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --header-text-color: white;
            --border-radius: 12px;
            --shadow-subtle: 0 1px 3px rgba(0,0,0,0.05);
            --shadow-medium: 0 4px 12px rgba(102, 126, 234, 0.4);
            --shadow-strong: 0 6px 16px rgba(102, 126, 234, 0.5);
            --st-primary-color: #667eea; /* Align Streamlit primary color with our design */
            --st-secondary-text: #6c757d; /* Consistent secondary text color */
        }

        /* Light Mode specific colors (defaults for [data-theme="light"]) */
        [data-theme="light"] {
            --background-color: #ffffff; /* Overall app background */
            --content-bg: #ffffff; /* Background for main content areas */
            --content-text-color: #333333; /* Dark text for content area */
            --border-color: #e9ecef;
            --chat-bg-assistant: #f8f9fa; /* Light grey for assistant bubble */
            --chat-bg-user: #e6f7ff; /* Light blue for user bubble */
            --file-item-bg: #ffffff;
            --component-bg-color: #f0f2f6; /* Streamlit's default light gray for some elements */
        }

        /* Dark Mode specific colors */
        [data-theme="dark"] {
            --background-color: #0e1117; /* Streamlit's default dark background */
            --content-bg: #1e1e1e; /* Dark background for main content areas */
            --content-text-color: #e0e0e0; /* Light text color for dark mode */
            --border-color: #333333; /* Darker border */
            --chat-bg-assistant: #2d2d2d; /* Slightly lighter dark for assistant bubble */
            --chat-bg-user: #3c3c3c; /* A bit lighter dark for user bubble */
            --file-item-bg: #282828;
            --component-bg-color: #262730; /* Streamlit's default dark gray for some elements */
        }

        /* --- Global Streamlit Overrides --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;} /* Hide default Streamlit header bar */
        
        /* Remove default Streamlit block container padding globally */
        .block-container {
            padding: 1rem 1rem !important; /* Restore a small default padding, adjust as needed */
        }
        
        /* --- General App Structure & Theming --- */
        /* Apply background and text color to the entire app content area */
        div[data-testid="stAppViewContainer"] > .main {
            background-color: var(--background-color) !important;
            color: var(--content-text-color) !important;
            padding: 0; /* Important to remove Streamlit's default main padding for custom layout */
            margin: 0;
            flex-grow: 1; /* Allow main content to grow */
            overflow: hidden; /* Prevent main .block-container from scrolling independently */
            display: flex; /* Make it a flex container for its children (columns and chat input) */
            flex-direction: column; /* Stack children vertically */
        }
        
        /* Our custom header */
        .app-header {
            background: var(--primary-gradient);
            color: var(--header-text-color);
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-shrink: 0; /* Prevent header from shrinking */
            width: 100%; /* Ensure header spans full width */
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

        /* Target the parent container of the columns to manage its overall styling */
        .main > div > div.st-emotion-cache-z5fcl4 { /* This is often the div wrapping st.columns, might change */
            padding: 20px; /* Overall padding around the columns content */
            margin: 0;
            width: 100%;
            max-width: 100%;
            flex-grow: 1; /* Allow columns to grow and take available space */
            display: flex; /* Make it a flex container for the columns */
            gap: 20px; /* Gap between columns */
        }

        /* The individual column blocks */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { /* This targets the actual column content wrapper */
            flex-grow: 1; /* Each column grows */
            background-color: var(--content-bg) !important;
            color: var(--content-text-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 20px !important;
            box-shadow: var(--shadow-subtle);
            min-height: 400px; /* Minimum height for columns */
            display: flex; /* Make columns a flex container for their internal content */
            flex-direction: column;
        }

        /* --- Left Column - Upload Section --- */
        .file-item, .chat-session-item { /* Added chat-session-item class */
            background-color: var(--file-item-bg); /* Theme-dependent file item background */
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px 15px;
            margin: 5px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--content-text-color);
            box-shadow: var(--shadow-subtle);
            flex-wrap: wrap; /* Allow items to wrap */
        }

        .chat-session-item {
            flex-direction: column; /* Stack name and buttons vertically */
            align-items: flex-start;
            gap: 5px;
        }
        .chat-session-item .st-emotion-cache-vdgyx6 { /* Target the inner column/div of buttons */
            width: 100%;
            display: flex;
            gap: 5px;
            justify-content: flex-end; /* Align buttons to the right */
        }
        .chat-session-item .st-emotion-cache-vdgyx6 > div {
            flex: 1; /* Allow buttons to share space */
        }
        .chat-session-item button {
            margin-top: 5px; /* Add some space for buttons */
        }
        
        .file-icon {
            color: var(--st-primary-color); /* Still use Streamlit's primary color */
            font-size: 16px;
        }
        
        /* --- Right Column - Chat Section --- */
        /* Chat history container - make it scrollable */
        /* Targets the specific block that contains chat messages within the chat column */
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatMessage"]) {
            flex-grow: 1; /* Allow history to take all available space */
            overflow-y: auto; /* Enable scrolling for chat messages */
            padding-right: 10px; /* Space for scrollbar */
            margin-bottom: 10px; /* Space before the input box */
        }
        
        .chat-header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
            flex-shrink: 0; /* Prevent header from shrinking */
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            flex-shrink: 0;
        }
        
        /* Status indicator colors - keep these consistent across themes */
        .status-ready { background-color: #d4edda; color: #155724; } /* Light green */
        .status-waiting { background-color: #fff3cd; color: #856404; } /* Light yellow */
        
        /* --- General UI Element Styling --- */
        
        /* Button styling */
        .stButton > button {
            background: var(--primary-gradient);
            color: white !important; /* Force white text */
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 500;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-medium);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-strong);
        }
        
        /* Text input and textarea styling */
        .stTextArea textarea, 
        .stTextInput input,
        .stFileUploader span[data-testid="stFileUploadDropzone"],
        .stChatInput > div > div > textarea {
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color); /* Theme-dependent border */
            padding: 12px 20px;
            background-color: var(--component-bg-color); /* Use Streamlit's default component background */
            color: var(--content-text-color);
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); /* Inner shadow for depth */
        }
        /* Ensure placeholders are visible */
        .stTextArea textarea::placeholder, 
        .stTextInput input::placeholder,
        .stChatInput > div > div > textarea::placeholder {
            color: var(--st-secondary-text);
            opacity: 0.7;
        }

        /* Streamlit's `st.markdown` for regular text. Ensure it adapts */
        .stMarkdown {
            color: var(--content-text-color); /* General markdown text adapts to theme */
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: var(--content-text-color); /* Headers within markdown adapt */
        }

        /* Headers defined in app.py */
        h1, h2, h3, h4, h5, h6 { 
            color: var(--content-text-color); /* Ensure all headers adapt to theme */
            font-weight: 600;
            margin: 0 0 15px 0;
        }
        
        /* --- Chat Message Styling --- */
        div[data-testid="stChatMessage"] {
            background-color: var(--content-bg); /* Fallback, specific styles below */
            color: var(--content-text-color);
            border-radius: var(--border-radius);
            padding: 10px 15px;
            margin-bottom: 10px;
            box-shadow: var(--shadow-subtle);
        }
        
        /* Assistant message specific styling */
        div[data-testid="stChatMessage"]:nth-of-type(odd) { /* Odd children in chat history are usually assistant */
            background-color: var(--chat-bg-assistant); 
            border: 1px solid var(--border-color);
        }
        
        /* User message specific styling */
        div[data-testid="stChatMessage"]:nth-of-type(even) { /* Even children in chat history are usually user */
            background-color: var(--chat-bg-user); 
            border: 1px solid var(--border-color);
        }

        /* Ensure markdown elements inside chat messages also respect the content-text-color */
        div[data-testid="stChatMessage"] .stMarkdown {
            color: var(--content-text-color) !important;
        }

        /* --- Chat Input and Global Controls Styling --- */
        /* The st.chat_input component itself */
        div[data-testid="stChatInput"] {
            background-color: var(--content-bg) !important;
            padding: 10px 0px 0px 0px !important; /* Adjust padding as it's inside a column */
            margin: 0 !important;
            flex-shrink: 0; /* Prevent chat input from shrinking */
        }

        /* Container for clear chat/clear all data buttons */
        .bottom-buttons-container {
            padding: 10px 0px 0px 0px; /* Adjust padding as it's inside a column */
            background-color: var(--content-bg);
            display: flex;
            gap: 15px;
            flex-shrink: 0;
        }
        .bottom-buttons-container > div { /* Target the columns within this container */
            flex: 1;
        }
        
        /* Footer Styling */
        .app-footer {
            text-align: center;
            color: var(--st-secondary-text); /* Use a more subtle color for footer text */
            font-size: 12px;
            padding: 10px;
            flex-shrink: 0;
            background-color: var(--content-bg); /* Footer background adapts */
            border-top: 1px solid var(--border-color);
            width: 100%;
        }

        /* Hide streamlit branding, specific to new versions */
        /* These often appear at the bottom right. */
        .css-1rs6os, .css-17eq0hr, /* Streamlit version 1.25.0+ footer */
        div[data-testid="stDecoration"], /* Small decorative elements */
        div[data-testid="stToolbar"] { /* Toolbar that sometimes appears */
            display: none !important;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main > div > div.st-emotion-cache-z5fcl4 {
                flex-direction: column; /* Stack columns vertically on smaller screens */
                padding: 10px; /* Reduce padding on small screens */
            }
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                margin-bottom: 20px; /* Add space between stacked columns */
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global client instance from cache (moved to top-level for clarity, already @st.cache_resource)
# client = get_cached_openai_client() # This line already exists but moved here logically for new functions

# Initialize persistent ChromaDB
# @st.cache_resource ensures the client and collection are created once and reused
@st.cache_resource
def get_cached_chroma_setup():
    ch_client_instance = get_chroma_client()
    if not ch_client_instance:
        st.error("Failed to initialize database. Please check your setup.")
        st.stop()
    collection_instance = get_or_create_collection(ch_client_instance, "kb_scout_documents")
    return ch_client_instance, collection_instance
ch_client, collection = get_cached_chroma_setup()


# Session state initialization and chat session management
if "history" not in st.session_state:
    st.session_state.history = []
if "collection" not in st.session_state:
    st.session_state.collection = collection # Assign the cached collection
if "ch_client" not in st.session_state:
    st.session_state.ch_client = ch_client # Assign the cached client

# New session state variables for chat history
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = "New Chat" # Default name
if "all_chat_metas" not in st.session_state:
    st.session_state.all_chat_metas = _load_all_chat_metas()

# Initial setup or load of a chat session if not already set
if st.session_state.current_chat_id is None:
    if st.session_state.all_chat_metas:
        # Load the most recent session if available
        most_recent_id = st.session_state.all_chat_metas[0]["id"]
        name, history = _load_chat_session_by_id(most_recent_id)
        st.session_state.current_chat_id = most_recent_id
        st.session_state.current_chat_name = name
        st.session_state.history = history
    else:
        # Create a brand new, empty session
        new_id = new_uuid()
        st.session_state.current_chat_id = new_id
        st.session_state.current_chat_name = "New Chat"
        st.session_state.history = []
        _save_chat_session(new_id, "New Chat", [])
        st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh to include the newly created one


# ---- App Layout Structure ----

# Custom Header (outside main content block to span full width)
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

# Main Content Area (two columns)
# This `st.container` acts as the primary content wrapper below the header
# and takes up remaining vertical space (flex-grow: 1 on main > div.st-emotion-cache-z5fcl4)
with st.container(): # Streamlit's auto-generated div for this container will be styled by the CSS
    col1, col2 = st.columns([1, 1.2])

    # Left column - File Upload
    with col1:
        # This inner container will be styled by div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]
        with st.container(): 
            st.markdown("### Upload your files")
            st.markdown("Drag & drop or click to browse")
            
            uploaded_files = st.file_uploader(
                "",
                type=["pdf", "csv", "xlsx", "xls", "txt"], # Added txt
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            st.markdown("**Supports:** .txt, .xls, .xlsx, .csv, .pdf")
            
            if uploaded_files:
                st.markdown(f"**Selected files ({len(uploaded_files)}):**")
                for file in uploaded_files:
                    file_type_icon = {
                        "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                        "txt": "📝"
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
                                existing_files = get_uploaded_files_from_collection(st.session_state.collection)
                                file_already_exists = any(existing_file[0] == file.name for existing_file in existing_files)
                                
                                if file_already_exists:
                                    st.info(f"📁 {file.name} already in database, skipping...")
                                    continue
                                
                                file_extension = file.name.split('.')[-1].lower()
                                if file_extension == "pdf":
                                    units = read_pdf(file)
                                elif file_extension == "csv":
                                    units = read_csv(file)
                                elif file_extension in ("xlsx", "xls"):
                                    units = read_xlsx(file)
                                elif file_extension == "txt":
                                    units = read_text(file)
                                else:
                                    st.warning(f"Unsupported file type: {file_extension} for {file.name}. Skipping.")
                                    continue
                                
                                st.write(f"Extracted {len(units)} units from {file.name}")
                                
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
                                st.rerun()
                            else:
                                status.update(label="❌ Failed to process files", state="error")
                        else:
                            status.update(label="ℹ️ No new content to add", state="complete")
            
            st.markdown("---")
            st.markdown("### Uploaded Files")
            
            uploaded_files_list = get_uploaded_files_from_collection(st.session_state.collection)
            
            if uploaded_files_list:
                st.markdown(f"**{len(uploaded_files_list)} file(s) in database:**")
                for filename, filetype in uploaded_files_list:
                    file_icon = {
                        "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                        "txt": "📝"
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
        
        # New Chat Sessions Management Section
        st.markdown("---")
        st.markdown("### Chat Sessions")

        # Current chat details and rename functionality
        st.markdown(f"**Current Chat:** `{st.session_state.current_chat_name}`")
        with st.expander("Rename Current Chat"):
            new_chat_name_input = st.text_input("New chat name", value=st.session_state.current_chat_name, key="rename_chat_input")
            if st.button("Save New Name", key="save_new_name_btn"):
                _rename_current_chat(new_chat_name_input)
                st.session_state.all_chat_metas = _load_all_chat_metas() # Ensure meta list is updated
        
        if st.button("✨ New Chat Session", key="new_chat_session_btn"):
            _new_chat_session_and_refresh(client)
        
        st.markdown("---")
        st.markdown("### All Sessions")
        
        if st.session_state.all_chat_metas:
            for chat_meta in st.session_state.all_chat_metas:
                is_current = (chat_meta["id"] == st.session_state.current_chat_id)
                status_icon = "➡️" if is_current else " "
                
                # Format timestamp for display
                try:
                    dt_obj = datetime.datetime.fromisoformat(chat_meta['timestamp'])
                    formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    formatted_time = chat_meta['timestamp'].split('.')[0].replace('T', ' ') # Fallback

                st.markdown(
                    f"""
                    <div class="chat-session-item">
                        <div>{status_icon} <strong>{chat_meta['name']}</strong></div>
                        <div style='font-size:0.8em; color: var(--st-secondary-text);'>{formatted_time}</div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Use columns for buttons within the item for better layout
                col_load, col_delete = st.columns(2)
                with col_load:
                    if st.button("Load", key=f"load_{chat_meta['id']}", disabled=is_current, help="Load this chat session"):
                        _load_chat_and_refresh(chat_meta["id"])
                with col_delete:
                    if st.button("Delete", key=f"delete_{chat_meta['id']}", help="Permanently delete this chat session"):
                        _delete_chat_and_refresh(chat_meta["id"])
                
                st.markdown("</div>", unsafe_allow_html=True) # Close chat-session-item
        else:
            st.info("No saved chat sessions.")
        
    # Right column - Chat Interface
    with col2:
        # This inner container will be styled by div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]
        with st.container():
            st.markdown('<div class="chat-header">', unsafe_allow_html=True)
            st.markdown("### Chat with K&B Scout AI")
            st.markdown("Ask questions about your uploaded documents")
            
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
                except Exception as e:
                    st.warning(f"Could not retrieve database status: {e}")
                    st.markdown(
                        """
                        <div class="status-indicator status-waiting">
                            🟡 Database not ready
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True) # Close chat-header div

            # Chat history area (scrollable)
            # This container takes up available vertical space and scrolls
            with st.container(): # This will become the scrollable history area
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
            
            # Chat input and controls (fixed at bottom of the right column)
            # These are placed directly under the scrollable history container
            st.markdown('<div class="chat-input-area-in-column">', unsafe_allow_html=True)
            prompt = st.chat_input("Ask K&B Scout AI about your documents...", key="chat_input_col2") # Added key for uniqueness

            if prompt:
                # If it's a brand new chat (first message) AND it still has the default name, try to intelligently name it
                if not st.session_state.history and st.session_state.current_chat_name == "New Chat":
                    st.session_state.current_chat_name = _generate_intelligent_name(client, prompt)
                    _save_chat_session(st.session_state.current_chat_id, st.session_state.current_chat_name, []) # Save the new name and empty history
                    st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list to reflect new name

                if not st.session_state.collection:
                    st.warning("Database not initialized. Please refresh the page.")
                else:
                    try:
                        doc_count = st.session_state.collection.count()
                        if doc_count == 0:
                            st.session_state.history.append({"role": "user", "content": prompt})
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(prompt)
                            
                            with st.chat_message("assistant", avatar="🤖"):
                                response = "I'd be happy to help, but I don't have any documents to search through yet. Please upload some files first, and then I can answer questions about their content!"
                                st.markdown(response)
                                st.session_state.history.append({"role": "assistant", "content": response})
                                _save_chat_session(st.session_state.current_chat_id, st.session_state.current_chat_name, st.session_state.history) # Save chat
                        else:
                            st.session_state.history.append({"role": "user", "content": prompt})
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(prompt)

                            with st.chat_message("assistant", avatar="🤖"):
                                placeholder = st.empty()
                                
                                retrieved = retrieve(st.session_state.collection, client, prompt)
                                
                                if not retrieved:
                                    answer = "I couldn't find any relevant information in your uploaded documents for this question. Please try rephrasing your question or upload more relevant documents."
                                    placeholder.markdown(answer)
                                    st.session_state.history.append({"role": "assistant", "content": answer})
                                    _save_chat_session(st.session_state.current_chat_id, st.session_state.current_chat_name, st.session_state.history) # Save chat
                                else:
                                    context_text = format_context(retrieved)
                                    
                                    try:
                                        stream = answer_with_rag(client, prompt, context_text)
                                        answer_accum = ""
                                        for chunk in stream:
                                            delta = chunk.choices[0].delta.content or ""
                                            answer_accum += delta
                                            placeholder.markdown(answer_accum)
                                        st.session_state.history.append({"role": "assistant", "content": answer_accum})
                                        _save_chat_session(st.session_state.current_chat_id, st.session_state.current_chat_name, st.session_state.history) # Save chat
                                        st.session_state.all_chat_metas = _load_all_chat_metas() # Refresh list, as timestamp might be updated
                                    except Exception as e:
                                        st.error(f"Error generating response: {e}")
                    except Exception as e:
                        st.error(f"Error processing question: {e}")

            # Chat controls below the input box, inside the column
            st.markdown('<div class="bottom-buttons-container-in-column">', unsafe_allow_html=True)
            col_a_chat, col_b_chat = st.columns(2) # Use separate columns for buttons inside chat section
            with col_a_chat:
                if st.button("🔄 Clear Chat", key="clear_chat_col2"):
                    st.session_state.history = []
                    _save_chat_session(st.session_state.current_chat_id, st.session_state.current_chat_name, []) # Save empty history
                    st.success("Current chat history cleared.")
                    st.rerun()
            with col_b_chat:
                if st.button("🗑️ Clear All Data", key="clear_all_data_col2"):
                    if st.session_state.collection:
                        try:
                            st.session_state.ch_client.delete_collection(name="kb_scout_documents")
                            st.session_state.collection = get_or_create_collection(st.session_state.ch_client, "kb_scout_documents")
                            st.session_state.history = []
                            
                            # Delete all chat session files
                            if os.path.exists(CHAT_SESSIONS_DIR):
                                for filename in os.listdir(CHAT_SESSIONS_DIR):
                                    if filename.endswith(".json"):
                                        os.remove(os.path.join(CHAT_SESSIONS_DIR, filename))
                            
                            st.session_state.all_chat_metas = []
                            # Create a fresh, empty session after clearing everything
                            _new_chat_session_and_refresh(client) 
                            st.success("All data cleared successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing data: {e}")
            st.markdown('</div>', unsafe_allow_html=True) # Close bottom-buttons-container-in-column

            st.markdown('</div>', unsafe_allow_html=True) # Close chat-input-area-in-column
        # The inner column container from `with st.container():` ends here


# Footer (placed at the very bottom of the entire app)
st.markdown('<div class="app-footer">', unsafe_allow_html=True)
st.markdown(
    """
    💡 <strong>Tip:</strong> Upload your documents on the left, then ask questions about them on the right!<br>
    Your files are permanently stored and will be available in future sessions.
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True) # Close app-footer
