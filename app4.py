import os
import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass

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

# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_openai_client(api_key: str = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("No OpenAI API key found. Please enter it in the sidebar or set OPENAI_API_KEY env var.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key
    return OpenAI()

def new_uuid() -> str:
    return str(uuid.uuid4())

def make_tokenizer():
    # cl100k_base works well for GPT-4/4o family & text-embedding-3-*
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
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        if end == len(tokens):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks

def read_pdf(file) -> List[Tuple[str, Dict]]:
    """
    Returns list of (text, metadata) for each page of the PDF.
    """
    try:
        reader = PdfReader(file)
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = txt.strip()
            st.write(f"Extracted text from page {i+1}: {txt}")
            if txt:  # Only add pages with content
                pages.append((txt, {"source": file.name, "type": "pdf", "page": i + 1}))
        return pages
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return []

def read_csv(file) -> List[Tuple[str, Dict]]:
    """
    Returns (row_text, metadata) per row; joins columns with ' | '.
    """
    try:
        df = pd.read_csv(file)
        rows = []
        for idx, row in df.iterrows():
            # Handle NaN values
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:  # Only add rows with content
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
            # Handle NaN values
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:  # Only add rows with content
                row_text = " | ".join(row_values)
                rows.append((row_text, {"source": file.name, "type": "xlsx", "row": int(idx) + 1}))
        return rows
    except Exception as e:
        st.error(f"Error reading Excel {file.name}: {e}")
        return []

def safe_clean(s: str) -> str:
    if not s:
        return ""
    # Remove null bytes and other problematic characters
    cleaned = s.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    # Remove extra whitespace
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

def get_chroma_client(persist: bool, persist_dir: str):
    """
    Creates a ChromaDB client. Uses PersistentClient if available, otherwise falls back.
    """
    if persist:
        try:
            # Create directory if it doesn't exist
            os.makedirs(persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            st.warning(f"Could not create persistent client: {e}. Using in-memory client.")
            client = chromadb.Client(Settings(anonymized_telemetry=False))
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))
    return client

def create_or_load_collection(chroma_client, collection_name: str):
    try:
        # Try to delete existing collection to avoid conflicts
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist
        
        return chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        return None

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[List[float]]:
    """
    Batches embeddings to avoid hitting request-size limits.
    """
    if not texts:
        return []
    
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        st.write(f"Processing embedding batch {batch_num}/{total_batches}...")
        
        batch = texts[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"Error creating embeddings for batch {batch_num}: {e}")
            return []
    
    return all_embeddings

def add_chunks_to_collection(
    collection,
    client: OpenAI,
    rag_chunks: List[RAGChunk],
    embedding_model: str
):
    if not rag_chunks or not collection:
        st.warning("No chunks to add or collection is None")
        return False
    
    st.write(f"Adding {len(rag_chunks)} chunks to collection...")
    
    # Filter out empty chunks
    valid_chunks = [c for c in rag_chunks if c.text and c.text.strip()]
    if not valid_chunks:
        st.warning("No valid chunks found after filtering")
        return False
    
    st.write(f"Valid chunks after filtering: {len(valid_chunks)}")
    
    documents = [c.text for c in valid_chunks]
    metadatas = [c.metadata for c in valid_chunks]
    ids = [c.id for c in valid_chunks]

    # Create embeddings
    embeddings = embed_texts(client, documents, model=embedding_model)
    
    if not embeddings:
        st.error("Failed to create embeddings")
        return False
    
    if len(embeddings) != len(documents):
        st.error(f"Mismatch: {len(embeddings)} embeddings for {len(documents)} documents")
        return False

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        st.success(f"Successfully added {len(valid_chunks)} chunks to vector database!")
        
        # Verify the collection has data
        count = collection.count()
        st.info(f"Collection now contains {count} documents")
        return True
        
    except Exception as e:
        st.error(f"Error adding to collection: {e}")
        return False

def retrieve(
    collection,
    client: OpenAI,
    query: str,
    embedding_model: str,
    top_k: int = 6
) -> List[Tuple[str, Dict, float]]:
    if not collection:
        return []
    
    # Check if collection has data
    count = collection.count()
    if count == 0:
        st.warning("Collection is empty. Please ingest files first.")
        return []
    
    try:
        q_emb = embed_texts(client, [query], model=embedding_model)[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),  # Don't request more than available
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        
        # Convert Chroma distance to similarity-like score (lower distance is better for cosine)
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

SYSTEM_PROMPT = """You are a highly accurate assistant for answering questions using the provided context.
Follow these rules:
- Use only the information in <context> ... </context>.
- If the answer cannot be found in the context, say you do not have enough information.
- Be concise and cite sources as [#] using the bracket numbers that appear in the context.
"""

def answer_with_rag(
    client: OpenAI,
    model: str,
    question: str,
    context_text: str,
    temperature: float = 0.0,
    stream: bool = False
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<context>\n{context_text}\n</context>\n\nQuestion: {question}\nAnswer:"
        }
    ]

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=stream
    )

# -----------------------------
# --------- UI Layer ----------
# -----------------------------

st.set_page_config(page_title="RAG over your files (PDF/CSV/XLSX) • ChromaDB + OpenAI", page_icon="🧠", layout="wide")

st.title("🧠 K&B Scout AI Enterprise Edition")

with st.sidebar:
    st.subheader("⚙️ Settings")

    # API key input (optional if set via env or secrets)
    api_key_input = st.text_input(
        "OpenAI API Key (optional if set in env or secrets)",
        type="password",
        placeholder="sk-...",
        help="You can also set OPENAI_API_KEY as an environment variable or in Streamlit secrets."
    )

    llm_model = st.selectbox(
        "LLM model",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        index=0,
        help="Use a lightweight model for speed/cost or a bigger one for best quality."
    )

    embed_model = st.selectbox(
        "Embedding model",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        index=0,
        help="Small is cheaper and usually sufficient. Large is most accurate."
    )

    top_k = st.slider("Top-K passages to retrieve", min_value=2, max_value=15, value=6)
    temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.0, 0.1)

    st.markdown("---")
    st.caption("Index Storage")
    persist_enable = st.toggle("Persist index to disk (local)", value=False)
    persist_dir = st.text_input("Persist directory (if enabled)", value=".rag_chromadb")

    st.markdown("---")
    st.caption("Chunking")
    chunk_tokens = st.slider("Chunk size (tokens)", 200, 1500, 800, 25)
    overlap_tokens = st.slider("Overlap (tokens)", 0, 400, 150, 10)

# Initialize OpenAI
client = get_openai_client(api_key_input)

# Session state
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"collection_{new_uuid()}"
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(role, content)]
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "collection" not in st.session_state:
    st.session_state.collection = None

# Chroma client + collection
ch_client = get_chroma_client(persist=persist_enable, persist_dir=persist_dir)

# Upload
st.subheader("1) Upload files")
uploaded_files = st.file_uploader(
    "Drop in PDFs, CSVs, or Excel (XLSX). Multiple files allowed.",
    type=["pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

ingest_btn = st.button("▶️ Ingest & Index", type="primary", disabled=not uploaded_files)

if ingest_btn:
    st.session_state.collection = create_or_load_collection(ch_client, st.session_state.collection_name)
    
    if st.session_state.collection is None:
        st.error("Failed to create collection")
        st.stop()
    
    tokenizer = make_tokenizer()
    rag_chunks: List[RAGChunk] = []
    
    with st.status("Indexing your files…", expanded=True) as status:
        total_files = len(uploaded_files)
        
        for file_idx, file in enumerate(uploaded_files, 1):
            st.write(f"Reading **{file.name}** ({file_idx}/{total_files})...")
            
            try:
                if file.name.lower().endswith(".pdf"):
                    units = read_pdf(file)
                elif file.name.lower().endswith(".csv"):
                    units = read_csv(file)
                else:
                    units = read_xlsx(file)
                
                st.write(f"Extracted {len(units)} units from {file.name}")
                
            except Exception as e:
                st.error(f"Failed to read {file.name}: {e}")
                continue

            # Chunk each unit
            for unit_idx, (unit_text, meta) in enumerate(units):
                unit_text = safe_clean(unit_text)
                if not unit_text:
                    continue
                
                chunks = chunk_text(unit_text, tokenizer, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
                st.write(f"Created {len(chunks)} chunks from unit {unit_idx + 1}")
                
                for chunk_idx, ch in enumerate(chunks):
                    if ch.strip():  # Only add non-empty chunks
                        # Add chunk index to metadata for better tracking
                        chunk_meta = meta.copy()
                        chunk_meta["chunk_id"] = chunk_idx + 1
                        rag_chunks.append(RAGChunk(id=new_uuid(), text=ch, metadata=chunk_meta))

        st.write(f"Total chunks created: {len(rag_chunks)}")
        
        if not rag_chunks:
            st.warning("No text content found to index. Are your PDFs scanned images? (OCR not included).")
            status.update(label="❌ No content found", state="error")
        else:
            st.write(f"Creating embeddings and adding to vector database...")
            success = add_chunks_to_collection(
                st.session_state.collection, 
                client, 
                rag_chunks, 
                embedding_model=embed_model
            )
            
            if success:
                st.session_state.indexed = True
                status.update(label="✅ Index built successfully", state="complete")
            else:
                status.update(label="❌ Failed to build index", state="error")

# Divider
st.divider()

st.subheader("2) Ask questions about your files")

# Show collection status
if st.session_state.collection and st.session_state.indexed:
    try:
        count = st.session_state.collection.count()
        st.info(f"📊 Vector database contains {count} document chunks ready for search")
    except:
        st.warning("Could not check collection status")

# Chat UI
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question (the model will cite [#] sources)")
if prompt:
    if not st.session_state.indexed or not st.session_state.collection:
        st.warning("Please ingest files first!")
    else:
        # Show user message
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            # Retrieve
            try:
                retrieved = retrieve(
                    st.session_state.collection, 
                    client, 
                    prompt, 
                    embedding_model=embed_model, 
                    top_k=top_k
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

            if not retrieved:
                answer = "_I couldn't retrieve any relevant context from your files._"
                placeholder.write(answer)
                st.session_state.history.append({"role": "assistant", "content": answer})
            else:
                context_text = format_context(retrieved)
                # Stream answer
                try:
                    stream = answer_with_rag(
                        client=client,
                        model=llm_model,
                        question=prompt,
                        context_text=context_text,
                        temperature=temperature,
                        stream=True
                    )
                    answer_accum = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        answer_accum += delta
                        placeholder.markdown(answer_accum)
                    st.session_state.history.append({"role": "assistant", "content": answer_accum})
                except Exception as e:
                    st.error(f"Generation error: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset chat"):
    st.session_state.history = []
    st.rerun()

if st.sidebar.button("🗃️ New empty collection"):
    st.session_state.collection_name = f"collection_{new_uuid()}"
    st.session_state.indexed = False
    st.session_state.collection = None
    st.success("Created a new, empty collection. Ingest files to use it.")
    st.rerun()
