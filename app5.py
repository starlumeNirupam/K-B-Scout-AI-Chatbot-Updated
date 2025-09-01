import sys
import os
import uuid
from typing import List, Dict
from dataclasses import dataclass

# SQLite compatibility fix
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import pandas as pd
from pypdf import PdfReader
import chromadb
from openai import OpenAI

# Load environment variables for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -----------------------------
# Configuration
# -----------------------------

def get_openai_api_key():
    """Get OpenAI API key from environment or Streamlit secrets."""
    # Try Streamlit secrets first
    try:
        key = st.secrets["OPENAI_API_KEY"]
        return key.strip()
    except:
        pass
    
    # Try environment variable
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    
    return None

def create_openai_client():
    """Create and validate OpenAI client."""
    api_key = get_openai_api_key()
    
    if not api_key:
        st.error("🚨 **OpenAI API Key Missing**")
        st.markdown("""
        **For Railway:** Add environment variable `OPENAI_API_KEY`
        
        **For Streamlit Cloud:** Add to secrets: `OPENAI_API_KEY = "your-key"`
        """)
        st.stop()
    
    if not api_key.startswith(('sk-', 'sk-proj-')):
        st.error("❌ Invalid API key format")
        st.stop()
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the client
        client.embeddings.create(input=["test"], model="text-embedding-3-small")
        return client
    except Exception as e:
        st.error(f"❌ **OpenAI Error:** {str(e)[:100]}...")
        st.error("Try generating a new API key at https://platform.openai.com/api-keys")
        st.stop()

# -----------------------------
# Document Processing
# -----------------------------

@dataclass
class Document:
    id: str
    text: str
    source: str
    page: int = 1

def simple_chunk_text(text: str, max_length: int = 3000) -> List[str]:
    """Simple text chunking by sentences."""
    if not text.strip():
        return []
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + sentence + ". "
        if len(test_chunk) < max_length:
            current_chunk = test_chunk
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def process_pdf(file) -> List[Document]:
    """Process PDF file."""
    try:
        reader = PdfReader(file)
        documents = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                chunks = simple_chunk_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        doc = Document(
                            id=str(uuid.uuid4()),
                            text=chunk,
                            source=file.name,
                            page=page_num
                        )
                        documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)[:100]}...")
        return []

def process_csv(file) -> List[Document]:
    """Process CSV file."""
    try:
        df = pd.read_csv(file)
        documents = []
        
        for idx, row in df.iterrows():
            row_data = []
            for col, val in row.items():
                if pd.notna(val):
                    row_data.append(f"{col}: {val}")
            
            if row_data:
                text = " | ".join(row_data)
                doc = Document(
                    id=str(uuid.uuid4()),
                    text=text,
                    source=file.name,
                    page=idx + 1
                )
                documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)[:100]}...")
        return []

def process_text_file(file) -> List[Document]:
    """Process text file."""
    try:
        content = str(file.read(), 'utf-8')
        chunks = simple_chunk_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    id=str(uuid.uuid4()),
                    text=chunk,
                    source=file.name,
                    page=i + 1
                )
                documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing text file: {str(e)[:100]}...")
        return []

# -----------------------------
# Vector Database
# -----------------------------

@st.cache_resource
def get_vector_db():
    """Initialize ChromaDB."""
    try:
        client = chromadb.PersistentClient(path="./chroma_data")
        collection = client.get_or_create_collection(name="documents")
        return collection
    except Exception as e:
        st.error(f"Database initialization error: {str(e)[:100]}...")
        return None

def add_documents_to_db(documents: List[Document], openai_client: OpenAI):
    """Add documents to vector database."""
    if not documents:
        return False
    
    collection = get_vector_db()
    if not collection:
        return False
    
    try:
        # Create embeddings
        texts = [doc.text for doc in documents]
        embeddings_response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in embeddings_response.data]
        
        # Add to collection
        collection.add(
            ids=[doc.id for doc in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": doc.source, "page": doc.page} for doc in documents]
        )
        
        return True
    except Exception as e:
        st.error(f"Error adding to database: {str(e)[:100]}...")
        return False

def search_similar_documents(query: str, openai_client: OpenAI, limit: int = 5):
    """Search for similar documents."""
    collection = get_vector_db()
    if not collection:
        return []
    
    try:
        # Create query embedding
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas"]
        )
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)[:100]}...")
        return []

def get_total_documents():
    """Get total document count."""
    collection = get_vector_db()
    if collection:
        try:
            return collection.count()
        except:
            return 0
    return 0

# -----------------------------
# AI Chat
# -----------------------------

def generate_response(question: str, context: str, openai_client: OpenAI) -> str:
    """Generate AI response."""
    system_prompt = """You are a helpful assistant that answers questions based on provided context.
If the answer isn't in the context, say you don't have enough information."""
    
    user_prompt = f"""Context: {context}

Question: {question}

Please provide a helpful answer based on the context above."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)[:100]}..."

# -----------------------------
# Streamlit App
# -----------------------------

# Page configuration
st.set_page_config(
    page_title="K&B Scout AI",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .upload-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1>🤖 K&B Scout AI</h1>
    <p>Enterprise Document Assistant - Upload, Search, and Chat with your documents</p>
</div>
""", unsafe_allow_html=True)

# Initialize OpenAI client
openai_client = create_openai_client()
st.success("✅ OpenAI connection established")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main layout
left_col, right_col = st.columns([1, 2])

# Left column - File upload
with left_col:
    st.markdown("### 📁 Document Upload")
    
    # Document count display
    doc_count = get_total_documents()
    st.markdown(f'<span class="status-badge status-success">📊 Total Documents: {doc_count}</span>', unsafe_allow_html=True)
    
    # File uploader (with explicit label)
    uploaded_files = st.file_uploader(
        label="Choose files to upload",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        key="document_uploader",
        help="Select PDF, CSV, or TXT files to upload"
    )
    
    # Process files button
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s)")
        
        if st.button("🚀 Process Documents", key="process_button"):
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0, text="Starting processing...")
                
                all_docs = []
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress(
                        (i + 1) / len(uploaded_files), 
                        text=f"Processing {file.name}..."
                    )
                    
                    if file.name.lower().endswith('.pdf'):
                        docs = process_pdf(file)
                    elif file.name.lower().endswith('.csv'):
                        docs = process_csv(file)
                    elif file.name.lower().endswith('.txt'):
                        docs = process_text_file(file)
                    else:
                        continue
                    
                    all_docs.extend(docs)
                
                if all_docs:
                    progress_bar.progress(1.0, text="Adding to database...")
                    
                    if add_documents_to_db(all_docs, openai_client):
                        st.success(f"✅ Successfully processed {len(all_docs)} document chunks!")
                        st.balloons()
                        # Rerun to update document count
                        st.rerun()
                    else:
                        st.error("❌ Failed to add documents to database")
                else:
                    st.warning("⚠️ No content extracted from uploaded files")
                
                progress_bar.empty()

# Right column - Chat interface  
with right_col:
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with explicit key
    user_input = st.chat_input(
        placeholder="Ask me anything about your documents...",
        key="chat_input_main"
    )
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if doc_count == 0:
                    response = "I don't have any documents to reference yet. Please upload some documents first!"
                else:
                    # Search for relevant documents
                    search_results = search_similar_documents(user_input, openai_client)
                    
                    if search_results and search_results.get('documents') and search_results['documents'][0]:
                        # Build context from search results
                        context_parts = []
                        documents = search_results['documents'][0]
                        metadatas = search_results['metadatas'][0]
                        
                        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                            source = meta.get('source', 'Unknown')
                            page = meta.get('page', 1)
                            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc}")
                        
                        context = "\n\n".join(context_parts)
                        response = generate_response(user_input, context, openai_client)
                    else:
                        response = "I couldn't find relevant information in your documents for this question."
            
            st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Sidebar with controls
with st.sidebar:
    st.markdown("### 🔧 Controls")
    
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📋 Supported Files")
    st.markdown("- 📄 **PDF** - Text documents")
    st.markdown("- 📊 **CSV** - Data tables") 
    st.markdown("- 📝 **TXT** - Plain text files")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ How to Use")
    st.markdown("1. **Upload** your documents")
    st.markdown("2. **Wait** for processing")
    st.markdown("3. **Ask** questions in natural language")
    st.markdown("4. **Get** answers based on your documents")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #6c757d; font-size: 14px;">🤖 K&B Scout AI - Powered by OpenAI & ChromaDB</div>',
    unsafe_allow_html=True
)
