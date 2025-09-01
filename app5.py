import sys
import os
import uuid
from typing import List, Dict
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from pypdf import PdfReader
import chromadb
from openai import OpenAI

# Simple configuration - environment variables only
def get_openai_client():
    """Get OpenAI client using environment variable only."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("🚨 **OpenAI API Key Required**")
        st.markdown("""
        ### Set your API key as an environment variable:
        
        **For Railway:**
        - Go to Variables tab
        - Add: `OPENAI_API_KEY = your-key-here`
        
        **For Streamlit Cloud:**
        - Go to App Settings → Secrets
        - Add: `OPENAI_API_KEY = "your-key-here"`
        
        **For Local Development:**
        - Create `.env` file with: `OPENAI_API_KEY=your-key-here`
        """)
        st.stop()
    
    try:
        client = OpenAI(api_key=api_key.strip())
        # Test the connection
        client.embeddings.create(input=["test"], model="text-embedding-3-small")
        return client
    except Exception as e:
        st.error(f"❌ **OpenAI API Error:** {str(e)}")
        st.error("**Common Solutions:**")
        st.error("- Generate a new API key at https://platform.openai.com/api-keys")
        st.error("- Ensure your OpenAI account has credits")
        st.error("- Check that your payment method is valid")
        st.stop()

@dataclass
class Document:
    id: str
    text: str
    source: str
    page: int = 1

def chunk_text_simple(text: str, max_chars: int = 2000) -> List[str]:
    """Simple text chunking."""
    if not text.strip():
        return []
    
    # Split by sentences and group them
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 50]  # Filter very short chunks

def process_pdf_file(uploaded_file) -> List[Document]:
    """Process uploaded PDF file."""
    documents = []
    try:
        reader = PdfReader(uploaded_file)
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                chunks = chunk_text_simple(text)
                for chunk in chunks:
                    doc = Document(
                        id=str(uuid.uuid4()),
                        text=chunk,
                        source=uploaded_file.name,
                        page=page_num
                    )
                    documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing PDF {uploaded_file.name}: {e}")
        return []

def process_csv_file(uploaded_file) -> List[Document]:
    """Process uploaded CSV file."""
    documents = []
    try:
        df = pd.read_csv(uploaded_file)
        
        for idx, row in df.iterrows():
            row_text_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    row_text_parts.append(f"{col}: {val}")
            
            if row_text_parts:
                text = " | ".join(row_text_parts)
                doc = Document(
                    id=str(uuid.uuid4()),
                    text=text,
                    source=uploaded_file.name,
                    page=idx + 1
                )
                documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing CSV {uploaded_file.name}: {e}")
        return []

def process_text_file(uploaded_file) -> List[Document]:
    """Process uploaded text file."""
    documents = []
    try:
        content = str(uploaded_file.read(), 'utf-8')
        chunks = chunk_text_simple(content)
        
        for i, chunk in enumerate(chunks, 1):
            doc = Document(
                id=str(uuid.uuid4()),
                text=chunk,
                source=uploaded_file.name,
                page=i
            )
            documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error processing text file {uploaded_file.name}: {e}")
        return []

# Vector Database Functions
@st.cache_resource
def init_vector_db():
    """Initialize ChromaDB."""
    try:
        client = chromadb.PersistentClient(path="./vector_db")
        collection = client.get_or_create_collection(name="kb_documents")
        return collection
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def add_docs_to_db(documents: List[Document], openai_client: OpenAI) -> bool:
    """Add documents to vector database."""
    if not documents:
        return False
    
    collection = init_vector_db()
    if not collection:
        return False
    
    try:
        # Create embeddings in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            texts = [doc.text for doc in batch_docs]
            
            # Get embeddings
            response = openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            embeddings = [data.embedding for data in response.data]
            
            # Add to collection
            collection.add(
                ids=[doc.id for doc in batch_docs],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"source": doc.source, "page": doc.page} for doc in batch_docs]
            )
        
        return True
    except Exception as e:
        st.error(f"Error adding documents: {e}")
        return False

def search_docs(query: str, openai_client: OpenAI, num_results: int = 5):
    """Search documents."""
    collection = init_vector_db()
    if not collection:
        return []
    
    try:
        # Get query embedding
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(num_results, collection.count()),
            include=["documents", "metadatas"]
        )
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def get_doc_count() -> int:
    """Get document count."""
    collection = init_vector_db()
    if collection:
        try:
            return collection.count()
        except:
            return 0
    return 0

def generate_ai_response(question: str, context: str, openai_client: OpenAI) -> str:
    """Generate AI response."""
    prompt = f"""Based on the following context from uploaded documents, answer the question.
If the answer is not in the context, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App
st.set_page_config(
    page_title="K&B Scout AI", 
    page_icon="🤖", 
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .doc-counter {
        background: #e8f5e8;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: #2d5a2d;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 K&B Scout AI</h1>
    <h3>Enterprise Document Assistant</h3>
    <p>Upload your documents and chat with them using AI</p>
</div>
""", unsafe_allow_html=True)

# Initialize OpenAI
openai_client = get_openai_client()
st.success("✅ OpenAI API connected successfully")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
col1, col2 = st.columns([1, 2])

# Upload Section
with col1:
    st.header("📤 Upload Documents")
    
    # Show document count
    doc_count = get_doc_count()
    st.markdown(f'<div class="doc-counter">📊 Documents in Database: {doc_count}</div>', unsafe_allow_html=True)
    
    # File uploader
    files = st.file_uploader(
        "Select files to upload",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, CSV, or TXT files"
    )
    
    if files:
        st.info(f"✅ {len(files)} file(s) selected")
        
        if st.button("🚀 Process Files", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_documents = []
            
            for i, file in enumerate(files):
                status_text.text(f"Processing: {file.name}")
                progress_bar.progress((i + 1) / len(files))
                
                if file.name.lower().endswith('.pdf'):
                    docs = process_pdf_file(file)
                elif file.name.lower().endswith('.csv'):
                    docs = process_csv_file(file)
                elif file.name.lower().endswith('.txt'):
                    docs = process_text_file(file)
                else:
                    continue
                
                all_documents.extend(docs)
                st.write(f"✅ {file.name}: {len(docs)} chunks extracted")
            
            if all_documents:
                status_text.text("Adding to vector database...")
                
                if add_docs_to_db(all_documents, openai_client):
                    st.success(f"🎉 Successfully processed {len(all_documents)} document chunks!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to add documents to database")
            else:
                st.warning("⚠️ No content could be extracted from the files")
            
            progress_bar.empty()
            status_text.empty()

# Chat Section
with col2:
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if doc_count == 0:
                response = "I don't have any documents to reference. Please upload some documents first!"
            else:
                with st.spinner("Searching documents..."):
                    search_results = search_docs(prompt, openai_client)
                    
                    if search_results and search_results.get('documents') and search_results['documents'][0]:
                        # Build context
                        docs = search_results['documents'][0]
                        metas = search_results['metadatas'][0]
                        
                        context_parts = []
                        for doc, meta in zip(docs, metas):
                            source = meta['source']
                            page = meta['page']
                            context_parts.append(f"From {source} (page {page}):\n{doc}")
                        
                        context = "\n\n".join(context_parts)
                        response = generate_ai_response(prompt, context, openai_client)
                    else:
                        response = "I couldn't find relevant information in your documents to answer this question."
            
            st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📁 Supported Files")
    st.markdown("- **PDF**: Text documents")
    st.markdown("- **CSV**: Spreadsheet data")  
    st.markdown("- **TXT**: Plain text files")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("1. Upload your documents first")
    st.markdown("2. Wait for processing to complete")
    st.markdown("3. Ask specific questions")
    st.markdown("4. Reference document names for clarity")

st.markdown("---")
st.markdown("*🤖 Powered by OpenAI GPT-4 and ChromaDB*", unsafe_allow_html=True)
