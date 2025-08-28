import streamlit as st
import ollama
import chromadb
import numpy as np
import pdfplumber
import pandas as pd
import uuid
from typing import List, Dict

# ---- Stylish Streamlit CSS ----
st.markdown("""
<style>
body, .main {
    background: linear-gradient(135deg, #ede7f6 0%, #c8e6c9 100%) !important;
    color: #232323 !important;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #512da8 0%, #00897b 100%);
    color: #fff !important;
}
[data-testid="stSidebar"] .css-1v0mbdj {
    color: #fff !important;
    font-weight: 600;
}
h1, h2, h3 {
    color: #512da8 !important;
    font-family: 'Montserrat', 'Segoe UI', sans-serif;
}
.stButton>button {
    background: linear-gradient(90deg, #00897b 0%, #512da8 100%);
    color: #fff !important;
    border-radius: 14px;
    font-weight: bold;
    padding: 0.7em 1.5em;
    border: none;
    box-shadow: 0 4px 12px #00897b44;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #512da8 0%, #00897b 100%);
    color: #fff !important;
    transform: translateY(-2px) scale(1.03);
}
.stTextArea textarea, .stTextInput input {
    border-radius: 15px;
    background: #f3e5f5 !important;
    color: #222 !important;
    box-shadow: 0 2px 8px #512da811;
    font-size: 1.05em;
}
.block-container {
    background: rgba(255,255,255,0.88) !important;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 6px 24px #512da818;
}
</style>
""", unsafe_allow_html=True)

# ---- ChromaDB Setup ----
@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB client and collection"""
    client = chromadb.Client()
    try:
        # Try to get existing collection, create if not exists
        collection = client.get_or_create_collection(
            name="legal_documents",
            metadata={"description": "Legal documents vector database"}
        )
    except Exception:
        collection = client.create_collection(
            name="legal_documents",
            metadata={"description": "Legal documents vector database"}
        )
    return client, collection

# ---- Embedding Utilities ----
def get_embedding(text: str, model: str = 'mistral') -> List[float]:
    """Get embedding for text using Ollama"""
    try:
        resp = ollama.embeddings(model=model, prompt=text)
        return resp['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []

# ---- Document Processing Functions ----
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_csv(file) -> str:
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(file)
        # Convert DataFrame to readable text format
        text_content = f"CSV Data Summary:\n"
        text_content += f"Columns: {', '.join(df.columns.tolist())}\n"
        text_content += f"Total Rows: {len(df)}\n\n"
        
        # Add column descriptions
        for col in df.columns:
            text_content += f"Column '{col}':\n"
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()[:10]  # First 10 unique values
                text_content += f"  Sample values: {', '.join(map(str, unique_vals))}\n"
            else:
                text_content += f"  Data type: {df[col].dtype}\n"
                text_content += f"  Range: {df[col].min()} to {df[col].max()}\n"
            text_content += "\n"
        
        # Add sample rows
        text_content += "Sample Data:\n"
        text_content += df.head(10).to_string(index=False)
        
        return text_content
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return ""

def extract_text_from_excel(file) -> str:
    """Extract text from Excel file"""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file)
        text_content = f"Excel File with {len(excel_file.sheet_names)} sheet(s):\n\n"
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            text_content += f"Sheet: {sheet_name}\n"
            text_content += f"Columns: {', '.join(df.columns.tolist())}\n"
            text_content += f"Rows: {len(df)}\n\n"
            
            # Add column info
            for col in df.columns:
                text_content += f"Column '{col}':\n"
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()[:5]
                    text_content += f"  Sample values: {', '.join(map(str, unique_vals))}\n"
                else:
                    text_content += f"  Data type: {df[col].dtype}\n"
                    if not df[col].dropna().empty:
                        text_content += f"  Range: {df[col].min()} to {df[col].max()}\n"
                text_content += "\n"
            
            # Add sample data
            text_content += "Sample Data:\n"
            text_content += df.head(5).to_string(index=False)
            text_content += "\n" + "="*50 + "\n\n"
        
        return text_content
    except Exception as e:
        st.error(f"Error reading Excel: {str(e)}")
        return ""

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Split text into chunks"""
    if not text.strip():
        return []
    
    # Split by double newlines first (paragraphs)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks
    return [chunk for chunk in chunks if len(chunk) > 50]

def store_in_chromadb(collection, text_chunks: List[str], filename: str):
    """Store text chunks in ChromaDB"""
    try:
        # Clear existing documents for this file
        existing_docs = collection.get(where={"filename": filename})
        if existing_docs['ids']:
            collection.delete(ids=existing_docs['ids'])
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in text_chunks]
        embeddings = []
        metadatas = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk)
            if embedding:  # Only add if embedding is successful
                embeddings.append(embedding)
                metadatas.append({
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_length": len(chunk)
                })
            progress_bar.progress((i + 1) / len(text_chunks))
        
        if embeddings:
            # Add to ChromaDB
            collection.add(
                ids=ids[:len(embeddings)],
                embeddings=embeddings,
                documents=text_chunks[:len(embeddings)],
                metadatas=metadatas
            )
            return len(embeddings)
        else:
            st.error("No embeddings could be created")
            return 0
            
    except Exception as e:
        st.error(f"Error storing in ChromaDB: {str(e)}")
        return 0

def search_chromadb(collection, query: str, n_results: int = 3) -> List[str]:
    """Search ChromaDB for relevant chunks"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        st.error(f"Error searching ChromaDB: {str(e)}")
        return []

def ollama_chat(prompt: str, model: str = 'mistral') -> str:
    """Generate response using Ollama"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error with Ollama chat: {str(e)}")
        return "Sorry, I couldn't generate a response."

# ---- Main Streamlit App ----
st.set_page_config(page_title="Local Legal Research AI", layout="wide")
st.title("⚖️ Local Legal Research AI with ChromaDB")

# Initialize ChromaDB
try:
    client, collection = init_chromadb()
    st.success("ChromaDB initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {str(e)}")
    st.stop()

# File Upload Section
st.header("📁 Upload Your Document")
st.markdown("*Supported formats: PDF, CSV, Excel (.xlsx)*")

uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["pdf", "csv", "xlsx"],
    help="Upload a PDF, CSV, or Excel file to create a searchable knowledge base"
)

if uploaded_file is not None:
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": uploaded_file.size
    }
    st.write("**File Details:**")
    st.json(file_details)
    
    if st.button("🚀 Process and Store Document", type="primary"):
        with st.spinner("Processing document..."):
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/csv":
                text = extract_text_from_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                text = extract_text_from_excel(uploaded_file)
            else:
                st.error("Unsupported file type!")
                st.stop()
            
            if not text.strip():
                st.error("No text could be extracted from the file!")
                st.stop()
            
            # Chunk the text
            st.info("Chunking document...")
            chunks = chunk_text(text)
            
            if not chunks:
                st.error("No valid chunks created from the document!")
                st.stop()
            
            # Store in ChromaDB
            st.info("Storing in vector database...")
            stored_count = store_in_chromadb(collection, chunks, uploaded_file.name)
            
            if stored_count > 0:
                st.success(f"✅ Document processed successfully! Stored {stored_count} chunks in ChromaDB.")
            else:
                st.error("Failed to store document in database!")

# Query Section
st.header("❓ Ask Questions About Your Documents")

# Show stored documents
try:
    all_docs = collection.get()
    if all_docs['metadatas']:
        filenames = list(set([meta.get('filename', 'Unknown') for meta in all_docs['metadatas']]))
        st.info(f"📚 Documents in database: {', '.join(filenames)}")
    else:
        st.warning("No documents stored yet. Please upload a document first.")
except Exception as e:
    st.warning("Could not retrieve document list.")

question = st.text_input("🔍 Enter your question:", placeholder="Ask anything about your uploaded documents...")

if st.button("🧠 Get Answer", type="primary") and question.strip():
    with st.spinner("Searching for relevant information..."):
        # Search ChromaDB
        relevant_chunks = search_chromadb(collection, question, n_results=5)
        
        if not relevant_chunks:
            st.warning("No relevant information found. Please try rephrasing your question.")
        else:
            # Create context from relevant chunks
            context = "\n\n".join(relevant_chunks)
            
            # Generate AI response
            with st.spinner("Generating answer..."):
                prompt = f"""Based on the following document excerpts, provide a clear, accurate, and concise answer to the question. If the information is not sufficient to answer the question completely, mention that.

Context from documents:
{context}

Question: {question}

Please provide a direct and helpful answer based on the context provided."""

                answer = ollama_chat(prompt)
                
                # Display results
                st.markdown("### 🎯 Answer:")
                st.success(answer)
                
                # Show relevant context in expander
                with st.expander("📄 View Source Context"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Context {i}:**")
                        st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#512da8;'>"
    "Built with ❤️ using Ollama, ChromaDB, and Streamlit"
    "</div>", 
    unsafe_allow_html=True
)