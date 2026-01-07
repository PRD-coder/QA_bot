import os
import shutil
import torch
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import traceback

# Ensure sqlite backend compatibility
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

CHROMA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore_chroma"))
GLOBAL_CHROMA = os.path.join(CHROMA_ROOT, "global")

@st.cache_resource
def get_embeddings():
    """Initializes and caches HuggingFace embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

@st.cache_resource
def get_vectorstore(session_id):
    """Loads a Chroma vectorstore for a specific session."""
    embeddings = get_embeddings()
    session_path = os.path.join(CHROMA_ROOT, session_id)
    if os.path.exists(session_path):
        try:
            return Chroma(persist_directory=session_path, embedding_function=embeddings)
        except Exception as e:
            print(f"Error loading Chroma vectorstore for session {session_id}: {e}")
            traceback.print_exc()
    return None

def _ensure_documents(chunks):
    """Convert dicts into Document objects if needed and validate metadata existence."""
    safe_chunks = []
    for c in chunks:
        if isinstance(c, Document):
            safe_chunks.append(c)
        elif isinstance(c, dict) and "page_content" in c:
            safe_chunks.append(Document(page_content=c["page_content"], metadata=c.get("metadata", {})))
        else:
            print(f"⚠️ Skipped invalid chunk: {type(c)}")
    return safe_chunks

def store_chunks(chunks, session_id, batch_size=5000):
    """Stores chunks into the session vectorstore."""
    if not chunks:
        return
    embeddings = get_embeddings()
    session_path = os.path.join(CHROMA_ROOT, session_id)
    os.makedirs(session_path, exist_ok=True)
    vectorstore = Chroma(persist_directory=session_path, embedding_function=embeddings)

    safe_chunks = _ensure_documents(chunks)  # normalize
    for i in range(0, len(safe_chunks), batch_size):
        batch = safe_chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
    # clear cached resources to pick up newly persisted data
    try:
        st.cache_resource.clear()
    except Exception:
        pass

# ---------------- Global Vectorstore ---------------- #
@st.cache_resource
def get_global_vectorstore():
    """Loads global vectorstore (all docs)."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=GLOBAL_CHROMA, embedding_function=embeddings)

def store_chunks_global(chunks, batch_size=5000):
    """Stores chunks into global vectorstore (shared across sessions)."""
    if not chunks:
        return
    embeddings = get_embeddings()
    os.makedirs(GLOBAL_CHROMA, exist_ok=True)
    vectorstore = Chroma(persist_directory=GLOBAL_CHROMA, embedding_function=embeddings)

    safe_chunks = _ensure_documents(chunks)
    for i in range(0, len(safe_chunks), batch_size):
        batch = safe_chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
    try:
        st.cache_resource.clear()
    except Exception:
        pass

def delete_vectorstore_for_session(session_id):
    """Deletes session vectorstore (global persists)."""
    session_path = os.path.join(CHROMA_ROOT, session_id)
    if os.path.exists(session_path):
        shutil.rmtree(session_path)
        try:
            st.cache_resource.clear()
        except Exception:
            pass
