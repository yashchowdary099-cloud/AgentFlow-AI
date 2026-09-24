"""
Document Processing & Vector Retrieval Service
Handles PDF text extraction, chunking, embedding generation, and semantic search.
"""

import os
import re
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import pymupdf  # Modern PyMuPDF API
from backend.config import UPLOADS_DIR, VECTOR_STORE_DIR, GEMINI_API_KEY, EMBEDDING_MODEL, is_gemini_configured

logger = logging.getLogger(__name__)

# State of currently indexed document
ACTIVE_DOCUMENT: Dict[str, Any] = {
    "filename": None,
    "filepath": None,
    "num_pages": 0,
    "num_chunks": 0,
    "uploaded_at": None,
    "chunks": [],           # List of dicts: {"text": str, "page": int, "chunk_id": int}
    "embeddings": None,     # np.ndarray of shape (num_chunks, embedding_dim)
    "embedding_mode": None  # "gemini" or "tfidf_vector"
}

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF file.
    Returns a list of dicts with page number and extracted text.
    """
    pages_data = []
    doc = pymupdf.open(str(pdf_path))
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page": page_idx + 1,
                    "text": text
                })
    finally:
        doc.close()
    return pages_data

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split extracted text into overlapping chunks while preserving page numbers.
    """
    chunks = []
    chunk_counter = 0

    for item in pages_data:
        page_num = item["page"]
        text = item["text"]
        
        # Clean excessive newlines/spaces
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text:
            continue
            
        start = 0
        text_len = len(cleaned_text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_slice = cleaned_text[start:end]
            
            # If not at the end of text, try to split at sentence or word boundary
            if end < text_len:
                last_period = chunk_slice.rfind('. ')
                if last_period > chunk_size * 0.6:
                    end = start + last_period + 1
                    chunk_slice = cleaned_text[start:end]
                else:
                    last_space = chunk_slice.rfind(' ')
                    if last_space > chunk_size * 0.6:
                        end = start + last_space
                        chunk_slice = cleaned_text[start:end]

            chunk_str = chunk_slice.strip()
            if len(chunk_str) > 20:  # Skip tiny noise chunks
                chunks.append({
                    "chunk_id": chunk_counter,
                    "page": page_num,
                    "text": chunk_str
                })
                chunk_counter += 1

            if end >= text_len:
                break

            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

    return chunks

class LocalTFIDFVectorizer:
    """
    Lightweight, deterministic local text vectorizer.
    Used for instant local fallback if Gemini API is offline or quota reached.
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tokenized = [re.findall(r'\b[a-zA-Z0-9_]{3,}\b', t.lower()) for t in texts]
        df = {}
        total_docs = len(texts)
        for doc in tokenized:
            seen = set(doc)
            for w in seen:
                df[w] = df.get(w, 0) + 1
                
        # Build top vocabulary
        sorted_words = sorted(df.items(), key=lambda x: x[1], reverse=True)[:2000]
        self.vocab = {w: i for i, (w, _) in enumerate(sorted_words)}
        self.idf = {w: math.log((1 + total_docs) / (1 + df[w])) + 1.0 for w, _ in sorted_words}
        
        matrix = np.zeros((total_docs, len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(tokenized):
            for w in doc:
                if w in self.vocab:
                    matrix[i, self.vocab[w]] += 1.0
            # Apply IDF and L2 normalize
            for w, idx in self.vocab.items():
                matrix[i, idx] *= self.idf[w]
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
        return matrix

    def transform(self, query: str) -> np.ndarray:
        tokens = re.findall(r'\b[a-zA-Z0-9_]{3,}\b', query.lower())
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w in tokens:
            if w in self.vocab:
                vec[self.vocab[w]] += 1.0 * self.idf[w]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

LOCAL_VECTORIZER = LocalTFIDFVectorizer()

def generate_embeddings(texts: List[str]) -> tuple[np.ndarray, str]:
    """
    Generate embeddings using LangChain GoogleGenerativeAIEmbeddings if key is present,
    otherwise fallback to fast local vectorizer.
    """
    if is_gemini_configured():
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GEMINI_API_KEY
            )
            # Batch embedding
            raw_embeddings = embedder.embed_documents(texts)
            arr = np.array(raw_embeddings, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            return arr, "gemini"
        except Exception as e:
            logger.warning(f"Gemini embeddings failed or quota limited ({e}). Falling back to local vectorizer.")
            
    # Fallback to local vectorizer
    arr = LOCAL_VECTORIZER.fit_transform(texts)
    return arr, "local_tfidf"

def process_and_index_pdf(pdf_path: Path, original_filename: str) -> Dict[str, Any]:
    """
    Full pipeline: extract, chunk, embed, and store in active document state.
    """
    global ACTIVE_DOCUMENT
    
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("Could not extract any readable text from the uploaded PDF. It might be scanned or image-only.")
        
    chunks = chunk_text(pages)
    if not chunks:
        raise ValueError("PDF content is too brief or contains no extractable paragraphs.")
        
    texts = [c["text"] for c in chunks]
    embeddings, mode = generate_embeddings(texts)
    
    from datetime import datetime, timezone
    ACTIVE_DOCUMENT = {
        "filename": original_filename,
        "filepath": str(pdf_path),
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_mode": mode
    }
    
    return {
        "success": True,
        "filename": original_filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "embedding_mode": mode,
        "status": "ready"
    }

def get_active_document_info() -> Dict[str, Any]:
    """Return status information for the currently active document."""
    if not ACTIVE_DOCUMENT.get("filename"):
        return {
            "has_document": False,
            "filename": None,
            "num_pages": 0,
            "num_chunks": 0,
            "status": "No document uploaded"
        }
    return {
        "has_document": True,
        "filename": ACTIVE_DOCUMENT["filename"],
        "num_pages": ACTIVE_DOCUMENT["num_pages"],
        "num_chunks": ACTIVE_DOCUMENT["num_chunks"],
        "uploaded_at": ACTIVE_DOCUMENT["uploaded_at"],
        "mode": ACTIVE_DOCUMENT["embedding_mode"],
        "status": "Ready for questions"
    }

def clear_active_document():
    """Remove active document and release resources."""
    global ACTIVE_DOCUMENT
    ACTIVE_DOCUMENT = {
        "filename": None,
        "filepath": None,
        "num_pages": 0,
        "num_chunks": 0,
        "uploaded_at": None,
        "chunks": [],
        "embeddings": None,
        "embedding_mode": None
    }

def retrieve_relevant_chunks(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a user question using cosine similarity.
    """
    if not ACTIVE_DOCUMENT.get("chunks") or ACTIVE_DOCUMENT.get("embeddings") is None:
        return []
        
    chunks = ACTIVE_DOCUMENT["chunks"]
    doc_embeddings = ACTIVE_DOCUMENT["embeddings"]
    mode = ACTIVE_DOCUMENT["embedding_mode"]
    
    try:
        if mode == "gemini" and is_gemini_configured():
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GEMINI_API_KEY
            )
            q_emb = np.array(embedder.embed_query(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm
        else:
            q_emb = LOCAL_VECTORIZER.transform(query)
            
        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, q_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(similarities[idx])
            })
        return results
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        # Return first 3 chunks as safe fallback
        return chunks[:top_k]
