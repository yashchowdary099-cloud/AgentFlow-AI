"""
Document Retrieval Tool for LangChain Agent
Provides context from the currently uploaded PDF document.
"""

from typing import Dict, Any, List
from backend.services.document_service import retrieve_relevant_chunks, get_active_document_info

def query_document(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Search the currently uploaded PDF document for information relevant to the user query.
    
    Returns:
        dict: {
            "success": bool,
            "has_document": bool,
            "filename": str | None,
            "chunks_found": int,
            "context": str,
            "sources": list[dict]
        }
    """
    doc_info = get_active_document_info()
    if not doc_info.get("has_document"):
        return {
            "success": False,
            "has_document": False,
            "filename": None,
            "chunks_found": 0,
            "context": "No document has been uploaded yet. Please upload a PDF first to ask questions about it.",
            "sources": []
        }
        
    chunks = retrieve_relevant_chunks(query, top_k=top_k)
    if not chunks:
        return {
            "success": True,
            "has_document": True,
            "filename": doc_info["filename"],
            "chunks_found": 0,
            "context": "I couldn't find enough information about that in the uploaded document.",
            "sources": []
        }
        
    # Format clean context with page references
    formatted_pieces = []
    sources = []
    for c in chunks:
        page_num = c.get("page", 1)
        text = c.get("text", "")
        formatted_pieces.append(f"[Page {page_num}]: {text}")
        sources.append({"page": page_num, "chunk_id": c.get("chunk_id", 0)})
        
    combined_context = "\n\n".join(formatted_pieces)
    
    return {
        "success": True,
        "has_document": True,
        "filename": doc_info["filename"],
        "chunks_found": len(chunks),
        "context": combined_context,
        "sources": sources
    }
