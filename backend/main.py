"""
FastAPI Server Entrypoint
Provides REST endpoints for chat, PDF upload, session management, and serves the frontend.
"""

import os
import shutil
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import (
    BASE_DIR,
    UPLOADS_DIR,
    FRONTEND_DIR,
    PRIMARY_MODEL,
    is_gemini_configured
)
from backend.services.chat_service import process_chat_message
from backend.services.document_service import (
    process_and_index_pdf,
    get_active_document_info,
    clear_active_document
)
from backend.services.session_service import (
    start_new_session,
    fetch_all_sessions,
    fetch_session_history,
    remove_session
)

app = FastAPI(
    title="LangChain & AI Agents Assistant",
    description="General Purpose Intelligent AI Assistant powered by LangChain, LangGraph, and Google Gemini.",
    version="1.0.0"
)

# Enable CORS for local development and cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User's prompt or question")
    conversation_id: Optional[str] = Field(None, description="Active session ID")

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    tool_used: str
    activity: list[str]

class NewChatResponse(BaseModel):
    conversation_id: str
    status: str

# ----------------- API Endpoints ----------------- #

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint reporting API status and configuration state.
    """
    doc_info = get_active_document_info()
    return {
        "status": "ok",
        "service": "LangChain AI Agent",
        "gemini_configured": is_gemini_configured(),
        "model": PRIMARY_MODEL,
        "active_document": doc_info.get("filename"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat endpoint:
    Processes user query through the LangGraph Agent and persists the turn.
    """
    cid = request.conversation_id or start_new_session()
    user_msg = request.message.strip()
    
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be blank."
        )
        
    try:
        result = process_chat_message(conversation_id=cid, message=user_msg)
        return ChatResponse(**result)
    except Exception as exc:
        return ChatResponse(
            answer="Something went wrong while generating the answer. Please try again.",
            conversation_id=cid,
            tool_used="none",
            activity=["Request received", "Execution encountered an error"]
        )

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document for RAG-based context retrieval.
    """
    # 1. Validation: MIME / Extension
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    # 2. Sanitize filename
    safe_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', filename)
    saved_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    target_path = UPLOADS_DIR / saved_filename

    # 3. Read content with size limit check (Max 25 MB)
    max_size = 25 * 1024 * 1024
    try:
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File is too large. Maximum allowed size is 25 MB."
            )
        with open(target_path, "wb") as buffer:
            buffer.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # 4. Extract, chunk, embed, and index
    try:
        summary = process_and_index_pdf(target_path, original_filename=filename)
        return {
            "success": True,
            "filename": filename,
            "pages": summary["pages"],
            "chunks": summary["chunks"],
            "embedding_mode": summary["embedding_mode"],
            "status": "Ready for questions"
        }
    except Exception as exc:
        try:
            if target_path.exists():
                target_path.unlink()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process PDF: {str(exc)}"
        )

@app.get("/api/document")
async def get_document_status():
    """Retrieve details on the active PDF document."""
    return get_active_document_info()

@app.delete("/api/document")
async def remove_document():
    """Clear the active PDF document index."""
    clear_active_document()
    return {"success": True, "message": "Active document removed successfully."}

@app.post("/api/new-chat", response_model=NewChatResponse)
async def new_chat():
    """Create a fresh conversation session."""
    cid = start_new_session()
    return NewChatResponse(conversation_id=cid, status="created")

@app.get("/api/conversations")
async def list_conversations():
    """List all saved conversations."""
    return fetch_all_sessions()

@app.get("/api/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Fetch message history for the requested session."""
    return fetch_session_history(conversation_id)

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation and its messages."""
    remove_session(conversation_id)
    return {"success": True, "message": f"Conversation {conversation_id} deleted."}

# ----------------- Frontend Static Mount ----------------- #
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        """Serve the frontend single page application."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "Frontend index.html not found"}, status_code=404)

    @app.get("/styles.css")
    async def serve_css():
        """Serve the frontend stylesheet."""
        css_file = FRONTEND_DIR / "styles.css"
        if css_file.exists():
            return FileResponse(str(css_file), media_type="text/css")
        return JSONResponse({"message": "styles.css not found"}, status_code=404)

    @app.get("/app.js")
    async def serve_js():
        """Serve the frontend JavaScript."""
        js_file = FRONTEND_DIR / "app.js"
        if js_file.exists():
            return FileResponse(str(js_file), media_type="text/javascript")
        return JSONResponse({"message": "app.js not found"}, status_code=404)
