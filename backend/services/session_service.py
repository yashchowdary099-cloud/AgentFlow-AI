"""
Session & History Service
Handles conversation creation, switching, history retrieval, and session deletion.
"""

from typing import List, Dict, Any, Optional
from backend.database.database import (
    create_conversation,
    get_conversations,
    get_conversation,
    get_messages,
    delete_conversation,
    clear_history
)

def start_new_session(title: str = "New Conversation") -> str:
    """Start and return a new conversation ID."""
    return create_conversation(title=title)

def fetch_all_sessions() -> List[Dict[str, Any]]:
    """Fetch all stored sessions."""
    return get_conversations()

def fetch_session_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve full message history for a given session."""
    return get_messages(conversation_id=conversation_id)

def remove_session(conversation_id: str):
    """Delete session and all its messages."""
    delete_conversation(conversation_id=conversation_id)
