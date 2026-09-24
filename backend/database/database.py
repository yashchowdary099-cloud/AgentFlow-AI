"""
SQLite Database Layer
Stores conversations, message history, tool usage, and execution activities.
Ensures connections are properly closed and handles timezone-aware timestamps.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from backend.config import DB_PATH

@contextmanager
def get_db():
    """Context manager for SQLite connections that guarantees connection closure."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def _utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def init_db():
    """Initialize database tables with appropriate indexes."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Conversations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                tool_used TEXT DEFAULT 'none',
                activity_json TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)")
        
        conn.commit()

def create_conversation(title: str = "New Conversation", conversation_id: Optional[str] = None) -> str:
    """Create a new conversation session and return its ID."""
    cid = conversation_id or str(uuid.uuid4())
    now = _utc_now_iso()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title, now, now)
        )
        conn.commit()
    return cid

def get_conversations() -> List[Dict[str, Any]]:
    """Retrieve all conversations ordered by recent activity."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get single conversation metadata."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_conversation_title(conversation_id: str, title: str):
    """Update conversation title."""
    now = _utc_now_iso()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conversation_id)
        )
        conn.commit()

def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()

def add_message(
    conversation_id: str,
    role: str,
    content: str,
    tool_used: str = "none",
    activity: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Insert a message into the conversation history."""
    msg_id = str(uuid.uuid4())
    now = _utc_now_iso()
    act_str = json.dumps(activity or [])
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            title = content[:30] + ("..." if len(content) > 30 else "")
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now)
            )
        else:
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )
            
        cursor.execute(
            """INSERT INTO messages (id, conversation_id, role, content, tool_used, activity_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conversation_id, role, content, tool_used, act_str, now)
        )
        conn.commit()
        
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "tool_used": tool_used,
        "activity": activity or [],
        "created_at": now
    }

def get_messages(conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve message history for a conversation."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, conversation_id, role, content, tool_used, activity_json, created_at
               FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (conversation_id, limit)
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            try:
                d["activity"] = json.loads(d.get("activity_json") or "[]")
            except Exception:
                d["activity"] = []
            result.append(d)
        return result

def clear_history(conversation_id: str):
    """Clear messages for a specific conversation."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.commit()

# Initialize tables on import
init_db()
