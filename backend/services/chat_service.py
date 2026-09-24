"""
Chat Service
Coordinates agent execution with database message persistence.
"""

from typing import Dict, Any
from backend.database.database import add_message

def process_chat_message(conversation_id: str, message: str) -> Dict[str, Any]:
    """
    Process incoming user message:
    1. Persist user message to SQLite.
    2. Invoke LangGraph Agent workflow.
    3. Persist AI assistant response to SQLite.
    4. Return payload for API and UI rendering.
    """
    from backend.agent import run_agent
    clean_msg = message.strip()
    if not clean_msg:
        raise ValueError("Message cannot be empty.")
        
    # 1. Save user message to database
    add_message(
        conversation_id=conversation_id,
        role="user",
        content=clean_msg,
        tool_used="none",
        activity=[]
    )
    
    # 2. Run agent
    agent_output = run_agent(conversation_id=conversation_id, user_query=clean_msg)
    
    answer = agent_output.get("answer", "")
    tool_used = agent_output.get("tool_used", "none")
    activity = agent_output.get("activity", [])
    
    # 3. Save assistant message to database
    add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        tool_used=tool_used,
        activity=activity
    )
    
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "tool_used": tool_used,
        "activity": activity
    }
