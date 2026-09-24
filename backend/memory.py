"""
Conversation Memory Management
Maintains multi-turn context per session using SQLite-backed history and LangChain message objects.
"""

from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from backend.database.database import get_messages, add_message

# System instructions for general-purpose assistant
SYSTEM_PROMPT = """You are a helpful, brilliant, and articulate general-purpose AI Assistant named AgentFlow AI.
You assist users with diverse queries including computer science concepts, programming, technical writing, study questions, mathematics, document analysis, and general knowledge.

Guidelines:
1. Provide clear, accurate, well-structured, and easy-to-understand explanations.
2. When answering programming questions, provide high-quality, formatted code with clear explanations.
3. If referencing previous facts or names told by the user in this conversation, recall them accurately.
4. Maintain a polite, professional, and engaging tone.
5. If answering based on uploaded document context, cite facts directly from the document accurately.
"""

def get_session_history_as_messages(conversation_id: str, limit: int = 20) -> List[BaseMessage]:
    """
    Retrieve message history for a conversation and convert it to LangChain BaseMessage instances.
    """
    db_messages = get_messages(conversation_id, limit=limit)
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    
    for msg in db_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    return messages

def format_history_for_prompt(conversation_id: str, limit: int = 10) -> str:
    """
    Format recent history as a readable dialogue string.
    """
    db_messages = get_messages(conversation_id, limit=limit)
    if not db_messages:
        return "No previous conversation history."
        
    formatted = []
    for msg in db_messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        formatted.append(f"{role}: {msg.get('content', '')}")
        
    return "\n".join(formatted)
