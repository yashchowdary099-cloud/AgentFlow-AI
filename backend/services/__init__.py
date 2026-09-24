"""
Services package
"""
from backend.services.chat_service import process_chat_message
from backend.services.document_service import (
    process_and_index_pdf,
    get_active_document_info,
    clear_active_document,
    retrieve_relevant_chunks
)
from backend.services.session_service import (
    start_new_session,
    fetch_all_sessions,
    fetch_session_history,
    remove_session
)
