"""
Tools package
"""
from backend.tools.calculator import calculate
from backend.tools.document_tool import query_document
from backend.tools.web_search import search_web, is_search_configured

__all__ = ["calculate", "query_document", "search_web", "is_search_configured"]
