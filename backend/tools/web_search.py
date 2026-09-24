"""
Web Search Tool
Modular search tool that queries real-time web results if configured (e.g. Tavily API or DDG),
or returns a clear, polite notice if live search is not enabled.
"""

import os
import logging
from typing import Dict, Any
from backend.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

def is_search_configured() -> bool:
    """Check if web search integration is configured."""
    return bool(TAVILY_API_KEY and TAVILY_API_KEY.strip())

def search_web(query: str) -> Dict[str, Any]:
    """
    Execute web search if configured, or gracefully notify the user.
    """
    if not is_search_configured():
        return {
            "success": False,
            "configured": False,
            "query": query,
            "result": "Live web search is not configured for this project. To enable real-time web lookups, provide a search provider API key (such as TAVILY_API_KEY) in the .env configuration.",
            "sources": []
        }
        
    try:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            snippets = [f"- {r.get('title')}: {r.get('content')}" for r in results]
            combined = "\n".join(snippets)
            sources = [r.get("url") for r in results]
            return {
                "success": True,
                "configured": True,
                "query": query,
                "result": combined,
                "sources": sources
            }
        else:
            return {
                "success": False,
                "configured": True,
                "query": query,
                "result": f"Search provider returned status code {resp.status_code}.",
                "sources": []
            }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {
            "success": False,
            "configured": True,
            "query": query,
            "result": f"Error performing live web search: {str(e)}",
            "sources": []
        }
