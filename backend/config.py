"""
Application Configuration
Centralizes paths, environment variables, and model settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

# Resolve project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)


# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FRONTEND_DIR = BASE_DIR / "frontend"
DB_PATH = DATA_DIR / "chat_history.db"

# Ensure required folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GOOGLE GEMINI CONFIGURATION
# ============================================================

# Keep Gemini because it may still be used for embeddings
# and can be used as a fallback later.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()

FALLBACK_MODEL = "gemini-flash-latest"

# Used by PDF/vector embedding functionality
EMBEDDING_MODEL = "models/gemini-embedding-001"


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
).strip()


# ============================================================
# OPTIONAL WEB SEARCH
# ============================================================

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    ""
).strip()


# ============================================================
# SERVER SETTINGS
# ============================================================

HOST = os.getenv(
    "HOST",
    "127.0.0.1"
)

PORT = int(
    os.getenv("PORT", "8000")
)

DEBUG = os.getenv(
    "DEBUG",
    "True"
).lower() in ("true", "1", "yes")


# ============================================================
# CONFIGURATION CHECKS
# ============================================================

def is_gemini_configured() -> bool:
    """Check whether a valid Gemini API key is configured."""

    return bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY != "your_gemini_api_key_here"
    )


def is_openrouter_configured() -> bool:
    """Check whether a valid OpenRouter API key is configured."""

    return bool(
        OPENROUTER_API_KEY
        and OPENROUTER_API_KEY != "your_private_key_here"
        and OPENROUTER_API_KEY != "your_openrouter_key_here"
    )