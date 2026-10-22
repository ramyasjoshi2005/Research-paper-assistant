import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = BASE_DIR / "papers"
PDF_DIR = PAPERS_DIR / "pdfs"
STORAGE_DIR = BASE_DIR / "storage"
CACHE_DIR = STORAGE_DIR / "cache"
VECTOR_DB_DIR = STORAGE_DIR / "vector_db"

# Ensure runtime directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
ARXIV_METADATA_PATH = DATA_DIR / "arxiv-metadata.json"
METADATA_SAMPLE_CACHE_PATH = CACHE_DIR / "metadata_sample.json"
EXTRACTED_TEXT_CACHE_DIR = CACHE_DIR / "extracted_text"
EXTRACTED_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking and Retrieval configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 4))
DEFAULT_DOWNLOAD_LIMIT = int(os.getenv("MAX_DOWNLOAD_LIMIT", 10))

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# arXiv Download Settings
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf/"
ARXIV_ABS_BASE_URL = "https://arxiv.org/abs/"
USER_AGENT = "ResearchPaperAssistant/1.0 (academic-demo; mailto:student@example.com)"
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 1.0


# ==============================================================================
# Centralized, Safe Configuration & Secrets Loader
# Prevents repeated "No secrets files found" exceptions across components
# ==============================================================================
_SECRETS_CHECK_DONE = False
_SECRETS_FILE_EXISTS = False


def _has_physical_secrets_file() -> bool:
    """Checks whether a valid secrets.toml file physically exists on disk."""
    global _SECRETS_CHECK_DONE, _SECRETS_FILE_EXISTS
    if not _SECRETS_CHECK_DONE:
        candidates = [
            BASE_DIR / ".streamlit" / "secrets.toml",
            Path(".streamlit/secrets.toml"),
            Path.home() / ".streamlit" / "secrets.toml",
        ]
        _SECRETS_FILE_EXISTS = any(p.exists() and p.is_file() for p in candidates)
        _SECRETS_CHECK_DONE = True
    return _SECRETS_FILE_EXISTS


def get_config_val(key: str, default: str = "") -> str:
    """
    Centralized safe configuration loader.
    Checks environment variables first, then Streamlit secrets ONLY if a secrets.toml
    file physically exists on disk.
    Never raises FileNotFoundError or displays raw exceptions to users.
    """
    # 1. Environment variable check
    env_val = os.getenv(key, "")
    if env_val:
        return env_val

    # 2. Safe Streamlit secrets check (only if secrets.toml exists on disk)
    if _has_physical_secrets_file():
        try:
            import streamlit as st

            # Access st.secrets safely
            if hasattr(st, "secrets") and key in st.secrets:
                val = st.secrets[key]
                if val:
                    return str(val).strip()
        except Exception:
            pass

    return default


# Static defaults for initial module load
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
