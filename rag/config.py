"""RAG pipeline configuration."""
import os
from pathlib import Path

BASE_DIR         = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR   = BASE_DIR / "embeddings"
CHROMA_DIR       = EMBEDDINGS_DIR / "chroma_db"
COLLECTION_NAME  = "newsgroups"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
RETRIEVAL_K          = 8
GRADED_K             = 5
RELEVANCE_THRESHOLD  = 0.6
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL     = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "4096"))
MAX_REWRITE_ATTEMPTS = 2