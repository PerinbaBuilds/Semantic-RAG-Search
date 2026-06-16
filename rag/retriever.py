"""Retriever using direct chromadb + sentence-transformers (no langchain-chroma)."""
from __future__ import annotations
import logging
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)

def retrieve_with_scores(query: str, k: int = RETRIEVAL_K) -> list[tuple[Document, float]]:
    model = _get_model()
    emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    col = _get_collection()
    results = col.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs_and_scores: list[tuple[Document, float]] = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1.0 - dist / 2.0, 4)
        docs_and_scores.append((Document(page_content=text, metadata=meta or {}), score))
    return docs_and_scores
