"""LangChain retriever wrapping the existing ChromaDB vector store."""
from __future__ import annotations
import logging
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

def get_retriever(k: int = RETRIEVAL_K) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})

def retrieve_with_scores(query: str, k: int = RETRIEVAL_K) -> list[tuple]:
    return get_vectorstore().similarity_search_with_relevance_scores(query, k=k)
