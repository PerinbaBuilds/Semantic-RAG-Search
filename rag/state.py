"""LangGraph state schema for the RAG pipeline."""
from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

def _merge_docs(a: list[Document], b: list[Document]) -> list[Document]:
    return b if b else a

class RAGState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    rewrite_attempts: int
    documents: Annotated[list[Document], _merge_docs]
    filtered_documents: list[Document]
    answer: str
    sources: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    has_relevant_docs: bool
