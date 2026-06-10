"""LangGraph RAG pipeline."""
from __future__ import annotations
import logging
from typing import Literal
import anthropic
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from rag.config import ANTHROPIC_API_KEY, GRADED_K, LLM_MAX_TOKENS, LLM_MODEL, MAX_REWRITE_ATTEMPTS, RELEVANCE_THRESHOLD, RETRIEVAL_K
from rag.prompts import GRADE_DOCUMENT_HUMAN, GRADE_DOCUMENT_SYSTEM, NO_DOCS_FALLBACK, QUERY_REWRITE_HUMAN, QUERY_REWRITE_SYSTEM, RAG_GENERATION_HUMAN, RAG_GENERATION_SYSTEM
from rag.retriever import retrieve_with_scores
from rag.state import RAGState

logger = logging.getLogger(__name__)

def _get_client(): return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def analyze_query(state: RAGState) -> dict:
    query, attempts = state["query"], state.get("rewrite_attempts", 0)
    if attempts == 0:
        return {"rewritten_query": query, "rewrite_attempts": 1}
    r = _get_client().messages.create(model=LLM_MODEL, max_tokens=128, system=QUERY_REWRITE_SYSTEM,
        messages=[{"role": "user", "content": QUERY_REWRITE_HUMAN.format(query=query)}])
    return {"rewritten_query": r.content[0].text.strip(), "rewrite_attempts": attempts + 1}

def retrieve(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = []
    for doc, score in retrieve_with_scores(query, k=RETRIEVAL_K):
        doc.metadata["retrieval_score"] = round(float(score), 4)
        docs.append(doc)
    return {"documents": docs}

def grade_documents(state: RAGState) -> dict:
    query, docs, client = state.get("rewritten_query") or state["query"], state["documents"], _get_client()
    candidates = [d for d in docs if d.metadata.get("retrieval_score", 0) >= RELEVANCE_THRESHOLD]
    relevant: list[Document] = []
    for doc in candidates[:GRADED_K * 2]:
        try:
            resp = client.messages.create(model=LLM_MODEL, max_tokens=4, system=GRADE_DOCUMENT_SYSTEM,
                messages=[{"role": "user", "content": GRADE_DOCUMENT_HUMAN.format(query=query, document=doc.page_content[:600])}])
            if resp.content[0].text.strip().upper().startswith("YES"):
                relevant.append(doc)
                if len(relevant) >= GRADED_K: break
        except Exception as e:
            relevant.append(doc)
            if len(relevant) >= GRADED_K: break
    return {"filtered_documents": relevant, "has_relevant_docs": len(relevant) > 0}

def generate(state: RAGState) -> dict:
    query, docs = state["query"], state.get("filtered_documents", [])
    if not docs:
        return {"answer": NO_DOCS_FALLBACK, "sources": [], "input_tokens": 0, "output_tokens": 0}
    parts, sources = [], []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(f"[{i}] Category: {m.get('category','unknown')} | Score: {m.get('retrieval_score',0):.3f}\n{doc.page_content.strip()}")
        sources.append({"index": i, "category": m.get("category","unknown"), "score": m.get("retrieval_score",0.0), "cluster": m.get("dominant_cluster",-1), "snippet": doc.page_content[:200]})
    r = _get_client().messages.create(model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, thinking={"type": "adaptive"},
        system=RAG_GENERATION_SYSTEM, messages=[{"role": "user", "content": RAG_GENERATION_HUMAN.format(query=query, context="\n\n---\n\n".join(parts))}])
    answer = next((b.text for b in r.content if b.type == "text"), "No answer generated.")
    return {"answer": answer, "sources": sources, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}

def route_after_grading(state: RAGState) -> Literal["generate", "analyze_query"]:
    if state.get("has_relevant_docs"): return "generate"
    if state.get("rewrite_attempts", 0) < MAX_REWRITE_ATTEMPTS: return "analyze_query"
    return "generate"

def build_rag_graph():
    wf = StateGraph(RAGState)
    wf.add_node("analyze_query", analyze_query)
    wf.add_node("retrieve", retrieve)
    wf.add_node("grade_documents", grade_documents)
    wf.add_node("generate", generate)
    wf.add_edge(START, "analyze_query")
    wf.add_edge("analyze_query", "retrieve")
    wf.add_edge("retrieve", "grade_documents")
    wf.add_conditional_edges("grade_documents", route_after_grading, {"generate": "generate", "analyze_query": "analyze_query"})
    wf.add_edge("generate", END)
    return wf.compile()

RAGGraph = type(build_rag_graph())
