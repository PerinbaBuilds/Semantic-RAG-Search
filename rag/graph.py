"""LangGraph RAG pipeline."""
from __future__ import annotations
import logging
from typing import Literal
from groq import Groq
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from rag.config import GROQ_API_KEY, GRADED_K, LLM_MAX_TOKENS, LLM_MODEL, MAX_REWRITE_ATTEMPTS, RELEVANCE_THRESHOLD, RETRIEVAL_K
from rag.prompts import GRADE_DOCUMENT_HUMAN, GRADE_DOCUMENT_SYSTEM, NO_DOCS_FALLBACK, QUERY_REWRITE_HUMAN, QUERY_REWRITE_SYSTEM, RAG_GENERATION_HUMAN, RAG_GENERATION_SYSTEM
from rag.retriever import retrieve_with_scores
from rag.state import RAGState

logger = logging.getLogger(__name__)

def _get_client(): return Groq(api_key=GROQ_API_KEY)

def _chat(client: Groq, system: str, user: str, max_tokens: int = 256) -> str:
    r = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""

def analyze_query(state: RAGState) -> dict:
    query, attempts = state["query"], state.get("rewrite_attempts", 0)
    if attempts == 0:
        return {"rewritten_query": query, "rewrite_attempts": 1}
    text = _chat(_get_client(), QUERY_REWRITE_SYSTEM, QUERY_REWRITE_HUMAN.format(query=query), max_tokens=128)
    return {"rewritten_query": text.strip(), "rewrite_attempts": attempts + 1}

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
            text = _chat(client, GRADE_DOCUMENT_SYSTEM,
                GRADE_DOCUMENT_HUMAN.format(query=query, document=doc.page_content[:600]), max_tokens=4)
            if text.strip().upper().startswith("YES"):
                relevant.append(doc)
                if len(relevant) >= GRADED_K: break
        except Exception:
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
    client = _get_client()
    r = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": RAG_GENERATION_SYSTEM},
            {"role": "user", "content": RAG_GENERATION_HUMAN.format(query=query, context="\n\n---\n\n".join(parts))},
        ],
    )
    answer = r.choices[0].message.content or "No answer generated."
    return {"answer": answer, "sources": sources, "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}

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