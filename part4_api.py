"""
Part 4: FastAPI Service
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from part3_cache import SemanticCache
from rag.graph import build_rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

EMBEDDINGS_DIR  = Path("embeddings")
CHROMA_DIR      = Path("embeddings/chroma_db")
COLLECTION_NAME = "newsgroups"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85

class AppState:
    embedding_model: Any = None
    chroma_collection: Any = None
    cache: Optional[SemanticCache] = None
    rag_graph: Any = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting up ===")
    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error(f"Embedding model error: {e}")
        state.embedding_model = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        state.chroma_collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB loaded: {state.chroma_collection.count()} docs")
    except Exception as e:
        logger.error(f"ChromaDB error: {e}")
        state.chroma_collection = None
    try:
        state.cache = SemanticCache(similarity_threshold=SIMILARITY_THRESHOLD, embeddings_dir=EMBEDDINGS_DIR)
        logger.info("Cache loaded")
    except Exception as e:
        logger.error(f"Cache error: {e}")
        state.cache = None
    try:
        state.rag_graph = build_rag_graph()
        logger.info(f"Ready in {time.perf_counter()-t0:.2f}s")
    except Exception as e:
        logger.error(f"RAG graph init failed: {e}")
        state.rag_graph = None
    yield
    logger.info("=== Shutting down ===")

app = FastAPI(title="Newsgroups Semantic Search", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(): return FileResponse("static/index.html")

class QueryRequest(BaseModel):
    query: str
    model_config = {"json_schema_extra": {"example": {"query": "space shuttle missions"}}}

class QueryResponse(BaseModel):
    query: str; cache_hit: bool; matched_query: Optional[str] = None
    similarity_score: Optional[float] = None; result: Any; dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int; hit_count: int; miss_count: int; hit_rate: float

class RAGQueryRequest(BaseModel):
    query: str

class RAGSource(BaseModel):
    index: int; category: str; score: float; cluster: int; snippet: str

class RAGQueryResponse(BaseModel):
    query: str; rewritten_query: Optional[str]; answer: str
    sources: list[RAGSource]; input_tokens: int; output_tokens: int; rewrite_attempts: int

def embed_query(query: str) -> np.ndarray:
    return state.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

def search_corpus(query_emb: np.ndarray, n_results: int = 5) -> list[dict]:
    if state.chroma_collection is None: return []
    results = state.chroma_collection.query(query_embeddings=[query_emb.tolist()], n_results=n_results, include=["documents","metadatas","distances"])
    return [{"text": d, "category": m.get("category","unknown"), "dominant_cluster": m.get("dominant_cluster",-1), "cluster_ids": m.get("cluster_ids",""), "similarity": round(1.0-dist/2.0,4)} for d,m,dist in zip(results["documents"][0],results["metadatas"][0],results["distances"][0])]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not request.query.strip(): raise HTTPException(422, "Query must not be empty.")
    if state.embedding_model is None: raise HTTPException(503, "Embedding model unavailable.")
    qt = request.query.strip(); qe = embed_query(qt)
    if state.cache is not None:
        ce = state.cache.lookup(qt, qe)
        if ce: return QueryResponse(query=qt, cache_hit=True, matched_query=ce.query, similarity_score=round(float(np.dot(qe,ce.embedding)),4), result=ce.result, dominant_cluster=ce.dominant_cluster)
    docs = search_corpus(qe); result = {"top_documents": docs, "query_time_ms": None}
    if state.cache is not None:
        entry = state.cache.store(qt, qe, result)
        dominant_cluster = entry.dominant_cluster
    else:
        dominant_cluster = -1
    return QueryResponse(query=qt, cache_hit=False, matched_query=None, similarity_score=None, result=result, dominant_cluster=dominant_cluster)

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    s = state.cache.stats
    return CacheStatsResponse(total_entries=s["total_entries"], hit_count=s["hit_count"], miss_count=s["miss_count"], hit_rate=s["hit_rate"])

@app.delete("/cache")
async def flush_cache() -> dict:
    state.cache.flush(); return {"status": "ok", "message": "Cache flushed."}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "corpus_loaded": state.chroma_collection is not None, "fcm_loaded": state.cache._fcm is not None if state.cache else False, "cache_entries": len(state.cache) if state.cache else 0, "rag_ready": state.rag_graph is not None}

@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    if not request.query.strip(): raise HTTPException(422, "Query must not be empty.")
    if state.rag_graph is None: raise HTTPException(503, "RAG graph unavailable. Check GROQ_API_KEY.")
    query = request.query.strip()
    try:
        result = await state.rag_graph.ainvoke({"query": query, "rewritten_query": None, "rewrite_attempts": 0, "documents": [], "filtered_documents": [], "answer": "", "sources": [], "input_tokens": 0, "output_tokens": 0, "has_relevant_docs": False})
    except Exception as e:
        raise HTTPException(500, f"RAG pipeline error: {e}")
    return RAGQueryResponse(query=query, rewritten_query=result.get("rewritten_query"), answer=result.get("answer",""), sources=[RAGSource(**s) for s in (result.get("sources") or [])], input_tokens=result.get("input_tokens",0), output_tokens=result.get("output_tokens",0), rewrite_attempts=result.get("rewrite_attempts",0))
