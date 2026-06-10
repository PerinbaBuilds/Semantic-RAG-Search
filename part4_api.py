"""
Part 4: FastAPI Service
========================

Endpoints:
  POST /query          — semantic search with cache
  GET  /cache/stats    — cache statistics
  DELETE /cache        — flush cache

State management:
  - The SentenceTransformer model and ChromaDB collection are loaded once
    at startup (lifespan context manager) and shared across requests.
  - The SemanticCache instance is application-level state (single instance,
    thread-safe due to internal locking in part3_cache.py).
  - All state lives in process memory; no external dependencies at runtime.

Startup sequence:
  1. Load sentence-transformers embedding model
  2. Connect to ChromaDB (must have been populated by Part 1)
  3. Instantiate SemanticCache (loads FCM model from Part 2)

On a miss, the service:
  1. Queries ChromaDB for the top-5 semantically similar documents
  2. Assembles a result dict with the top document text + metadata
  3. Stores the query + result in the semantic cache
  4. Returns the response
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from part3_cache import SemanticCache
from rag.graph import build_rag_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

EMBEDDINGS_DIR  = Path("embeddings")
CHROMA_DIR      = Path("embeddings/chroma_db")
COLLECTION_NAME = "newsgroups"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------

class AppState:
    embedding_model: Any = None
    chroma_collection: Any = None
    cache: Optional[SemanticCache] = None
    rag_graph: Any = None


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavyweight resources once at startup; release on shutdown.
    Using the lifespan approach (rather than @app.on_event) is the modern
    FastAPI pattern and avoids deprecation warnings.
    """
    logger.info("=== Starting up Newsgroups Semantic Search API ===")
    t0 = time.perf_counter()

    # 1. Embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f"Embedding model loaded in {time.perf_counter()-t0:.2f}s")

    # 2. ChromaDB
    logger.info("Connecting to ChromaDB...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        state.chroma_collection = client.get_collection(COLLECTION_NAME)
        logger.info(
            f"ChromaDB connected. Collection '{COLLECTION_NAME}' has "
            f"{state.chroma_collection.count():,} documents."
        )
    except Exception as e:
        logger.error(
            f"ChromaDB not found or empty ({e}). "
            "Run part1_prepare.py and part2_clustering.py first."
        )
        # Allow startup to proceed so the API can still serve cache-related endpoints
        state.chroma_collection = None

    # 3. Semantic cache
    logger.info("Initialising semantic cache...")
    state.cache = SemanticCache(
        similarity_threshold=SIMILARITY_THRESHOLD,
        embeddings_dir=EMBEDDINGS_DIR,
    )
    logger.info(f"SemanticCache ready (θ={SIMILARITY_THRESHOLD})")

    logger.info("Compiling RAG graph (LangGraph + Claude)...")
    try:
      state.rag_graph = build_rag_graph()
      logger.info(f"RAG graph ready. Total startup time: {time.perf_counter()-t0:.2f}s")
    except Exception as e:
      logger.error(f"RAG graph init failed ({e}). /rag/query will be unavailable.")
      state.rag_graph = None

    yield   # App runs here

    logger.info("=== Shutting down ===")
    # Cleanup (models are GC'd automatically; nothing else to close)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Newsgroups Semantic Search",
    description=(
        "Semantic search over 20 Newsgroups with fuzzy clustering "
        "and cluster-accelerated semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

    model_config = {"json_schema_extra": {"example": {"query": "space shuttle missions"}}}


class QueryResponse(BaseModel):
    query:            str
    cache_hit:        bool
    matched_query:    Optional[str]  = None
    similarity_score: Optional[float] = None
    result:           Any
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float


# ---------------------------------------------------------------------------
# Helper: embed a single query string
# ---------------------------------------------------------------------------

def embed_query(query: str) -> np.ndarray:
    """
    Embed a query string into a L2-normalised float32 vector.
    single=True avoids unnecessary batch overhead for real-time use.
    """
    emb = state.embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: query ChromaDB for top-k similar documents
# ---------------------------------------------------------------------------

def search_corpus(query_emb: np.ndarray,
                   n_results: int = 5) -> list[dict]:
    """
    Query the vector store and return top-n matching documents with metadata.
    """
    if state.chroma_collection is None:
        return []

    results = state.chroma_collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine *distance* (0=identical, 2=opposite)
        # Convert to similarity: sim = 1 - dist/2  (for normalised embeddings)
        similarity = 1.0 - dist / 2.0
        docs.append({
            "text":             doc,
            "category":         meta.get("category", "unknown"),
            "dominant_cluster": meta.get("dominant_cluster", -1),
            "cluster_ids":      meta.get("cluster_ids", ""),
            "similarity":       round(similarity, 4),
        })
    return docs


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a natural language query against the newsgroups corpus.
    
    Flow:
      1. Embed the query.
      2. Check the semantic cache.
         - HIT:  return cached result immediately.
         - MISS: search ChromaDB, build result, store in cache, return.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    query_text = request.query.strip()
    query_emb  = embed_query(query_text)

    # --- Cache lookup ---
    cache_entry = state.cache.lookup(query_text, query_emb)

    if cache_entry is not None:
        logger.info(
            f"CACHE HIT  | '{query_text[:60]}' → matched '{cache_entry.query[:60]}'"
        )
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=cache_entry.query,
            similarity_score=round(float(np.dot(query_emb, cache_entry.embedding)), 4),
            result=cache_entry.result,
            dominant_cluster=cache_entry.dominant_cluster,
        )

    # --- Cache miss: search corpus ---
    logger.info(f"CACHE MISS | '{query_text[:60]}'")
    top_docs = search_corpus(query_emb, n_results=5)

    result = {
        "top_documents": top_docs,
        "query_time_ms": None,   # could add timing here
    }

    dominant_cluster = top_docs[0]["dominant_cluster"] if top_docs else -1

    # --- Store in cache ---
    entry = state.cache.store(query_text, query_emb, result)

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=entry.dominant_cluster,
    )


# ---------------------------------------------------------------------------
# GET /cache/stats
# ---------------------------------------------------------------------------

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    """Return current cache statistics."""
    s = state.cache.stats
    return CacheStatsResponse(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


# ---------------------------------------------------------------------------
# DELETE /cache
# ---------------------------------------------------------------------------

@app.delete("/cache")
async def flush_cache() -> dict:
    """Flush the semantic cache and reset all counters."""
    state.cache.flush()
    return {"status": "ok", "message": "Cache flushed and stats reset."}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

# REPLACE WITH:
@app.get("/health")
async def health() -> dict:
    return {
        "status":        "ok",
        "corpus_loaded": state.chroma_collection is not None,
        "fcm_loaded":    state.cache._fcm is not None if state.cache else False,
        "cache_entries": len(state.cache) if state.cache else 0,
        "rag_ready":     state.rag_graph is not None,
    }


class RAGQueryRequest(BaseModel):
    query: str
    model_config = {"json_schema_extra": {"example": {"query": "What did people think about the space shuttle?"}}}


class RAGSource(BaseModel):
    index: int
    category: str
    score: float
    cluster: int
    snippet: str


class RAGQueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str]
    answer: str
    sources: list[RAGSource]
    input_tokens: int
    output_tokens: int
    rewrite_attempts: int


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    if state.rag_graph is None:
        raise HTTPException(status_code=503, detail="RAG graph unavailable. Check ANTHROPIC_API_KEY.")
    query = request.query.strip()
    try:
        result = await state.rag_graph.ainvoke({
            "query": query, "rewritten_query": None, "rewrite_attempts": 0,
            "documents": [], "filtered_documents": [], "answer": "",
            "sources": [], "input_tokens": 0, "output_tokens": 0, "has_relevant_docs": False,
        })
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")
    return RAGQueryResponse(
        query=query,
        rewritten_query=result.get("rewritten_query"),
        answer=result.get("answer", ""),
        sources=[RAGSource(**s) for s in (result.get("sources") or [])],
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        rewrite_attempts=result.get("rewrite_attempts", 0),
    )
