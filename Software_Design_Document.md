# Software Design Document (SDD)
## Newsgroups Semantic Search — RAG Pipeline

**Version:** 1.0  
**Author:** Perinba Athiban  
**Date:** June 2026

---

## 1. Introduction

### 1.1 Purpose
This document describes the architecture, component design, and data flow of the Newsgroups Semantic Search system. It is intended for developers who want to understand, extend, or deploy the system.

### 1.2 Scope
Covers all backend components (FastAPI, LangGraph RAG pipeline, ChromaDB, Semantic Cache) and the frontend (static HTML UI). Does not cover infrastructure provisioning.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
Browser / HTTP Client
        |
        v
  FastAPI (port 7860)
        |
   +----+--------------------+
   |                         |
   v                         v
Semantic Cache      LangGraph RAG Graph
(FCM Clustering)         |
   |              +------+---------------+
   |              v                      v
   |         ChromaDB              Groq LLM
   |       (Vector Store)     (llama-3.3-70b)
   |
   v
Cached Response
```

### 2.2 Component Overview

| Component | File(s) | Responsibility |
|---|---|---|
| API Layer | `part4_api.py` | HTTP routing, request/response models, startup |
| RAG Pipeline | `rag/graph.py` | LangGraph state machine — query rewrite → retrieve → grade → generate |
| Retriever | `rag/retriever.py` | ChromaDB vector search |
| Prompts | `rag/prompts.py` | All LLM prompt templates |
| Config | `rag/config.py` | Environment variables, model names, paths |
| Semantic Cache | `part3_cache.py` | FCM-based cache — lookup and store |
| Data Prep | `part1_prepare.py` | Downloads dataset, builds ChromaDB |
| FCM Training | `part2_clustering.py` | Trains Fuzzy C-Means model for cache |
| Evaluation | `part5_evaluate.py` | Offline RAGAS evaluation CLI |
| Frontend | `static/index.html` | Single-page UI |

---

## 3. Component Design

### 3.1 LangGraph RAG Pipeline (`rag/graph.py`)

The pipeline is a directed state machine with the following nodes:

```
[rewrite_query]
      |
      v
  [retrieve]
      |
      v
[grade_documents]
      |
   +--+---------------------------+
   | has_relevant_docs?            |
   v yes                           v no (retry up to 3x)
[generate_answer]           [rewrite_query]
      |
      v
   [END]
```

**State fields:**

| Field | Type | Description |
|---|---|---|
| `query` | str | Original user query |
| `rewritten_query` | str | LLM-rewritten query |
| `rewrite_attempts` | int | Number of rewrite cycles |
| `documents` | list | Raw retrieved documents |
| `filtered_documents` | list | Graded (relevant) documents |
| `answer` | str | Final generated answer |
| `sources` | list | Source metadata for the response |
| `input_tokens` | int | Cumulative LLM input tokens |
| `output_tokens` | int | Cumulative LLM output tokens |
| `has_relevant_docs` | bool | Whether grading found relevant docs |

### 3.2 Retriever (`rag/retriever.py`)

- Uses `chromadb.PersistentClient` directly (no LangChain wrapper) to avoid `_type` compatibility issues.
- Returns top-8 documents by cosine similarity.
- Collection: `newsgroups` — 18,159 documents with metadata (`category`, `dominant_cluster`, `cluster_ids`).

### 3.3 Semantic Cache (`part3_cache.py`)

- Stores query embeddings and results as `CacheEntry` objects.
- On lookup: computes cosine similarity between incoming embedding and all stored embeddings.
- Cache hit threshold: 0.85 similarity.
- Cluster assignment uses the pre-trained FCM model (`embeddings/fcm_model.pkl`).
- In-memory store (resets on restart) — intentional for stateless HF Spaces deployment.

### 3.4 FastAPI Layer (`part4_api.py`)

- **Lifespan startup:** loads embedding model → ChromaDB → Semantic Cache → RAG graph.
- ChromaDB is copied from `embeddings/chroma_db` to `/tmp/chroma_db` at startup (HF Spaces has a read-only app filesystem).
- All heavy state is held in a single `AppState` singleton.

### 3.5 Evaluation (`part5_evaluate.py`, `/rag/evaluate`)

- Stubs `langchain_community.chat_models.vertexai` and `tensorflow` before ragas import (ragas 0.4.x pulls these in transitively).
- Metrics used:
  - `Faithfulness` — answer only claims things supported by context
  - `AnswerRelevancy` — answer addresses the question
  - `LLMContextPrecisionWithoutReference` — retrieved chunks are relevant

---

## 4. Data Design

### 4.1 ChromaDB Collection Schema

| Field | Type | Description |
|---|---|---|
| `id` | str | Document ID (`doc_0` … `doc_18158`) |
| `document` | str | Raw post text (truncated to 500 chars) |
| `embedding` | float[384] | `all-MiniLM-L6-v2` embedding |
| `category` | str | Newsgroup category (e.g., `sci.space`) |
| `dominant_cluster` | int | Primary FCM cluster |
| `cluster_ids` | str | All cluster assignments (comma-separated) |

### 4.2 API Request / Response Models

**POST `/rag/query`**
```json
// Request
{ "query": "What did people think about the space shuttle?" }

// Response
{
  "query": "...",
  "rewritten_query": "...",
  "answer": "...",
  "sources": [
    { "index": 0, "category": "sci.space", "score": 0.85, "cluster": 3, "snippet": "..." }
  ],
  "input_tokens": 1400,
  "output_tokens": 290,
  "rewrite_attempts": 1
}
```

**POST `/rag/evaluate`**
```json
// Request
{ "queries": ["question 1", "question 2"] }

// Response
{
  "scores": [
    { "query": "...", "answer": "...", "faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.75 }
  ],
  "aggregate": { "faithfulness": 0.9, "answer_relevancy": 0.85, "llm_context_precision_without_reference": 0.75 }
}
```

---

## 5. Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| API Framework | FastAPI + uvicorn | latest |
| RAG Orchestration | LangGraph | latest |
| LLM | Groq (`llama-3.3-70b-versatile`) | — |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | — |
| Vector Store | ChromaDB | >= 1.0.0 |
| Evaluation | RAGAS | >= 0.4.0 |
| Clustering | `skfuzzy` (Fuzzy C-Means) | latest |
| Containerization | Docker (multi-stage) | — |
| Hosting | Hugging Face Spaces | Docker SDK |
| Frontend | Vanilla HTML/CSS/JS | — |

---

## 6. Deployment Design

### 6.1 Docker Multi-Stage Build

**Stage 1 (builder):** Installs all Python dependencies into `/opt/venv`.  
**Stage 2 (runtime):** Copies only the venv and app code — keeps final image lean.

The image does **not** include ChromaDB data — it must be pre-built and placed in `embeddings/` before `docker build`.

### 6.2 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |

### 6.3 Port
The application listens on port **7860** (Hugging Face Spaces requirement).

### 6.4 Filesystem Note
HF Spaces mounts the app directory as read-only after startup. ChromaDB requires write access, so the startup routine copies it to `/tmp/chroma_db`.

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| Direct `chromadb` client instead of LangChain wrapper | Avoids `_type` deserialization errors from chromadb 1.x vs 0.5.x schema changes |
| In-memory semantic cache | Stateless restarts suit ephemeral HF Spaces containers; persistent cache would require a writable volume |
| LangGraph state machine | Makes pipeline steps explicit, debuggable, and easy to extend with new nodes |
| Vertexai + TensorFlow stubs | ragas 0.4.x imports these transitively; stubbing avoids 2 GB of unnecessary dependencies |
| Multi-stage Docker build | Reduces final image size from ~3 GB to ~1.5 GB by leaving build tools in the builder stage |
