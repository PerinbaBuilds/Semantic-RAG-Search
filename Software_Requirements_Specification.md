# Software Requirements Specification (SRS)
## Newsgroups Semantic Search — RAG Pipeline

**Version:** 1.0  
**Author:** Perinba Athiban  
**Date:** June 2026

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Newsgroups Semantic Search system — a Retrieval-Augmented Generation (RAG) pipeline that answers plain-English questions using 18,000+ real Usenet forum posts from 1993.

### 1.2 Scope
The system allows users to ask questions through a web UI or REST API and receive AI-generated answers grounded in the retrieved forum posts, along with cited sources and evaluation metrics.

### 1.3 Definitions

| Term | Definition |
|---|---|
| RAG | Retrieval-Augmented Generation — a pipeline that retrieves relevant documents before generating an answer |
| ChromaDB | Vector database used to store and query document embeddings |
| RAGAS | Reference-free evaluation framework for RAG pipelines |
| FCM | Fuzzy C-Means — clustering algorithm used for the semantic cache |
| LLM | Large Language Model (Groq-hosted `llama-3.3-70b-versatile`) |
| HF Spaces | Hugging Face Spaces — cloud platform used for deployment |

---

## 2. Overall Description

### 2.1 Product Perspective
The system is a standalone web application deployed on Hugging Face Spaces. It exposes a FastAPI backend and a static HTML frontend. The pipeline runs entirely server-side; users interact via browser or HTTP client.

### 2.2 User Classes

| User | Description |
|---|---|
| End User | Asks questions via the web UI |
| Developer | Calls the REST API directly (`/rag/query`, `/query`) |
| Evaluator | Runs RAGAS evaluation via UI or `part5_evaluate.py` |

### 2.3 Operating Environment
- **Hosting:** Hugging Face Spaces (Docker, 2 vCPU, 16 GB RAM, free tier)
- **Runtime:** Python 3.12, uvicorn
- **LLM Provider:** Groq (free tier — 100k tokens/day limit)
- **Port:** 7860 (HF Spaces requirement)

---

## 3. Functional Requirements

### FR-01 — Semantic Search
- The system shall accept a plain-English query and return the top-8 most semantically similar documents from ChromaDB.
- Similarity shall be computed using cosine distance on `all-MiniLM-L6-v2` embeddings.

### FR-02 — RAG Query Pipeline
- The system shall rewrite the user query using an LLM before retrieval.
- The system shall grade retrieved documents for relevance before generating an answer.
- The system shall generate a grounded answer using only the filtered documents.
- The response shall include: answer text, source snippets, token counts, and rewrite attempt count.

### FR-03 — Semantic Cache
- The system shall cache query embeddings and their results using Fuzzy C-Means clustering.
- Cache hits shall be returned without invoking the LLM.
- The cache shall expose hit/miss statistics via `/cache/stats`.

### FR-04 — RAGAS Evaluation
- The system shall support evaluation of up to 10 queries per request via `/rag/evaluate`.
- Evaluation shall return per-query and aggregate scores for:
  - Faithfulness
  - Answer Relevancy
  - Context Precision (without reference)
- Evaluation results shall be displayed as metric bars in the web UI.

### FR-05 — Health Check
- The system shall expose a `/health` endpoint reporting readiness of the embedding model, ChromaDB, FCM cache, and RAG graph.

### FR-06 — Web UI
- The system shall serve a single-page HTML UI with:
  - A query input and "Ask" button
  - Answer display with source cards
  - RAGAS evaluation panel with metric bars

---

## 4. Non-Functional Requirements

### NFR-01 — Performance
- RAG query response time should be under 10 seconds under normal LLM load.
- Cache hits should respond in under 500 ms.

### NFR-02 — Availability
- The system should be available 24/7 on HF Spaces free tier (best-effort).
- ChromaDB shall be copied to `/tmp` at startup to work around HF Spaces read-only filesystem.

### NFR-03 — Security
- The system shall not expose the `GROQ_API_KEY` in any API response.
- The Docker image shall run as a non-root user (`appuser`).

### NFR-04 — Scalability
- The system is designed for single-instance deployment. Horizontal scaling is not a requirement.

### NFR-05 — Portability
- The system shall run identically in Docker locally and on HF Spaces.
- Local setup shall require only `pip install -r requirements.txt` and a valid `.env` file.

---

## 5. External Interface Requirements

### 5.1 REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/rag/query` | POST | RAG pipeline — returns AI answer with sources |
| `/rag/evaluate` | POST | RAGAS evaluation for up to 10 queries |
| `/query` | POST | Pure semantic search, no LLM |
| `/health` | GET | System readiness check |
| `/cache/stats` | GET | Cache hit/miss statistics |

### 5.2 External Services
- **Groq API** — LLM inference (requires `GROQ_API_KEY`)
- **Hugging Face Hub** — Embedding model download at first run

---

## 6. Constraints
- Groq free tier limits the system to 100,000 tokens per day.
- RAGAS evaluation is compute-intensive; max 10 queries per request is enforced.
- ChromaDB must be pre-built locally (`python part1_prepare.py`) before Docker image build.
