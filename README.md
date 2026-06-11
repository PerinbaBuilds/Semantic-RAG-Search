---
title: Newsgroups Search
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Newsgroups Semantic Search

A RAG-powered semantic search engine over the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) corpus (18,000+ Usenet posts from 1993). Ask natural language questions and get answers grounded in the original posts.

**Stack:** FastAPI · ChromaDB · Sentence Transformers · LangGraph · Groq (llama-3.3-70b-versatile) · Fuzzy C-Means clustering · Semantic cache

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The full dependency list is in `requirements.txt`. Key packages: `fastapi`, `uvicorn`, `chromadb`, `sentence-transformers`, `langchain`, `langgraph`, `groq`, `scikit-learn`.

### 2. Set your Groq API key

Copy `.env.example` to `.env` and add your free [Groq API key](https://console.groq.com):

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

### 3. Download dataset + build embeddings

```bash
python setup.py
```

This runs two steps:
- **part1_prepare.py** — downloads the 20 Newsgroups dataset via sklearn, cleans and embeds 18,000+ posts into ChromaDB (~5–10 min on CPU)
- **part2_clustering.py** — fits a Fuzzy C-Means model over the embeddings for cluster-accelerated cache lookups (~2 min)

### 4. Start the server

```bash
uvicorn part4_api:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              FastAPI  (part4_api.py)         │
│                                             │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │ Semantic     │    │   RAG Pipeline     │ │
│  │ Cache        │    │   (LangGraph)      │ │
│  │ (part3)      │    │                    │ │
│  │              │    │  analyze_query     │ │
│  │ FCM cluster  │    │       ↓            │ │
│  │ acceleration │    │  retrieve (Chroma) │ │
│  └──────────────┘    │       ↓            │ │
│                      │  grade_documents   │ │
│                      │       ↓            │ │
│                      │  generate (Groq)   │ │
│                      └────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Components

| File | Purpose |
|---|---|
| `part1_prepare.py` | Download, clean, embed corpus into ChromaDB |
| `part2_clustering.py` | Fuzzy C-Means clustering over PCA-reduced embeddings |
| `part3_cache.py` | Cluster-accelerated semantic cache (no external libs) |
| `part4_api.py` | FastAPI server — `/query`, `/rag/query`, `/health`, `/cache/stats` |
| `rag/graph.py` | LangGraph RAG pipeline (query rewrite → retrieve → grade → generate) |
| `rag/config.py` | Configuration (model, thresholds, paths) |
| `static/index.html` | Dark-mode search UI |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/rag/query` | POST | RAG query — returns answer + sources |
| `/query` | POST | Semantic search with cache |
| `/health` | GET | Health check |
| `/cache/stats` | GET | Cache hit/miss stats |
| `/cache` | DELETE | Flush cache |

### Example

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What did people think about the space shuttle?"}'
```

---

## Dataset

The [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/) is downloaded automatically via `sklearn.datasets.fetch_20newsgroups`. It is **not** committed to this repository. Run `python setup.py` to fetch and index it.

**Categories:** alt.atheism, comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, soc.religion.christian, talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc
