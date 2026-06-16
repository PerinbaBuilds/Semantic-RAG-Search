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
# Semantic RAG Search
Ask a question in plain English — get a smart, cited answer pulled from 18,000+ real online discussions.

Built as a full RAG (Retrieval-Augmented Generation) system: it finds the most relevant posts, filters out noise, and generates a grounded answer using an LLM — all in one pipeline.
A semantic search engine that lets you ask natural language questions about **1993 Usenet newsgroup discussions** and get intelligent, cited answers.

**Live Demo:** [perinbabuilds-newsgroups-search.hf.space](https://perinbabuilds-newsgroups-search.hf.space)

---

## The Problem It Solves

The 20 Newsgroups dataset contains 18,000+ text posts from 1993 covering topics like space exploration, politics, religion, sports, and technology. Traditional keyword search fails on this data — you need to understand meaning, not just match words.

This project builds a full RAG (Retrieval-Augmented Generation) pipeline that:
1. Understands what you're asking
2. Finds the most relevant posts from the corpus
3. Generates a grounded answer with sources

---

## How It Works

1. **Query Rewriter** — LLM rewrites your query for better search results
2. **Semantic Retriever** — Finds the top 8 similar posts using vector embeddings
3. **Document Grader** — LLM filters out irrelevant results
4. **Answer Generator** — LLM writes an answer grounded in the retrieved posts
5. **Answer + Sources** returned to you

Each step is a node in a **LangGraph** state machine, making the pipeline easy to extend or debug.
---
## Tech Stack
| What | How |
|---|---|
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB (18,159 documents) |
| LLM | Groq — `llama-3.3-70b-versatile` (free tier) |
| Pipeline | LangGraph |
| Semantic cache | Custom Fuzzy C-Means clustering |
| Evaluation | RAGAS (no ground truth needed) |
| API | FastAPI |
| Hosting | Hugging Face Spaces (Docker) |
---
## Key Features
- **Query rewriting** — automatically improves vague queries before searching
- **Document grading** — filters retrieved chunks for relevance before generating
- **Semantic cache** — avoids redundant LLM calls for similar queries using FCM clustering
- **RAGAS evaluation** — measures pipeline quality with faithfulness, answer relevancy, and context precision
- **Live eval UI** — run evaluation and see metric bars directly in the browser
---
## Project Files
| File | Purpose |
|---|---|
| `part1_prepare.py` | Downloads the dataset, generates embeddings, stores in ChromaDB |
| `part2_clustering.py` | Trains a Fuzzy C-Means model for the semantic cache |
| `part3_cache.py` | Semantic cache — returns cached answers for similar past queries |
| `part4_api.py` | FastAPI app — all HTTP endpoints |
| `part5_evaluate.py` | Runs RAGAS evaluation from the command line |
| `rag/graph.py` | The LangGraph RAG pipeline |
| `rag/retriever.py` | ChromaDB vector search |
| `rag/prompts.py` | All LLM prompt templates |
| `static/index.html` | The web UI |
---
## API Endpoints
| Endpoint | Method | What It Does |
|---|---|---|
| `/` | GET | Opens the web UI |
| `/rag/query` | POST | Ask a question, get an AI answer with sources |
| `/rag/evaluate` | POST | Score the pipeline with RAGAS metrics |
| `/query` | POST | Pure semantic search, no LLM generation |
| `/health` | GET | Check if all components are loaded |
| `/cache/stats` | GET | See cache hit rate |
**Example request:**
```bash
curl -X POST https://perinbabuilds-newsgroups-search.hf.space/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What did people think about the space shuttle program?"}'
Example response:

{
  "answer": "People had mixed views on the space shuttle...",
  "sources": [
    { "category": "sci.space", "score": 0.85, "snippet": "..." }
  ],
  "input_tokens": 1400,
  "output_tokens": 290,
  "rewrite_attempts": 1
}
Evaluation Metrics (RAGAS)
No ground truth needed — these metrics evaluate quality automatically:

Metric	What It Checks
Faithfulness	Does the answer only say things supported by the retrieved posts?
Answer Relevancy	Does the answer actually address the question asked?
Context Precision	Were the right posts retrieved for the question?
Run it locally:

python part5_evaluate.py
Or click Run Evaluation in the UI.

Local Setup
# 1. Clone and install
git clone https://github.com/PerinbaBuilds/Semantic-RAG-Search
cd Semantic-RAG-Search
pip install -r requirements.txt

# 2. Add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 3. Build the vector database (one-time, ~10 min)
python part1_prepare.py
python part2_clustering.py

# 4. Run the server
uvicorn part4_api:app --host 0.0.0.0 --port 7860
Visit http://localhost:7860

Dataset
20 Newsgroups — a widely used NLP benchmark dataset of Usenet posts from 1993.
18,000+ posts across 20 categories including sci.space, talk.politics.guns, alt.atheism, rec.sport.baseball, sci.crypt, and more.

Downloaded automatically via scikit-learn — not stored in this repo.
