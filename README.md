# Semantic RAG Search &nbsp;[![CI](https://github.com/PerinbaBuilds/Semantic-RAG-Search/actions/workflows/ci.yml/badge.svg)](https://github.com/PerinbaBuilds/Semantic-RAG-Search/actions/workflows/ci.yml)

Ask a question in plain English and get a smart, cited answer pulled from 18,000+ real Usenet discussions from 1993.

**Live demo:** [perinbabuilds-newsgroups-search.hf.space](https://perinbabuilds-newsgroups-search.hf.space)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat-square&logo=databricks&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_LLM-F55036?style=flat-square&logo=lightning&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![RAGAS](https://img.shields.io/badge/RAGAS-6E56CF?style=flat-square&logo=probot&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face_Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## Why this exists

Traditional keyword search falls apart on messy, decades-old text — you can't match words when people phrase the same idea ten different ways. I wanted to build a full Retrieval-Augmented Generation (RAG) pipeline end to end, not just a demo: something that understands a question, finds the genuinely relevant posts, filters out the noise, and writes a grounded answer with sources. The 20 Newsgroups corpus was the perfect messy, real-world dataset to prove it on.

---

## How It Works

The RAG pipeline runs as a **LangGraph** state machine — each step is a node, so it's easy to extend or debug:

1. **Query Rewriter** — on a retry, the LLM rewrites your query for better recall.
2. **Semantic Retriever** — embeds the query and pulls the top 8 similar posts from the vector store.
3. **Document Grader** — the LLM keeps only the posts that are actually relevant.
4. **Router** — no relevant posts? Loop back and rewrite (up to 2 attempts). Otherwise, generate.
5. **Answer Generator** — the LLM writes an answer grounded strictly in the retrieved posts, and returns them as sources.

For the full component breakdown, data flow, and design tradeoffs, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Features

- **Natural-language answers with sources** — ask in plain English, get a grounded answer that cites the posts it came from.
- **Self-correcting retrieval** — if the first search returns weak results, the pipeline rewrites the query and tries again.
- **Relevance grading** — an LLM filters retrieved posts before answering, so noise never reaches the generation step.
- **Semantic cache** — near-duplicate queries reuse past results instead of re-calling the LLM, accelerated by query clustering.
- **Live quality evaluation** — score the pipeline with RAGAS (faithfulness, answer relevancy, context precision) from the UI or CLI — no ground-truth labels needed.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, fast to iterate, serves the UI and JSON from one process |
| Pipeline | LangGraph | Models the rewrite→retrieve→grade→generate retry loop as an inspectable state machine |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, fast, runs on CPU — good enough semantic quality for short posts |
| Vector DB | ChromaDB (18,159 docs) | Persistent local vector store, zero external service to run |
| LLM | Groq — `llama-3.3-70b-versatile` | Fast enough to call several times per request, and free-tier friendly |
| Semantic cache | Custom Fuzzy C-Means clustering | Cluster routing turns cache lookups from O(N) into ~O(N/K) |
| Evaluation | RAGAS | Measures quality with no ground-truth labels — the dataset has none |
| Hosting | Hugging Face Spaces (Docker) | Free Docker runtime for the live demo |

---

## Architecture

```
Browser UI ──HTTP──► FastAPI ──┬──► Semantic Cache ──► FCM cluster routing
                               │
                               └──► LangGraph pipeline ──► Retriever ──► ChromaDB
                                         │                                  ▲
                                         └──► Groq LLM (rewrite/grade/gen)   │
                                                                    SentenceTransformer
```

- **FastAPI** — HTTP surface, request validation, startup wiring (`part4_api.py`)
- **LangGraph pipeline** — rewrite → retrieve → grade → generate (`rag/graph.py`)
- **Retriever** — query embedding + ChromaDB vector search (`rag/retriever.py`)
- **Semantic cache** — cluster-accelerated nearest-neighbour cache over past queries (`part3_cache.py`)

**The one interesting decision:** the semantic cache doesn't scan every stored query. It uses the Fuzzy C-Means clusters to route a lookup to just the top-2 relevant cluster buckets — a roughly K-fold speedup, traded against the occasional false miss when a query sits on a cluster boundary. The full reasoning (including why the similarity threshold is 0.85) is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Getting Started

```bash
# 1. Clone and install
git clone https://github.com/PerinbaBuilds/Semantic-RAG-Search
cd Semantic-RAG-Search
pip install -r requirements.txt

# 2. Configure your environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY (free key: https://console.groq.com/keys)

# 3. Build the vector database + cluster model (one-time, ~10 min)
python scripts/part1_prepare.py
python scripts/part2_clustering.py

# 4. Run the server
uvicorn part4_api:app --host 0.0.0.0 --port 7860
# Visit http://localhost:7860
```

**Requirements:** Python 3.12+, a free [Groq API key](https://console.groq.com/keys), and ~2 GB disk for the corpus + embeddings.

### Run with Docker

```bash
cp .env.example .env   # set GROQ_API_KEY
docker compose up --build
# Visit http://localhost:7860
```

The build step (`part1`/`part2`) populates `embeddings/`, which the container mounts read-only.

---

## Usage

Ask a question and get an answer with sources:

```bash
curl -X POST http://localhost:7860/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What did people think about the space shuttle program?"}'
```

Or run pure semantic search (no LLM), score the pipeline, or check health:

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Opens the web UI |
| `/rag/query` | POST | Ask a question, get an AI answer with sources |
| `/rag/evaluate` | POST | Score the pipeline with RAGAS metrics |
| `/query` | POST | Pure semantic search, no LLM generation |
| `/health` | GET | Check which components loaded |
| `/cache/stats` | GET | See cache hit rate |

Run the evaluation from the CLI (or click **Run Evaluation** in the UI):

```bash
python scripts/part5_evaluate.py
```

---

## Known Limitations / What I'd Do Differently

- **The semantic cache is in-memory** — it resets on restart. In production I'd back it with Redis or a persistent store so warm state survives deploys.
- **No automated test suite yet.** CI currently byte-compiles and lints the code; the retry loop and cache routing deserve real unit tests.
- **Per-document grading is a serial LLM call per post**, which adds latency. Batching the grading into a single prompt (or a cheaper cross-encoder re-ranker) would be faster and cheaper.
- **Corpus is frozen at build time.** There's no incremental ingestion — adding documents means re-running the full prep pipeline.
- **Grading falls open, not closed** — if the grader LLM errors, the document is kept rather than dropped. That protects recall but could let a weak post through on a bad API day.

---

## Dataset

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) — a widely used NLP benchmark of ~18,000 Usenet posts from 1993 across 20 categories (sci.space, talk.politics.guns, alt.atheism, rec.sport.baseball, sci.crypt, and more). Downloaded automatically via scikit-learn; not stored in this repo.

---

## License

MIT
