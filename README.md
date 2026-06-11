# Semantic RAG Search

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
Your Question
│
▼
Query Rewriter → LLM rewrites query for better search results
│
▼
Semantic Retriever → Finds top 8 similar posts using vector embeddings
│
▼
Document Grader → LLM filters out irrelevant results
│
▼
Answer Generator → LLM writes an answer grounded in the retrieved posts
│
▼
Answer + Sources

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
