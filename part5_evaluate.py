"""
Part 5: RAGAS evaluation of the RAG pipeline (ragas >= 0.4).

Metrics (no ground truth required):
  - Faithfulness                        : answer grounded in retrieved context?
  - AnswerRelevancy                     : answer relevant to the question?
  - LLMContextPrecisionWithoutReference : retrieved chunks relevant to the question?

Usage:
    python part5_evaluate.py
    python part5_evaluate.py --queries "question 1" "question 2"
"""

import sys
import unittest.mock
sys.modules.setdefault("langchain_community.chat_models.vertexai", unittest.mock.MagicMock())
sys.modules.setdefault("tensorflow", unittest.mock.MagicMock())
sys.modules.setdefault("tensorflow.python", unittest.mock.MagicMock())
sys.modules.setdefault("tensorflow.python.framework", unittest.mock.MagicMock())

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import sys

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag.graph import build_rag_graph
from rag.config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL

DEFAULT_QUERIES = [
    "What did people think about the space shuttle program?",
    "How did people discuss encryption and privacy in 1993?",
    "What were common opinions about gun control?",
    "What did people think about the Middle East conflict?",
    "How did people discuss atheism and religion?",
    "What were early discussions about computer graphics?",
    "What did people say about baseball in 1993?",
    "What were peoples views on the Clinton administration?",
]


async def run_query(graph, query: str) -> dict:
    result = await graph.ainvoke({
        "query": query, "rewritten_query": None, "rewrite_attempts": 0,
        "documents": [], "filtered_documents": [], "answer": "",
        "sources": [], "input_tokens": 0, "output_tokens": 0, "has_relevant_docs": False,
    })
    contexts = [s["snippet"] for s in (result.get("sources") or [])]
    return {
        "question": query,
        "answer": result.get("answer", ""),
        "contexts": contexts or ["no context retrieved"],
    }


async def collect_results(queries: list[str]) -> list[dict]:
    graph = build_rag_graph()
    rows = []
    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q[:70]}")
        row = await run_query(graph, q)
        print(f"         -> {row['answer'][:100]}...")
        rows.append(row)
    return rows


def score_with_ragas(rows: list[dict]) -> "object":
    llm = LangchainLLMWrapper(ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]
    return evaluate(dataset=dataset, metrics=metrics)


def print_report(result, rows: list[dict]) -> None:
    df = result.to_pandas()
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"] if c in df.columns]

    print("\n" + "=" * 68)
    print("  RAGAS Evaluation Report")
    print("=" * 68)
    for i, row_data in df.iterrows():
        q = rows[i]["question"][:65]
        print(f"\n  Q: {q}")
        for col in metric_cols:
            val = float(row_data.get(col, float("nan")))
            filled = int(val * 20)
            bar = "█" * filled + "░" * (20 - filled)
            label = col.replace("llm_context_precision_without_reference", "context_precision")
            print(f"     {label:22s}  {bar}  {val:.3f}")

    print("\n" + "-" * 68)
    print("  Aggregate averages:")
    for col in metric_cols:
        mean = float(df[col].mean())
        filled = int(mean * 20)
        bar = "█" * filled + "░" * (20 - filled)
        label = col.replace("llm_context_precision_without_reference", "context_precision")
        print(f"     {label:22s}  {bar}  {mean:.3f}")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the newsgroups RAG pipeline")
    parser.add_argument("--queries", nargs="+", help="Custom queries to evaluate")
    args = parser.parse_args()

    queries = args.queries if args.queries else DEFAULT_QUERIES

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)

    print(f"=== RAGAS Evaluation  ({len(queries)} queries) ===\n")
    print("Step 1/2  Running queries through RAG pipeline...")
    rows = asyncio.run(collect_results(queries))

    print("\nStep 2/2  Scoring with RAGAS (faithfulness · answer_relevancy · context_precision)...")
    result = score_with_ragas(rows)

    print_report(result, rows)


if __name__ == "__main__":
    main()
