"""Prompt templates for the RAG pipeline."""

QUERY_REWRITE_SYSTEM = """You are a search query optimizer for a 20 Newsgroups corpus (1993 Usenet posts).
The corpus covers: alt.atheism, comp.*, misc.forsale, rec.*, sci.*, soc.*, talk.*
Rewrite the user query to maximise recall. Return ONLY the rewritten query."""
QUERY_REWRITE_HUMAN = "Original query: {query}"

GRADE_DOCUMENT_SYSTEM = """You are a relevance judge for a Usenet newsgroup search engine.
Respond with ONLY one word: YES or NO."""
GRADE_DOCUMENT_HUMAN = "Query: {query}\n\nDocument:\n{document}\n\nIs this document relevant to the query?"

RAG_GENERATION_SYSTEM = """You are an expert assistant answering questions based on Usenet newsgroup posts from 1993.
Answer directly based on the provided context. Cite newsgroup categories when available."""
RAG_GENERATION_HUMAN = "Question: {query}\n\nContext (retrieved newsgroup posts):\n{context}\n\nProvide a clear, accurate answer."

NO_DOCS_FALLBACK = (
    "I could not find relevant documents in the newsgroup corpus for your query. "
    "Try rephrasing with different keywords or a more specific topic."
)
