"""
Part 1: Corpus Preparation, Embedding, and Vector Database Setup
================================================================

Design decisions:
- Embedding model: `all-MiniLM-L6-v2` from sentence-transformers.
  Rationale: 384-dim embeddings are compact enough to keep in memory for a ~20k doc corpus
  while still capturing rich semantic content. Larger models (e.g. all-mpnet-base-v2) give
  marginally better quality but 2-3× slower encoding — not worth it for this scale.
  
- Vector store: ChromaDB (local persistence, no server required).
  Rationale: We need filtered retrieval by cluster later. ChromaDB supports metadata filtering
  out of the box. FAISS would be faster but requires custom metadata wrangling.
  
- Text cleaning strategy (deliberate choices):
  * KEEP: Subject line — it's the densest signal in a newsgroup post.
  * KEEP: First ~300 tokens of body — diminishing returns after that; headers dominate.
  * DISCARD: Email headers (From:, Message-ID:, X-*, NNTP-*, etc.) — pure noise for semantic search.
  * DISCARD: Quoted reply blocks (lines starting with ">") — duplicate content that inflates
    similarity between unrelated posts that happen to quote the same message.
  * DISCARD: PGP/signature blocks, MIME boundaries.
  * DISCARD: Posts with < 20 tokens after cleaning — too little signal to embed meaningfully.
  * DISCARD: Near-duplicate posts (cosine sim > 0.98 after embedding) — 20 Newsgroups contains
    cross-posted articles; duplicates corrupt cluster centroids.
"""

import re
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
EMBEDDINGS_DIR = Path("embeddings")
CHROMA_DIR = Path("embeddings/chroma_db")
COLLECTION_NAME = "newsgroups"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_TOKENS = 300          # truncate body at ~300 whitespace-split tokens
MIN_TOKENS = 20           # discard posts shorter than this
DEDUP_THRESHOLD = 0.98    # cosine similarity above which we consider docs duplicates

# Headers to strip (from RFC 1036 + NNTP-specific)
HEADER_PATTERNS = re.compile(
    r"^(From|Newsgroups|Subject|Date|Message-ID|References|X-[A-Za-z\-]+|"
    r"NNTP-Posting-Host|Organization|Lines|Path|Distribution|Xref|"
    r"Return-Path|Received|Reply-To|Mime-Version|Content-Type|"
    r"Content-Transfer-Encoding|In-Reply-To|Followup-To|Summary|Keywords|"
    r"Approved):.*",
    re.IGNORECASE,
)

QUOTE_LINE = re.compile(r"^\s*>.*")           # quoted reply lines
SIG_SEP   = re.compile(r"^--\s*$")           # sig separator
PGP_BLOCK = re.compile(r"^-{5}BEGIN PGP")    # PGP header


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_post(raw_text: str) -> str:
    """
    Strip RFC 1036 headers, quoted replies, signatures, and PGP blocks.
    Preserve Subject: value as a synthetic first line (highest-signal field).
    """
    lines = raw_text.splitlines()
    
    # Extract subject before stripping headers
    subject = ""
    for line in lines:
        m = re.match(r"^Subject:\s*(.+)", line, re.IGNORECASE)
        if m:
            # Clean Re:/Fwd: prefixes — they add noise without semantic value
            subject = re.sub(r"^(Re|Fwd|Fw):\s*", "", m.group(1), flags=re.IGNORECASE).strip()
            break

    # State machine: skip header block (everything before first blank line)
    in_header = True
    in_sig    = False
    body_lines = []

    for line in lines:
        if in_header:
            if line.strip() == "":
                in_header = False
            continue

        # Once we're in body:
        if SIG_SEP.match(line) or PGP_BLOCK.match(line):
            in_sig = True
        if in_sig:
            continue
        if QUOTE_LINE.match(line):
            continue

        body_lines.append(line)

    body = " ".join(body_lines)
    # Collapse whitespace
    body = re.sub(r"\s+", " ", body).strip()

    # Truncate to MAX_TOKENS whitespace tokens
    tokens = body.split()
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    body = " ".join(tokens)

    # Prepend subject as a synthetic sentence
    if subject:
        return f"{subject}. {body}".strip()
    return body


def is_meaningful(text: str) -> bool:
    """Return True if the post has enough signal to be worth embedding."""
    return len(text.split()) >= MIN_TOKENS


# ---------------------------------------------------------------------------
# Dataset loading  (works with sklearn's fetch_20newsgroups OR raw files)
# ---------------------------------------------------------------------------

def load_dataset_sklearn(subset: str = "all") -> list[dict]:
    """
    Primary loader: use sklearn's bundled 20 newsgroups corpus.
    subset = 'train' | 'test' | 'all'
    
    We intentionally load WITHOUT removing headers/footers ourselves
    (remove=['headers','footers','quotes']) because we want fine-grained
    control over what gets cleaned — sklearn's removal is too aggressive
    and would discard the Subject line we want to keep.
    """
    from sklearn.datasets import fetch_20newsgroups

    logger.info(f"Loading 20 Newsgroups ({subset}) via sklearn...")
    data = fetch_20newsgroups(
        subset=subset,
        remove=(),          # we do our own cleaning
        shuffle=False,
    )

    docs = []
    for idx, (raw, label_id) in enumerate(zip(data.data, data.target)):
        cleaned = clean_post(raw)
        if not is_meaningful(cleaned):
            continue
        docs.append({
            "id":           hashlib.md5(raw.encode()).hexdigest()[:12],
            "raw_idx":      idx,
            "category":     data.target_names[label_id],
            "label_id":     int(label_id),
            "text":         cleaned,
        })

    logger.info(f"Retained {len(docs):,} / {len(data.data):,} documents after cleaning")
    return docs


def load_dataset_files(root_dir: str) -> list[dict]:
    """
    Fallback loader for raw downloaded dataset directory.
    Expects directory structure: <root>/<category>/<filename>
    """
    root = Path(root_dir)
    docs = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for post_file in category_dir.iterdir():
            if not post_file.is_file():
                continue
            try:
                raw = post_file.read_text(errors="replace")
            except Exception:
                continue
            cleaned = clean_post(raw)
            if not is_meaningful(cleaned):
                continue
            docs.append({
                "id":       hashlib.md5(raw.encode()).hexdigest()[:12],
                "raw_idx":  None,
                "category": category,
                "label_id": -1,
                "text":     cleaned,
            })
    logger.info(f"Retained {len(docs):,} documents from {root_dir}")
    return docs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_documents(docs: list[dict], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Encode all cleaned texts using sentence-transformers.
    Returns float32 array of shape (N, D).
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    logger.info(f"Encoding {len(texts):,} documents...")

    # batch_size=64 is a sweet spot: large enough for GPU throughput, small enough
    # to avoid OOM on a machine with 8 GB VRAM when using a larger model.
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → dot product == cosine similarity
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Near-duplicate removal
# ---------------------------------------------------------------------------

def remove_near_duplicates(docs: list[dict], embeddings: np.ndarray,
                            threshold: float = DEDUP_THRESHOLD) -> tuple[list[dict], np.ndarray]:
    """
    Greedy near-duplicate removal.
    Walk through docs in order; if a doc's max cosine similarity to any
    already-accepted doc exceeds `threshold`, discard it.
    
    O(N²) in the worst case but fast enough for 18k docs in practice (~2s on CPU).
    For larger corpora, use an ANN index (e.g. HNSW) to find near-neighbours.
    """
    logger.info(f"Running near-duplicate removal (threshold={threshold})...")
    n = len(docs)
    kept_idx = [0]
    kept_emb = embeddings[[0]]   # (1, D)

    for i in tqdm(range(1, n), desc="Dedup"):
        # cosine similarities to all kept embeddings (already L2-normalised → dot product)
        sims = kept_emb @ embeddings[i]      # shape (K,)
        if sims.max() < threshold:
            kept_idx.append(i)
            kept_emb = np.vstack([kept_emb, embeddings[i]])

    logger.info(f"Kept {len(kept_idx):,} / {n:,} after dedup")
    kept_docs = [docs[i] for i in kept_idx]
    kept_emb  = embeddings[kept_idx]
    return kept_docs, kept_emb


# ---------------------------------------------------------------------------
# Vector store (ChromaDB)
# ---------------------------------------------------------------------------

def build_vector_store(docs: list[dict], embeddings: np.ndarray,
                        chroma_dir: str = str(CHROMA_DIR)) -> None:
    """
    Persist documents + embeddings in ChromaDB with metadata for filtered retrieval.
    
    Metadata stored per document:
      - category:      original 20-news label (useful for evaluation)
      - label_id:      integer label
      - cluster_ids:   comma-separated top-3 soft cluster ids (added later by Part 2)
      - cluster_probs: comma-separated top-3 soft cluster probabilities
    """
    import chromadb

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)

    # Drop existing collection if rerunning
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # Use cosine distance; embeddings are already L2-normalised so this
        # is equivalent to inner product, but ChromaDB's UI shows it as cosine.
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 500
    for start in tqdm(range(0, len(docs), BATCH), desc="Upserting to ChromaDB"):
        batch_docs  = docs[start:start+BATCH]
        batch_embs  = embeddings[start:start+BATCH]
        collection.upsert(
            ids         = [d["id"] for d in batch_docs],
            embeddings  = batch_embs.tolist(),
            documents   = [d["text"] for d in batch_docs],
            metadatas   = [
                {
                    "category":     d["category"],
                    "label_id":     d["label_id"],
                    "cluster_ids":  "",    # filled in by Part 2
                    "cluster_probs":"",
                }
                for d in batch_docs
            ],
        )
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' has {collection.count():,} entries")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_artifacts(docs: list[dict], embeddings: np.ndarray) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "embeddings.npy", embeddings)
    with open(EMBEDDINGS_DIR / "docs_metadata.json", "w") as f:
        # Save everything except the (large) text to keep metadata file lean
        meta = [{k: v for k, v in d.items()} for d in docs]
        json.dump(meta, f)
    logger.info(f"Saved embeddings ({embeddings.shape}) and metadata ({len(docs)} docs)")


def load_artifacts() -> tuple[list[dict], np.ndarray]:
    with open(EMBEDDINGS_DIR / "docs_metadata.json") as f:
        docs = json.load(f)
    embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")
    return docs, embeddings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="Path to raw 20news directory (optional; uses sklearn if omitted)")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip near-duplicate removal (faster, less clean)")
    args = parser.parse_args()

    # 1. Load
    if args.data_dir:
        docs = load_dataset_files(args.data_dir)
    else:
        docs = load_dataset_sklearn(subset="all")

    # 2. Embed
    embeddings = embed_documents(docs)

    # 3. Deduplicate
    if not args.skip_dedup:
        docs, embeddings = remove_near_duplicates(docs, embeddings)

    # 4. Persist
    save_artifacts(docs, embeddings)
    build_vector_store(docs, embeddings)

    logger.info("Part 1 complete. Run part2_clustering.py next.")