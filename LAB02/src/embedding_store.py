"""
embedding_store.py
-------------------
Stage 3 (Embedding) and Stage 4 (Vector Store / Indexing) of the pipeline.

REAL PROBLEMS FOUND DURING DEVELOPMENT:

(a) Rate limits / transient failures on the embeddings API.
    Symptom: calling client.embeddings.create() for many chunks in a tight
    loop occasionally raised openai.RateLimitError or a transient
    APIConnectionError, which crashed the whole indexing run and wasted all
    the embeddings computed so far.
    Fix: `get_embeddings()` batches requests (EMBEDDING_BATCH_SIZE at a
    time, not one call per chunk) and retries failed batches with
    exponential backoff (MAX_RETRIES, RETRY_BASE_DELAY in config.py)
    instead of letting one flaky call kill the entire indexing job.

(b) Re-indexing the same document twice created duplicate vectors.
    Symptom: running `python src/pipeline.py --build` a second time (e.g.
    after adding one new file) re-embedded and re-inserted every chunk
    again, doubling memory usage and causing the same passage to be
    retrieved multiple times in the top-k results, crowding out other
    relevant chunks.
    Fix: `VectorStore.add()` computes a SHA-256 hash of each chunk's text
    and skips any chunk whose hash is already present in the store. Note
    this only catches *exact* duplicates -- data/docs/leave_policy.txt and
    leave_policy_copy_2025.txt are near-duplicates with slightly different
    wording, so they are NOT caught by this hash check. That's a known,
    documented limitation (see README.md Section 4) rather than a claim
    that this fully solves duplicate content.

(c) Index was rebuilt from scratch every time the program started.
    Symptom: every `python main.py` run re-embedded all documents,
    burning API quota and taking noticeably longer to start, even when
    no documents had changed.
    Fix: `VectorStore.save()` / `VectorStore.load()` persist the FAISS
    index and chunk metadata to disk (data/index/). `pipeline.py` only
    rebuilds when the index files are missing or `--rebuild` is passed.
"""

import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional

import numpy as np
import faiss
from openai import OpenAI

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("embedding_store")


def _client() -> OpenAI:
    """Return a client pointed at either OpenAI or a local Ollama server.

    Ollama exposes an OpenAI-compatible API at /v1, so the same `openai`
    package works for both providers -- we just swap base_url/api_key
    based on config.LLM_PROVIDER.
    """
    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key, or set LLM_PROVIDER=ollama to run for free with "
            "no API key."
        )
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Embed a list of texts, batching requests and retrying on failure.

    Returns an (N, EMBEDDING_DIM) float32 numpy array, in the same order
    as `texts`.
    """
    client = _client()
    all_vectors: List[List[float]] = []

    for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + config.EMBEDDING_BATCH_SIZE]
        attempt = 0
        while True:
            try:
                response = client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=batch,
                )
                all_vectors.extend([d.embedding for d in response.data])
                break
            except Exception as e:  # openai raises several distinct error
                                      # types (RateLimitError, APIConnection
                                      # Error, APITimeoutError, ...); we
                                      # retry all of them the same way.
                attempt += 1
                if attempt > config.MAX_RETRIES:
                    logger.error(
                        f"Giving up on batch {start}-{start + len(batch)} "
                        f"after {config.MAX_RETRIES} retries: {e}"
                    )
                    raise
                delay = config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Embedding batch {start}-{start + len(batch)} failed "
                    f"({e}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{config.MAX_RETRIES})"
                )
                time.sleep(delay)

    return np.array(all_vectors, dtype="float32")


class VectorStore:
    """Thin wrapper around a FAISS flat index plus chunk metadata.

    Uses inner-product search over L2-normalized vectors, which is
    mathematically equivalent to cosine similarity -- important because
    OpenAI embeddings are meant to be compared with cosine similarity, not
    raw Euclidean distance.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []   # parallel array to index vectors
        self._seen_hashes = set()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms

    def add(self, chunk_records: List[Dict]):
        """Embed and add chunk_records, skipping exact-duplicate text.

        chunk_records: list of {"chunk_id", "source", "text"}
        """
        new_records = []
        for rec in chunk_records:
            h = self._hash(rec["text"])
            if h in self._seen_hashes:
                logger.info(f"Skipping exact-duplicate chunk: {rec['chunk_id']}")
                continue
            self._seen_hashes.add(h)
            new_records.append(rec)

        if not new_records:
            logger.info("No new (non-duplicate) chunks to add.")
            return

        logger.info(f"Embedding {len(new_records)} new chunk(s)...")
        vectors = get_embeddings([r["text"] for r in new_records])
        vectors = self._normalize(vectors)
        self.index.add(vectors)
        self.metadata.extend(new_records)
        logger.info(f"Vector store now holds {self.index.ntotal} chunk(s).")

    def search(self, query: str, k: int = config.TOP_K) -> List[Dict]:
        """Embed the query and return the top-k most similar chunks.

        Each result dict includes a "score" field (cosine similarity,
        higher = more similar) so callers can apply a relevance threshold.
        """
        if self.index.ntotal == 0:
            return []
        q_vec = get_embeddings([query])
        q_vec = self._normalize(q_vec)
        scores, indices = self.index.search(q_vec, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = dict(self.metadata[idx])
            rec["score"] = float(score)
            results.append(rec)
        return results

    def save(self, index_path: str = config.FAISS_INDEX_PATH,
              metadata_path: str = config.METADATA_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index ({self.index.ntotal} vectors) to {index_path}")

    @classmethod
    def load(cls, index_path: str = config.FAISS_INDEX_PATH,
              metadata_path: str = config.METADATA_PATH) -> Optional["VectorStore"]:
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            return None
        store = cls()
        store.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        store._seen_hashes = {cls._hash(r["text"]) for r in store.metadata}
        logger.info(f"Loaded existing index with {store.index.ntotal} vectors from {index_path}")
        return store
