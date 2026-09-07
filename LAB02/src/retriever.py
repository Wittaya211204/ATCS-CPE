"""
retriever.py
------------
Stage 5 of the pipeline: Retrieval.

REAL PROBLEM FOUND DURING DEVELOPMENT:
Using pure vector (embedding) search, the query
    "ยกเลิก Pro Plan ต้องทำอย่างไร"
sometimes ranked a chunk about the mobile app above the actual cancellation
answer, because the embedding model placed general "account settings"
language close together in vector space even though only one chunk actually
answers the question. This is a classic vocabulary/semantic-gap symptom:
short, keyword-heavy Thai queries don't always embed close to the passage
that contains the literal answer, especially when the passage mixes Thai
and English (see data/docs/product_faq.txt).

Also, with only 5 short documents in the demo corpus, a fixed top_k=4 with
no relevance filtering meant that on some queries the 3rd or 4th result was
barely related to the question at all, but still got passed to the LLM as
"context", which encourages hallucination/dilution in generation.

FIX:
1. Hybrid retrieval: combine a lexical BM25 score (rank_bm25) with the
   vector cosine similarity score. BM25 catches exact keyword overlaps
   ("ยกเลิก", "Pro Plan") that embeddings alone sometimes under-weight,
   while vector search still handles paraphrases/synonyms.
   final_score = alpha * vector_score + (1 - alpha) * normalized_bm25_score
2. Similarity threshold: drop any result whose *vector* similarity score is
   below SIMILARITY_THRESHOLD, so clearly irrelevant chunks are never
   forwarded to the generation step even if they technically make the
   top-k cut.
"""

import logging
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi

import config
from embedding_store import VectorStore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("retriever")


def _tokenize(text: str) -> List[str]:
    # Very simple whitespace + lowercase tokenizer. Thai has no whitespace
    # between words, so this under-segments Thai text into long "words";
    # it still works reasonably for BM25 here because our documents mix in
    # enough Latin-script keywords (product names, plan names) that exact
    # keyword overlap on those terms is what we actually need BM25 for.
    return text.lower().split()


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self._bm25 = None
        self._bm25_corpus_ids = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        if not self.store.metadata:
            self._bm25 = None
            return
        tokenized_corpus = [_tokenize(rec["text"]) for rec in self.store.metadata]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_corpus_ids = list(range(len(self.store.metadata)))

    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        alpha: float = config.HYBRID_ALPHA,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """Return up to k chunks, ranked by a blend of vector + BM25 score.

        Chunks whose *vector* similarity is below similarity_threshold are
        dropped entirely, even if their BM25 score is high, since a chunk
        that is lexically similar but semantically off-topic is still a
        poor context for generation.
        """
        if self.store.index.ntotal == 0:
            logger.warning("Vector store is empty -- returning no results.")
            return []

        # Over-fetch on the vector side so the BM25 re-ranking has more
        # candidates to work with than the final k we return.
        vector_results = self.store.search(query, k=min(self.store.index.ntotal, max(k * 3, k)))
        vector_results = [r for r in vector_results if r["score"] >= similarity_threshold]

        if not vector_results:
            logger.info(
                f"No chunks passed the similarity threshold ({similarity_threshold}) "
                f"for query: {query!r}"
            )
            return []

        if self._bm25 is None:
            return vector_results[:k]

        bm25_scores_all = self._bm25.get_scores(_tokenize(query))
        max_bm25 = max(bm25_scores_all) if max(bm25_scores_all) > 0 else 1.0

        # Match each vector_result back to its BM25 score by chunk_id.
        id_to_bm25_idx = {
            rec["chunk_id"]: i for i, rec in enumerate(self.store.metadata)
        }
        blended = []
        for rec in vector_results:
            bm25_idx = id_to_bm25_idx.get(rec["chunk_id"])
            bm25_norm = (bm25_scores_all[bm25_idx] / max_bm25) if bm25_idx is not None else 0.0
            final_score = alpha * rec["score"] + (1 - alpha) * bm25_norm
            merged = dict(rec)
            merged["vector_score"] = rec["score"]
            merged["bm25_score"] = float(bm25_norm)
            merged["score"] = float(final_score)
            blended.append(merged)

        blended.sort(key=lambda r: r["score"], reverse=True)
        return blended[:k]
