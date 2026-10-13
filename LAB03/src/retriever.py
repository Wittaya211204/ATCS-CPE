"""
retriever.py
------------
Stage 5 of the pipeline: Retrieval.

LAB03 changes compared with LAB02 (results measured in eval/evaluate.py):

1. Thai word segmentation for BM25.
   LAB02 tokenized with `text.lower().split()`. Thai has no spaces between
   words, so a whole Thai sentence became ONE token and BM25 could only
   match when the query and the chunk shared an identical long string.
   That was acceptable for LAB02 (English product names did the matching),
   but the constitution is almost entirely Thai. We now segment with
   PyThaiNLP `word_tokenize(engine="newmm")` and drop Thai stopwords.

2. Thai/Arabic digit normalization.
   The documents write "มาตรา ๓๔" (Thai digits); users type "มาตรา 34".
   chunking.py converts the documents and this module converts the query,
   so both sides use Arabic digits.

3. Direct section lookup.
   If the question names a section ("มาตรา 34 ว่าอย่างไร"), that exact
   chunk is fetched by metadata and ranked first. Semantic search is a poor
   way to find "section 34": every section looks alike to an embedding.

4. Year (edition) filter.
   Three constitutions talk about the same topics in similar words, so
   unfiltered search mixes 2540/2550/2560 answers. If the question names a
   year we search only that edition, otherwise config.DEFAULT_YEAR (2560,
   the one currently in force).

5. Hybrid = union, not re-rank.
   LAB02 only re-ranked the vector top-k*3 with BM25, so a chunk that BM25
   found but the embedding missed could never be returned. We now score
   every chunk of the selected edition with both methods and blend.
"""

import re
import logging
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

import config
from chunking import thai_to_arabic_digits

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("retriever")

try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    _STOPWORDS = set(thai_stopwords())
except ImportError:  # pragma: no cover
    word_tokenize = None
    _STOPWORDS = set()

_HEADER_RE = re.compile(r"^\[[^\]]*\]\n")


def tokenize_whitespace(text: str) -> List[str]:
    """LAB02 tokenizer, kept for comparison."""
    return text.lower().split()


def tokenize_thai(text: str) -> List[str]:
    if word_tokenize is None:
        raise ImportError("pythainlp is required: pip install pythainlp")
    tokens = word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)
    return [t for t in tokens if t.strip() and t not in _STOPWORDS and not re.fullmatch(r"\W+", t)]


TOKENIZERS = {"thai": tokenize_thai, "whitespace": tokenize_whitespace}


def detect_year(query: str, available_years) -> Optional[str]:
    q = thai_to_arabic_digits(query)
    for y in re.findall(r"25\d\d", q):
        if y in available_years:
            return y
    return None


def detect_section(query: str) -> Optional[str]:
    m = re.search(r"มาตรา\s*(\d+(?:/\d+)?)", thai_to_arabic_digits(query))
    return m.group(1) if m else None


class HybridRetriever:
    def __init__(self, metadata: List[Dict], store=None, tokenizer: str = config.BM25_TOKENIZER):
        """metadata: chunk records. store: VectorStore, or None for BM25-only."""
        self.metadata = metadata
        self.store = store
        self.tokenize = TOKENIZERS[tokenizer]
        self.years = sorted({r.get("year") for r in metadata if r.get("year")})
        # Index the body only; the [header] line is identical boilerplate.
        corpus = [self.tokenize(_HEADER_RE.sub("", r["text"])) for r in metadata]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = config.TOP_K, alpha: float = config.HYBRID_ALPHA,
               similarity_threshold: float = config.SIMILARITY_THRESHOLD,
               year: Optional[str] = None) -> List[Dict]:
        if not self.metadata:
            return []
        query_n = thai_to_arabic_digits(query)
        year = year or detect_year(query_n, self.years) or config.DEFAULT_YEAR
        mask = np.array([r.get("year") == year for r in self.metadata])

        results: List[Dict] = []
        seen = set()

        # (3) exact section lookup
        section = detect_section(query_n)
        if section:
            for i, r in enumerate(self.metadata):
                if mask[i] and r.get("section") == section:
                    rec = dict(r, score=1.0, match="section_lookup")
                    results.append(rec)
                    seen.add(i)

        # BM25 scores, normalised to 0..1 within the selected edition
        bm25 = np.zeros(len(self.metadata))
        if self._bm25 is not None:
            raw = np.asarray(self._bm25.get_scores(self.tokenize(query_n)))
            top = raw[mask].max() if mask.any() else 0
            bm25 = raw / top if top > 0 else raw * 0

        use_vectors = self.store is not None and alpha > 0
        if use_vectors:
            vec = self.store.score_all(query_n)
            blended = alpha * vec + (1 - alpha) * bm25
            keep = mask & (vec >= similarity_threshold)
        else:
            vec = None
            blended = bm25
            keep = mask & (bm25 > 0)

        for i in np.argsort(-blended):
            if len(results) >= k:
                break
            if not keep[i] or i in seen:
                continue
            rec = dict(self.metadata[i], score=float(blended[i]), bm25_score=float(bm25[i]),
                       match="hybrid" if use_vectors else "bm25")
            if vec is not None:
                rec["vector_score"] = float(vec[i])
            results.append(rec)
            seen.add(i)

        if not results:
            logger.info(f"No chunks passed retrieval filters for query: {query!r}")
        return results[:max(k, len([r for r in results if r.get('match') == 'section_lookup']))]
