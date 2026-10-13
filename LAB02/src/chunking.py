"""
chunking.py
-----------
Stage 2 of the RAG pipeline: Chunking.

REAL PROBLEM FOUND DURING DEVELOPMENT (see README.md, Section 2):
An early version of this project used `naive_fixed_chunk()` below: it simply
cut every document into fixed-size windows of N characters with no overlap
and no awareness of sentence/paragraph boundaries.

Symptom observed while testing on data/docs/troubleshooting_guide.txt with
naive_fixed_chunk(text, 400): chunk #3 ends with
"...1. จำกัดจำนวนไฟล์ที่ซิงค์พร้อมกันในเมนู Settings > Bandwidth > Max
concurrent uploads\n2. เปลี่ยนภูมิภาคเซิร์ฟเวอร์" and chunk #4 begins with
"(Data center region) ให้ใกล้กับตำแหน่งผู้ใช้งานมากที่สุด\n3. เพิ่มโฟลเดอร์
ซิงค์ลงในรายการยกเว้น...". Step 2's own instruction was split across the
boundary, and step 3 was separated from the heading/cause text that gives
it context. A chunk retrieved from either half alone gives the LLM an
incomplete numbered list, e.g. it may render step 2 with no target region
mentioned. (Reproduce with: `python chunking.py`.)

FIX: `smart_chunk()` below chunks along paragraph boundaries first, and only
falls back to sentence-level splitting when a single paragraph is longer
than chunk_size. It also adds character overlap between consecutive chunks
so that context near a boundary is not lost entirely. Both functions are
kept in this file (instead of deleting the naive one) so the before/after
behavior can be reproduced and compared directly -- see
`eval/evaluate.py::demo_chunking_fix()`.
"""

import re
from typing import List, Dict


def naive_fixed_chunk(text: str, chunk_size: int = 400) -> List[str]:
    """The original, buggy approach: blind fixed-width slicing.

    Kept intentionally for comparison/demo purposes -- do NOT use this in
    the production pipeline (see smart_chunk instead).
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?๚๛])\s+|(?<=\n)\s*\n")


def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def _split_paragraph_into_sentences(paragraph: str) -> List[str]:
    # Thai text often has no spaces between sentences and no clear sentence
    # terminators, so this is a best-effort split on line breaks / Latin
    # punctuation. It is not perfect, but it is far better than slicing
    # mid-word, and is enough to keep numbered steps intact in our data.
    parts = re.split(r"(?<=[.!?])\s+|\n", paragraph)
    return [p.strip() for p in parts if p.strip()]


def smart_chunk(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[str]:
    """Paragraph-aware chunking with overlap.

    Strategy:
      1. Split on paragraph boundaries (blank lines) first, since a
         paragraph is usually a self-contained unit of meaning in our docs
         (e.g. one FAQ answer, one troubleshooting category).
      2. If a paragraph is still longer than chunk_size, split it further
         on sentence-like boundaries rather than raw character count.
      3. Greedily pack sentences/paragraphs into chunks up to chunk_size.
      4. Carry the last `overlap` characters of each chunk into the start
         of the next chunk, so information near a boundary still appears
         in at least one retrievable chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units: List[str] = []
    for paragraph in _split_into_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_paragraph_into_sentences(paragraph))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # seed next chunk with overlap from the tail of this one
                current = current[-overlap:] + "\n" + unit
            else:
                # single unit longer than chunk_size even after sentence
                # splitting (rare) -> hard-wrap it as a last resort
                for i in range(0, len(unit), chunk_size - overlap):
                    chunks.append(unit[i:i + chunk_size])
                current = ""
    if current:
        chunks.append(current)

    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Dict]:
    """Apply smart_chunk to every loaded document.

    Returns a flat list of chunk records:
      {"chunk_id": str, "source": filename, "text": chunk_text}
    """
    chunk_records = []
    for doc in documents:
        pieces = smart_chunk(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunk_records.append({
                "chunk_id": f"{doc['doc_id']}::chunk{i}",
                "source": doc["source"],
                "text": piece,
            })
    return chunk_records


if __name__ == "__main__":
    from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingest import load_documents

    docs = load_documents(DOCS_DIR)
    naive_total = sum(len(naive_fixed_chunk(d["text"], CHUNK_SIZE)) for d in docs)
    smart_records = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"naive_fixed_chunk -> {naive_total} chunks")
    print(f"smart_chunk       -> {len(smart_records)} chunks")
    print("\nSample smart chunk:\n", smart_records[0]["text"][:300])
