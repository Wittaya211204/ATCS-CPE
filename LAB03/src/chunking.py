"""
chunking.py
-----------
Stage 2 of the RAG pipeline: Chunking.

LAB03 change: the knowledge base is now Thai constitutions (2560, 2550,
2540). Legal text has an explicit structure:

    หมวด ๓  สิทธิและเสรีภาพของปวงชนชาวไทย      <- chapter
        มาตรา ๓๔  บุคคลย่อมมีเสรีภาพ...           <- section (the unit people cite)

The LAB02 `smart_chunk()` splits on blank lines and a character budget, so
one chunk can hold the tail of มาตรา ๓๓ plus the head of มาตรา ๓๔, and the
chunk no longer says which chapter or which constitution it came from.
`legal_chunk()` below splits on "มาตรา N" headings instead, so:

  * one chunk = one section (long sections are sub-split with smart_chunk,
    and every piece keeps the same section header),
  * every chunk carries metadata: year, chapter number, chapter title,
    section number (as an int, Thai digits converted),
  * the chunk text starts with a header line
        [รัฐธรรมนูญ 2560 | หมวด 3 สิทธิและเสรีภาพของปวงชนชาวไทย | มาตรา 34]
    so both the embedding and BM25 "see" the section number and chapter.

`smart_chunk()` and `naive_fixed_chunk()` from LAB02 are kept so the
before/after comparison can be run in eval/evaluate.py.
"""

import re
from typing import List, Dict, Optional

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_THAI_TO_ARABIC = str.maketrans(THAI_DIGITS, "0123456789")


def thai_to_arabic_digits(text: str) -> str:
    """'มาตรา ๓๔' -> 'มาตรา 34'. Used on both documents and queries."""
    return text.translate(_THAI_TO_ARABIC)


# ---------------------------------------------------------------- LAB02 ---

def naive_fixed_chunk(text: str, chunk_size: int = 400) -> List[str]:
    """Blind fixed-width slicing (kept only for comparison)."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs if paragraphs else [text.strip()]


def _split_paragraph_into_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n", paragraph)
    return [p.strip() for p in parts if p.strip()]


def smart_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Paragraph-aware chunking with overlap (unchanged from LAB02)."""
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
                current = current[-overlap:] + "\n" + unit
            else:
                for i in range(0, len(unit), chunk_size - overlap):
                    chunks.append(unit[i:i + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- LAB03 ---

# A section heading is "มาตรา <number>" at the START of a line. Mentions in
# the middle of a sentence ("...ตามมาตรา ๒๕๖") are cross-references, not new
# sections, so they must not start a chunk.
_SECTION_RE = re.compile(r"^[ \t]*มาตรา[ \t]+([๐-๙0-9]+(?:/[๐-๙0-9]+)?)[ \t]+", re.M)
_CHAPTER_RE = re.compile(r"^[ \t]*หมวด[ \t]+([๐-๙0-9]+)[ \t]*\n[ \t]*(.+?)[ \t]*$", re.M)
_TRANSITORY_RE = re.compile(r"^[ \t]*บทเฉพาะกาล[ \t]*$", re.M)


def _year_from_source(source: str) -> Optional[str]:
    m = re.search(r"(\d{4})", source)
    return m.group(1) if m else None


def _chapter_at(position: int, chapters: List[Dict]) -> Dict:
    current = {"chapter": None, "chapter_title": "บทนำ/คำปรารภ"}
    for ch in chapters:
        if ch["pos"] <= position:
            current = ch
        else:
            break
    return current


def legal_chunk_document(doc: Dict, max_chars: int = 1200, overlap: int = 100) -> List[Dict]:
    """Split one constitution into section-level chunk records."""
    text = doc["text"]
    year = _year_from_source(doc["source"])

    chapters = [
        {"pos": m.start(), "chapter": int(thai_to_arabic_digits(m.group(1))),
         "chapter_title": m.group(2).strip()}
        for m in _CHAPTER_RE.finditer(text)
    ]
    chapters += [
        {"pos": m.start(), "chapter": None, "chapter_title": "บทเฉพาะกาล"}
        for m in _TRANSITORY_RE.finditer(text)
    ]
    chapters.sort(key=lambda c: c["pos"])

    sections = list(_SECTION_RE.finditer(text))
    records: List[Dict] = []

    def emit(body: str, section: Optional[str], ch: Dict, part_id: str):
        body = body.strip()
        if not body:
            return
        chapter_label = (f"หมวด {ch['chapter']} {ch['chapter_title']}"
                         if ch["chapter"] is not None else ch["chapter_title"])
        section_label = f"มาตรา {section}" if section else "คำปรารภ"
        header = f"[รัฐธรรมนูญ {year} | {chapter_label} | {section_label}]"
        pieces = [body] if len(body) <= max_chars else smart_chunk(body, max_chars, overlap)
        for i, piece in enumerate(pieces):
            records.append({
                "chunk_id": f"{doc['doc_id']}::{part_id}" + (f"::part{i}" if len(pieces) > 1 else ""),
                "source": doc["source"],
                "year": year,
                "chapter": ch["chapter"],
                "chapter_title": ch["chapter_title"],
                "section": section,
                "text": f"{header}\n{thai_to_arabic_digits(piece)}",
            })

    # Preamble: everything before the first section heading.
    first = sections[0].start() if sections else len(text)
    preamble = text[:first]
    if chapters and chapters[0]["pos"] < first:
        preamble = text[:chapters[0]["pos"]]
    emit(preamble, None, {"chapter": None, "chapter_title": "คำปรารภ"}, "preamble")

    for i, m in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        body = text[m.start():end]
        # Drop a chapter heading that belongs to the NEXT section.
        for ch in chapters:
            if m.start() < ch["pos"] < end:
                body = text[m.start():ch["pos"]]
                break
        section = thai_to_arabic_digits(m.group(1))
        emit(body, section, _chapter_at(m.start(), chapters), f"s{section}")

    return records


def chunk_documents(documents: List[Dict], chunk_size: int = 1200, overlap: int = 100,
                    strategy: str = "legal") -> List[Dict]:
    """strategy='legal' (LAB03 default) or 'smart' (LAB02 behaviour)."""
    records: List[Dict] = []
    for doc in documents:
        if strategy == "legal":
            records.extend(legal_chunk_document(doc, chunk_size, overlap))
        else:
            for i, piece in enumerate(smart_chunk(doc["text"], chunk_size, overlap)):
                records.append({
                    "chunk_id": f"{doc['doc_id']}::chunk{i}",
                    "source": doc["source"],
                    "year": _year_from_source(doc["source"]),
                    "section": None,
                    "text": thai_to_arabic_digits(piece),
                })
    return records


if __name__ == "__main__":
    from collections import Counter
    from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingest import load_documents

    docs = load_documents(DOCS_DIR)
    recs = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"legal_chunk -> {len(recs)} chunks")
    for year, n in sorted(Counter(r["year"] for r in recs).items()):
        secs = {r["section"] for r in recs if r["year"] == year and r["section"]}
        print(f"  {year}: {n} chunks, {len(secs)} distinct sections")
    sample = next(r for r in recs if r["year"] == "2560" and r["section"] == "34")
    print("\nSample:\n" + sample["text"][:400])
