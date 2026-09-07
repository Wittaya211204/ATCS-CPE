"""
ingest.py
---------
Stage 1 of the RAG pipeline: Document Loading / Ingestion.

REAL PROBLEM FOUND DURING DEVELOPMENT:
When loading data/docs/server_maintenance_notice.txt with plain
open(path, encoding="utf-8"), Python raised:

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa1 in position ...

Root cause: the file was produced/exported on a Windows machine that saved
Thai text using the legacy "Windows-874" (cp874) codepage instead of UTF-8.
This is extremely common with Thai documents exported from old systems,
Excel, or government portals.

FIX: try a small list of encodings in order of likelihood (utf-8, then
cp874 / tis-620 for Thai legacy text) instead of assuming utf-8 blindly.
We log which files needed a fallback so it's visible during ingestion,
rather than failing silently or corrupting the text into mojibake.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

# Ordered by likelihood for this project's data sources (Thai + English docs)
CANDIDATE_ENCODINGS = ["utf-8", "cp874", "tis-620", "latin-1"]


def _read_with_fallback_encoding(path: str) -> str:
    """Try multiple encodings until one decodes without error.

    Returns the decoded text. Raises RuntimeError if every candidate fails,
    which is preferable to silently returning garbled text.
    """
    last_error = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            if enc != "utf-8":
                logger.warning(
                    f"'{os.path.basename(path)}' was NOT valid UTF-8; "
                    f"successfully decoded using fallback encoding '{enc}'."
                )
            return text
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Could not decode '{path}' with any known encoding "
        f"{CANDIDATE_ENCODINGS}. Last error: {last_error}"
    )


def load_documents(docs_dir: str):
    """Load every .txt file in docs_dir.

    Returns a list of dicts: {"doc_id": str, "source": filename, "text": str}

    Handles / detects (do not just crash on first file that fails):
      - encoding problems (see module docstring)
      - empty files (0 bytes or whitespace-only) -> skipped with a warning,
        since an empty chunk provides zero retrieval value and would waste
        embedding API calls
      - non-.txt files in the folder -> ignored
    """
    documents = []
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    filenames = sorted(f for f in os.listdir(docs_dir) if f.endswith(".txt"))
    if not filenames:
        logger.warning(f"No .txt files found in {docs_dir}")

    for filename in filenames:
        path = os.path.join(docs_dir, filename)
        try:
            text = _read_with_fallback_encoding(path)
        except RuntimeError as e:
            logger.error(f"Skipping '{filename}': {e}")
            continue

        if not text.strip():
            logger.warning(f"Skipping '{filename}': file is empty after stripping whitespace.")
            continue

        documents.append({
            "doc_id": filename,
            "source": filename,
            "text": text,
        })
        logger.info(f"Loaded '{filename}' ({len(text)} characters)")

    return documents


if __name__ == "__main__":
    from config import DOCS_DIR
    docs = load_documents(DOCS_DIR)
    print(f"\nLoaded {len(docs)} document(s) successfully.")
