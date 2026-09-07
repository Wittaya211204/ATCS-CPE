"""
evaluate.py
-----------
Small, reproducible demonstrations of the problems documented in README.md,
run directly against this project's own code (not hypothetical examples).

Run:
    cd rag_system
    python eval/evaluate.py chunking      # no API key needed
    python eval/evaluate.py encoding      # no API key needed
    python eval/evaluate.py dedup         # no API key needed
    python eval/evaluate.py hallucination # needs OPENAI_API_KEY in .env
    python eval/evaluate.py all           # runs everything
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
from ingest import load_documents  # noqa: E402
from chunking import naive_fixed_chunk, smart_chunk  # noqa: E402


def demo_encoding_fix():
    print("\n--- Demo: encoding fallback (ingest.py) ---")
    docs = load_documents(config.DOCS_DIR)
    names = [d["source"] for d in docs]
    assert "server_maintenance_notice.txt" in names, "cp874-encoded file failed to load!"
    print("PASS: server_maintenance_notice.txt (saved in cp874) loaded correctly "
          "via the encoding fallback list in ingest.py.")


def demo_chunking_fix():
    print("\n--- Demo: naive_fixed_chunk vs smart_chunk (chunking.py) ---")
    docs = load_documents(config.DOCS_DIR)
    doc = next(d for d in docs if d["doc_id"] == "troubleshooting_guide.txt")

    naive_chunks = naive_fixed_chunk(doc["text"], config.CHUNK_SIZE)
    smart_chunks = smart_chunk(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    marker = "concurrent uploads"
    naive_hit = [c for c in naive_chunks if marker in c]
    smart_hit = [c for c in smart_chunks if marker in c]

    naive_has_step3 = any("เพิ่มโฟลเดอร์ซิงค์ลงในรายการยกเว้น" in c for c in naive_hit)
    smart_has_step3 = any("เพิ่มโฟลเดอร์ซิงค์ลงในรายการยกเว้น" in c for c in smart_hit)

    print(f"naive_fixed_chunk: chunk containing '{marker}' also contains step 3? {naive_has_step3}")
    print(f"smart_chunk:       chunk containing '{marker}' also contains step 3? {smart_has_step3}")
    if not naive_has_step3 and smart_has_step3:
        print("PASS: smart_chunk keeps the full numbered fix list in one chunk; naive_fixed_chunk splits it.")
    else:
        print("NOTE: result differs from the documented example -- re-check README.md Section 2.")


def demo_dedup():
    print("\n--- Demo: exact-duplicate rejection (embedding_store.py) ---")
    import numpy as np
    import embedding_store

    def fake_get_embeddings(texts):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), config.EMBEDDING_DIM)).astype("float32")

    embedding_store.get_embeddings = fake_get_embeddings
    from embedding_store import VectorStore
    from chunking import chunk_documents

    docs = load_documents(config.DOCS_DIR)
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    store = VectorStore()
    store.add(chunks)
    before = store.index.ntotal
    store.add(chunks)  # add the exact same chunks again
    after = store.index.ntotal
    print(f"vectors after first add: {before}, after re-adding identical chunks: {after}")
    assert before == after, "Duplicate chunks were inserted again!"
    print("PASS: re-adding identical chunks did not grow the index.")


def demo_hallucination_guard():
    print("\n--- Demo: 'not found' guard on unanswerable question (generator.py) ---")
    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        print("SKIPPED: set OPENAI_API_KEY in .env, or switch LLM_PROVIDER=ollama, to run this demo.")
        return
    try:
        from pipeline import RAGPipeline
        rag = RAGPipeline()
        result = rag.ask("บริษัท TechNova มีสาขาในต่างประเทศกี่ประเทศ?")
        print("Q: บริษัท TechNova มีสาขาในต่างประเทศกี่ประเทศ?")
        print("A:", result["answer"])
        print("(This question is not covered by any document in data/docs/. "
              "The answer should be the fixed 'not found' message, not a made-up number.)")
    except Exception as e:
        print(f"SKIPPED: could not reach the LLM provider ({config.LLM_PROVIDER}): {e}")
        if config.LLM_PROVIDER == "ollama":
            print("Make sure Ollama is running (`ollama serve`) and models are pulled "
                  "(`ollama pull nomic-embed-text` and `ollama pull llama3.2`).")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    demos = {
        "encoding": demo_encoding_fix,
        "chunking": demo_chunking_fix,
        "dedup": demo_dedup,
        "hallucination": demo_hallucination_guard,
    }
    if target == "all":
        for fn in demos.values():
            fn()
    elif target in demos:
        demos[target]()
    else:
        print(f"Unknown target '{target}'. Choose from: {list(demos)} or 'all'.")
