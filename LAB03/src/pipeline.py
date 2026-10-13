"""
pipeline.py
-----------
Wires all pipeline stages together:
  ingest -> chunk -> embed & index (build phase, done once / on --rebuild)
  retrieve -> generate (query phase, done per user question)
"""

import argparse
import logging

import config
from ingest import load_documents
from chunking import chunk_documents
from embedding_store import VectorStore
from retriever import HybridRetriever
from generator import generate_answer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


class RAGPipeline:
    def __init__(self, rebuild: bool = False):
        self.store = None if rebuild else VectorStore.load()
        if self.store is None:
            self.store = self._build_index()
        self.retriever = HybridRetriever(self.store.metadata, self.store)

    def _build_index(self) -> VectorStore:
        logger.info("Building index from scratch...")
        documents = load_documents(config.DOCS_DIR)
        chunk_records = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
                                        strategy=config.CHUNK_STRATEGY)
        logger.info(f"{len(documents)} document(s) -> {len(chunk_records)} chunk(s)")
        store = VectorStore()
        store.add(chunk_records)
        store.save()
        return store

    def ask(self, question: str) -> dict:
        contexts = self.retriever.search(question)
        result = generate_answer(question, contexts)
        result["retrieved_chunks"] = contexts
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force re-embedding of all documents")
    parser.add_argument("question", nargs="?", help="Question to ask (skip for interactive mode)")
    args = parser.parse_args()

    rag = RAGPipeline(rebuild=args.rebuild)

    if args.question:
        result = rag.ask(args.question)
        print("\n=== คำตอบ ===")
        print(result["answer"])
    else:
        print("พิมพ์คำถาม (พิมพ์ 'exit' เพื่อออก)")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            result = rag.ask(q)
            print("\n=== คำตอบ ===")
            print(result["answer"])
            if result["sources_used"]:
                print(f"\n(ใช้ chunk จาก: {', '.join(set(result['sources_used']))})")
