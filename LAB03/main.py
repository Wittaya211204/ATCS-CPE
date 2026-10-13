"""
main.py  (LAB03 - RAG over Thai constitutions 2540 / 2550 / 2560)
-------
    python main.py                                   -> interactive mode
    python main.py "เสรีภาพในการแสดงความคิดเห็น"        -> one question (2560 by default)
    python main.py "รัฐธรรมนูญ 2550 มาตรา 45 ว่าอย่างไร"  -> a specific edition / section
    python main.py --rebuild                         -> re-embed all documents

Default provider is Ollama (free). Pull the models once:
    ollama pull bge-m3
    ollama pull llama3.2
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RAGPipeline  # noqa: E402


def show(result):
    print("\n=== คำตอบ ===")
    print(result["answer"])
    if result["retrieved_chunks"]:
        print("\n--- มาตราที่ค้นเจอ ---")
        for r in result["retrieved_chunks"]:
            print(f"  [{r.get('match')}] {r['text'].splitlines()[0]}  score={r['score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    rag = RAGPipeline(rebuild=args.rebuild)

    if args.question:
        show(rag.ask(args.question))
    else:
        print("ถามเรื่องรัฐธรรมนูญได้เลย (ระบุปี 2540/2550 ได้, ไม่ระบุ = 2560). พิมพ์ 'exit' เพื่อออก")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q:
                show(rag.ask(q))
