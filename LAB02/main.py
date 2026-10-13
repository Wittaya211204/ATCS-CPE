"""
main.py
-------
Entry point. Run:
    python main.py                       -> interactive Q&A over the sample docs
    python main.py "คำถามของคุณ"          -> ask a single question and exit
    python main.py --rebuild             -> force re-embedding of all documents

Requires OPENAI_API_KEY set in a .env file (copy .env.example -> .env first).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RAGPipeline  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("question", nargs="?")
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
