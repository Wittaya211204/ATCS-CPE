"""
evaluate.py  (LAB03)
-----------
Reproducible checks for the LAB03 changes. Everything except
`hallucination` runs WITHOUT Ollama/OpenAI.

    python eval/evaluate.py structure     # section counts per edition
    python eval/evaluate.py lookup        # digit normalisation, section/year detection
    python eval/evaluate.py retrieval     # hit@1 / hit@3 / MRR, LAB02 vs LAB03 settings
    python eval/evaluate.py hallucination # needs Ollama (or OpenAI) running
    python eval/evaluate.py all
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.INFO)

import config  # noqa: E402
from ingest import load_documents  # noqa: E402
from chunking import chunk_documents, thai_to_arabic_digits  # noqa: E402
from retriever import HybridRetriever, detect_section, detect_year  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "retrieval_results.json")

# Questions are paraphrased on purpose (not copied from the text).
# `answer` is a phrase that occurs in the correct section of the 2560
# constitution; a retrieved chunk counts as a hit if it contains it.
# `section` was checked against data/docs/constitution_2560.txt.
GOLD = [
    ("ประเทศไทยแบ่งแยกดินแดนได้หรือไม่", "1", "ราชอาณาจักรอันหนึ่งอันเดียว"),
    ("ถ้ากฎหมายขัดกับรัฐธรรมนูญจะเป็นอย่างไร", "5", "กฎหมายสูงสุด"),
    ("ใครดำรงตำแหน่งจอมทัพไทย", "8", "จอมทัพไทย"),
    ("ประธานองคมนตรีและองคมนตรีมีได้ทั้งหมดกี่คน", "10", "องคมนตรีอื่นอีกไม่เกินสิบแปดคน"),
    ("ผู้หญิงกับผู้ชายมีสิทธิเท่ากันหรือไม่", "27", "ชายและหญิงมีสิทธิเท่าเทียมกัน"),
    ("รัฐธรรมนูญคุ้มครองความเป็นส่วนตัวและชื่อเสียงของบุคคลอย่างไร", "32", "ความเป็นอยู่ส่วนตัว"),
    ("ประชาชนพูดหรือเขียนแสดงความเห็นได้อย่างเสรีหรือไม่", "34", "เสรีภาพในการแสดงความคิดเห็น"),
    ("ประชาชนรวมตัวกันประท้วงโดยสงบได้หรือไม่", "44", "เสรีภาพในการชุมนุม"),
    ("คนไทยมีสิทธิได้รับการรักษาพยาบาลจากรัฐหรือไม่", "47", "สิทธิได้รับบริการสาธารณสุข"),
    ("การออกไปเลือกตั้งเป็นหน้าที่ของคนไทยหรือไม่", "50", "ไปใช้สิทธิเลือกตั้ง"),
    ("รัฐต้องจัดการศึกษาฟรีให้เด็กเป็นเวลากี่ปี", "54", "สิบสองปี"),
    ("สภาผู้แทนราษฎรมีสมาชิกทั้งหมดกี่คน", "83", "ห้าร้อยคน"),
    ("อายุเท่าไรจึงมีสิทธิเลือกตั้ง", "95", "สิบแปดปี"),
    ("วุฒิสภามีสมาชิกกี่คน", "107", "สองร้อยคน"),
    ("คณะรัฐมนตรีมีรัฐมนตรีได้มากที่สุดกี่คน", "158", "สามสิบห้าคน"),
    ("ศาลรัฐธรรมนูญมีตุลาการกี่คน", "200", "ตุลาการศาลรัฐธรรมนูญจำนวนเก้าคน"),
    ("ผู้ตรวจการแผ่นดินมีกี่คน", "228", "ผู้ตรวจการแผ่นดินมีจำนวนสามคน"),
    ("รัฐบาลต้องดูแลให้คนเข้าถึงบริการสาธารณสุขที่มีประสิทธิภาพอย่างทั่วถึงหรือไม่", "47", "สิทธิได้รับบริการสาธารณสุข"),
]


def _load(strategy):
    docs = load_documents(config.DOCS_DIR)
    return chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP, strategy=strategy)


def demo_structure():
    print("\n--- Structure: one chunk per section (chunking.py) ---")
    recs = _load("legal")
    expected = {"2540": 336, "2550": 309, "2560": 279}
    for year, n in expected.items():
        got = len({r["section"] for r in recs if r["year"] == year and r["section"]})
        status = "PASS" if got == n else "FAIL"
        print(f"{status}: {year} -> {got} sections (official: {n})")
    for q, sec, ans in GOLD:
        chunk = " ".join(r["text"] for r in recs if r["year"] == "2560" and r["section"] == sec)
        assert ans in chunk, f"gold answer for มาตรา {sec} not found in that section: {ans}"
    print(f"PASS: all {len(GOLD)} gold answers really occur in their gold section.")


def demo_lookup():
    print("\n--- Digit normalisation / section & year detection (retriever.py) ---")
    assert thai_to_arabic_digits("มาตรา ๓๔") == "มาตรา 34"
    assert detect_section("มาตรา ๓๔ ว่าอย่างไร") == "34"
    assert detect_section("มาตรา 34 ว่าอย่างไร") == "34"
    assert detect_year("รัฐธรรมนูญ ๒๕๕๐ เรื่องสิทธิ", ["2540", "2550", "2560"]) == "2550"
    print("PASS: 'มาตรา ๓๔' == 'มาตรา 34'; year 2550 detected from Thai digits.")

    recs = _load("legal")
    r = HybridRetriever(recs, store=None, tokenizer="thai")
    for q, year, sec in [("มาตรา 34 ว่าอย่างไร", "2560", "34"),
                         ("รัฐธรรมนูญ 2550 มาตรา 45 ว่าอย่างไร", "2550", "45"),
                         ("รัฐธรรมนูญ ๒๕๔๐ มาตรา ๑", "2540", "1")]:
        top = r.search(q, k=3)[0]
        ok = top["year"] == year and top["section"] == sec
        print(f"{'PASS' if ok else 'FAIL'}: {q!r} -> {top['text'].splitlines()[0]}")


def _evaluate(retriever, k=3):
    hit1 = hit3 = rr = 0.0
    rows = []
    for q, sec, ans in GOLD:
        res = retriever.search(q, k=k, year="2560")
        rank = next((i + 1 for i, c in enumerate(res) if ans in c["text"]), None)
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        rows.append({"question": q, "gold_section": sec, "rank": rank,
                     "top1": res[0]["text"].splitlines()[0][:80] if res else None})
    n = len(GOLD)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "MRR": rr / n}, rows


def demo_retrieval():
    print("\n--- Retrieval benchmark, BM25 only, 2560 edition, "
          f"{len(GOLD)} paraphrased questions ---")
    configs = [
        ("A  LAB02: smart_chunk + whitespace BM25", "smart", "whitespace"),
        ("B  smart_chunk + Thai-tokenised BM25", "smart", "thai"),
        ("C  LAB03: legal_chunk + Thai-tokenised BM25", "legal", "thai"),
    ]
    out = {}
    print(f"{'config':48s} hit@1  hit@3  MRR")
    for name, strat, tok in configs:
        recs = _load(strat)
        metrics, rows = _evaluate(HybridRetriever(recs, store=None, tokenizer=tok))
        out[name] = {"metrics": metrics, "per_question": rows}
        print(f"{name:48s} {metrics['hit@1']:.2f}   {metrics['hit@3']:.2f}   {metrics['MRR']:.2f}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"(per-question results saved to {os.path.relpath(RESULTS_PATH)})")


def demo_hallucination_guard():
    print("\n--- 'not found' guard on an unanswerable question (needs LLM) ---")
    try:
        from pipeline import RAGPipeline
        rag = RAGPipeline()
        q = "รัฐธรรมนูญ 2560 กำหนดอัตราภาษีมูลค่าเพิ่มไว้กี่เปอร์เซ็นต์"
        result = rag.ask(q)
        print("Q:", q)
        print("A:", result["answer"])
        print("(The constitution does not set a VAT rate -> expected the fixed 'not found' reply.)")
    except Exception as e:
        print(f"SKIPPED: could not reach {config.LLM_PROVIDER}: {e}")
        print("Start Ollama and run: ollama pull bge-m3 && ollama pull llama3.2")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    demos = {"structure": demo_structure, "lookup": demo_lookup,
             "retrieval": demo_retrieval, "hallucination": demo_hallucination_guard}
    if target == "all":
        for fn in demos.values():
            fn()
    elif target in demos:
        demos[target]()
    else:
        print(f"Unknown target '{target}'. Choose from: {list(demos)} or 'all'.")
