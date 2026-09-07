"""
generator.py
------------
Stage 6 of the pipeline: Generation.

REAL PROBLEMS FOUND DURING DEVELOPMENT:

(a) Hallucination when context was insufficient.
    Symptom: asking a question not covered by any document (e.g. "บริษัทมี
    สาขาต่างประเทศไหม") still produced a confident-sounding made-up answer,
    because the first version of the prompt only said "answer the question
    using the context below" without telling the model what to do when the
    context doesn't contain the answer.
    Fix: the system prompt in `build_prompt()` explicitly instructs the
    model to reply with a fixed "not found" sentence when the retrieved
    context does not contain the answer, and to never use outside
    knowledge. This is verified in eval/evaluate.py with an
    intentionally-unanswerable question.

(b) Context window / cost blow-up on longer queries.
    Symptom: with top_k pushed up during testing, concatenating every
    retrieved chunk sometimes produced a prompt far longer than needed,
    increasing latency and cost with no accuracy benefit (most of the
    tokens were irrelevant padding from lower-ranked chunks).
    Fix: `build_prompt()` truncates the concatenated context to
    MAX_CONTEXT_CHARS, taking chunks in ranked order and stopping once the
    budget is used, rather than truncating mid-chunk or exceeding budget.

(c) No traceability -- couldn't tell which document an answer came from.
    Symptom: reviewers (i.e. testing this ourselves) had no way to verify
    an answer without manually re-reading every source document.
    Fix: every chunk injected into the prompt is labeled with its source
    filename ([source: filename.txt]), and the system prompt instructs the
    model to cite the source filename(s) it used at the end of its answer.
"""

import logging
from typing import List, Dict

from openai import OpenAI

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("generator")

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามโดยอ้างอิงจากเอกสารที่ให้มาเท่านั้น (RAG assistant).

กฎที่ต้องปฏิบัติตามอย่างเคร่งครัด:
1. ตอบคำถามโดยใช้ข้อมูลจาก "บริบท" (context) ที่ให้มาด้านล่างเท่านั้น ห้ามใช้ความรู้ภายนอกหรือคาดเดา
2. หากบริบทที่ให้มาไม่มีข้อมูลเพียงพอที่จะตอบคำถาม ให้ตอบเพียงว่า:
   "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ กรุณาตรวจสอบกับแหล่งข้อมูลอื่นหรือฝ่ายที่เกี่ยวข้อง"
   ห้ามแต่งคำตอบขึ้นเองโดยเด็ดขาด
3. ท้ายคำตอบให้ระบุแหล่งอ้างอิงในรูปแบบ "อ้างอิง: <ชื่อไฟล์>" โดยระบุเฉพาะไฟล์ที่ใช้ตอบจริง
4. ตอบให้กระชับ ตรงประเด็น เป็นภาษาเดียวกับคำถาม"""


def build_prompt(query: str, contexts: List[Dict], max_context_chars: int = config.MAX_CONTEXT_CHARS):
    """Assemble the user-turn prompt from ranked retrieved chunks.

    Truncates at a chunk boundary once max_context_chars is reached, so we
    never cut a chunk in half or blow the context budget.
    """
    context_blocks = []
    used_chars = 0
    for rec in contexts:
        block = f"[source: {rec['source']}]\n{rec['text']}"
        if used_chars + len(block) > max_context_chars and context_blocks:
            break
        context_blocks.append(block)
        used_chars += len(block)

    context_text = "\n\n---\n\n".join(context_blocks)
    user_prompt = f"บริบท (Context):\n{context_text}\n\nคำถาม: {query}"
    return user_prompt, [c["source"] for c in contexts[:len(context_blocks)]]


def generate_answer(query: str, contexts: List[Dict]) -> Dict:
    """Call the chat model and return {"answer": str, "sources_used": [...]}.

    If contexts is empty (retriever found nothing above the similarity
    threshold), we skip the API call entirely and return the fixed
    "not found" message -- there is no point spending tokens asking the
    model to answer from an empty context.
    """
    if not contexts:
        return {
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ กรุณาตรวจสอบกับแหล่งข้อมูลอื่นหรือฝ่ายที่เกี่ยวข้อง",
            "sources_used": [],
        }

    if config.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill "
            "in your key, or set LLM_PROVIDER=ollama to run for free with "
            "no API key."
        )

    user_prompt, sources = build_prompt(query, contexts)
    # Ollama exposes an OpenAI-compatible endpoint at /v1, so the same
    # `openai` client works for both providers -- see config.py.
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=config.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources_used": sources}
