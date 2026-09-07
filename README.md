# RAG System – ปัญหาและการแก้ไขปัญหา

ระบบนี้เป็น RAG (Retrieval-Augmented Generation) ขนาดเล็กที่สร้างขึ้นเพื่อตอบคำถามจากชุดเอกสารตัวอย่าง (นโยบายบริษัท, FAQ สินค้า, คู่มือแก้ไขปัญหา) โดยใช้ Embeddings + FAISS + Chat Completion เอกสารนี้วิเคราะห์ปัญหาที่พบจริงระหว่างพัฒนาในแต่ละขั้นตอนของ pipeline พร้อมสาเหตุ วิธีตรวจสอบ และวิธีแก้ไข โดยอ้างอิงตรงกับซอร์สโค้ดในโปรเจกต์

ระบบรองรับ 2 ตัวเลือกสำหรับ embedding และ generation:
- **Ollama** (ค่าเริ่มต้น) — ฟรี รันบนเครื่องตัวเอง ไม่ต้องมี API key
- **OpenAI** — เสียเงิน ต้องมี `OPENAI_API_KEY`

สลับได้ด้วยตัวแปร `LLM_PROVIDER` ใน `.env` (ดูหัวข้อ "การติดตั้ง" ด้านล่าง)

## การติดตั้ง (แบบฟรี ใช้ Ollama)

1. ติดตั้ง Ollama จาก https://ollama.com/download
2. เปิด terminal แล้วดึงโมเดล 2 ตัวที่ระบบใช้ (ครั้งเดียว):
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. ตรวจสอบว่า Ollama รันอยู่เบื้องหลัง (โดยปกติจะรันอัตโนมัติหลังติดตั้ง ที่ `http://localhost:11434`)
4. `cp .env.example .env` — ไม่ต้องแก้อะไรเพิ่ม เพราะค่าเริ่มต้นตั้งเป็น Ollama ไว้แล้ว
5. `pip install -r requirements.txt`
6. `python main.py "การลาป่วยต้องมีใบรับรองแพทย์เมื่อไหร่?"`

หากต้องการใช้ OpenAI แทน ให้เปิด `.env` แล้วตั้ง `LLM_PROVIDER=openai` และใส่ `OPENAI_API_KEY` ของตัวเอง

## สถาปัตยกรรมโดยรวม

```
data/docs/*.txt
      │
      ▼
[1] ingest.py        โหลดเอกสาร
      ▼
[2] chunking.py       ตัดเอกสารเป็น chunk
      ▼
[3-4] embedding_store.py   สร้าง embedding + เก็บใน FAISS index
      ▼
[5] retriever.py      ค้นหา chunk ที่เกี่ยวข้อง (hybrid: vector + BM25)
      ▼
[6] generator.py      ให้ LLM ตอบคำถามจาก context ที่ค้นเจอ
      ▼
pipeline.py / main.py  ประกอบทุกขั้นตอนเป็นระบบเดียว
```

รันทดสอบทั้งหมด (ไม่ต้องมี API key):
```bash
pip install -r requirements.txt
python eval/evaluate.py encoding
python eval/evaluate.py chunking
python eval/evaluate.py dedup
```
รันแบบเต็ม (ต้องมี API key ใน `.env`):
```bash
cp .env.example .env   # ใส่ OPENAI_API_KEY ของตัวเอง
python main.py "การลาป่วยต้องมีใบรับรองแพทย์เมื่อไหร่?"
```

---

## 1. Ingestion (`src/ingest.py`)

**ปัญหาที่เกิดขึ้นจริงและแก้ไขแล้ว:** โหลดไฟล์ `data/docs/server_maintenance_notice.txt` ด้วย `open(path, encoding="utf-8")` แล้วได้ error:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa1 in position ...
```

**สาเหตุ:** ไฟล์นี้ถูกเซฟด้วย encoding แบบเก่า (Windows-874 / cp874) ซึ่งเป็นปัญหาที่พบได้บ่อยกับเอกสารภาษาไทยที่ export มาจากระบบเก่าหรือ Excel บน Windows ไม่ใช่ UTF-8 ตามที่คาดไว้

**วิธีตรวจสอบ:** รัน `python src/ingest.py` แล้วดู log — ระบบจะ log คำเตือนทุกครั้งที่ต้องใช้ fallback encoding แทนที่จะปล่อยให้ error หรือแปลงข้อความแบบผิด ๆ (mojibake) แบบเงียบ ๆ

**วิธีแก้ไข:** ฟังก์ชัน `_read_with_fallback_encoding()` ไล่ลองถอดรหัสด้วยรายการ `CANDIDATE_ENCODINGS = ["utf-8", "cp874", "tis-620", "latin-1"]` ตามลำดับ จนกว่าจะสำเร็จ

**หลักฐาน (รันจริงแล้ว):**
```
[WARNING] 'server_maintenance_notice.txt' was NOT valid UTF-8;
successfully decoded using fallback encoding 'cp874'.
Loaded 5 document(s) successfully.
```

**ปัญหาอื่นที่ป้องกันไว้ในขั้นตอนนี้:** ไฟล์ว่างเปล่า (0 ไบต์ หรือมีแต่ whitespace) จะถูกข้ามพร้อม log คำเตือน แทนที่จะถูกส่งต่อไป embed โดยเปล่าประโยชน์และเสีย API quota

---

## 2. Chunking (`src/chunking.py`)

**ปัญหาที่เกิดขึ้นจริงและแก้ไขแล้ว:** เวอร์ชันแรกใช้ `naive_fixed_chunk()` ตัดข้อความทุก ๆ 400 ตัวอักษรแบบตรง ๆ ไม่สนใจขอบเขตประโยค/ย่อหน้า

**สาเหตุ:** การตัดแบบ fixed-width ทำให้ chunk บางอันถูกตัดกลางเนื้อหาที่ควรอยู่ด้วยกัน

**วิธีตรวจสอบ:** ทดสอบกับ `data/docs/troubleshooting_guide.txt` (`CHUNK_SIZE=400`) พบว่า chunk #3 จบด้วย
`"...1. จำกัดจำนวนไฟล์ที่ซิงค์พร้อมกันในเมนู Settings > Bandwidth > Max concurrent uploads\n2. เปลี่ยนภูมิภาคเซิร์ฟเวอร์"`
และ chunk #4 เริ่มด้วย
`"(Data center region) ให้ใกล้กับตำแหน่งผู้ใช้งานมากที่สุด\n3. เพิ่มโฟลเดอร์ซิงค์ลงในรายการยกเว้น..."`
— ขั้นตอนที่ 2 ของวิธีแก้ปัญหาถูกตัดขาดครึ่งหนึ่ง และขั้นตอนที่ 3 ถูกแยกออกจากหัวข้อของมัน หาก retriever ดึงมาแค่ chunk เดียว LLM จะเห็นวิธีแก้ปัญหาไม่ครบ

**วิธีแก้ไข:** `smart_chunk()` ตัดตามย่อหน้าก่อน ถ้าย่อหน้ายาวเกิน `chunk_size` ค่อยตัดย่อยตามประโยค พร้อมใส่ overlap (`CHUNK_OVERLAP=80` ตัวอักษร) ระหว่าง chunk ที่ติดกัน

**หลักฐาน (รันจริงแล้วผ่าน `python eval/evaluate.py chunking`):**
```
naive_fixed_chunk: chunk containing 'concurrent uploads' also contains step 3? False
smart_chunk:       chunk containing 'concurrent uploads' also contains step 3? True
PASS: smart_chunk keeps the full numbered fix list in one chunk; naive_fixed_chunk splits it.
```

**ข้อจำกัดที่ยังไม่แก้:** ตัวตัดประโยคภาษาไทย (`_split_paragraph_into_sentences`) เป็นแบบ heuristic (ตัดตาม newline / เครื่องหมายวรรคตอนแบบละติน) เพราะภาษาไทยไม่มีช่องว่างระหว่างประโยคจริง ๆ วิธีนี้ใช้ได้ดีกับข้อมูลตัวอย่างในโปรเจกต์ (มี numbered list และ newline ชัดเจน) แต่ถ้าเป็นย่อหน้าไทยยาว ๆ ที่ไม่มีการขึ้นบรรทัดใหม่เลย อาจยังต้องพึ่ง fallback แบบ hard-wrap ตอนท้ายของฟังก์ชัน

---

## 3-4. Embedding & Vector Store (`src/embedding_store.py`)

### ปัญหา (a): Rate limit / การเชื่อมต่อขัดข้องระหว่าง embed
**สาเหตุ:** เรียก `embeddings.create()` ทีละ chunk จำนวนมากในลูปเดียว บาง request ล้มเหลวชั่วคราว (`RateLimitError`, `APIConnectionError`) แล้วทำให้ทั้ง job ล้มทันที เสีย embedding ที่คำนวณไปแล้วทั้งหมด

**วิธีแก้ไข:** `get_embeddings()` แบ่ง batch ตาม `EMBEDDING_BATCH_SIZE` (ค่าเริ่มต้น 16 ต่อครั้ง แทนที่จะยิงทีละ 1) และ retry แบบ exponential backoff สูงสุด `MAX_RETRIES` ครั้ง ก่อนจะยอมแพ้และโยน exception จริง ๆ

### ปัญหา (b): เพิ่มเอกสารซ้ำทำให้เกิด vector ซ้ำ
**สาเหตุ:** รัน indexing ซ้ำ (เช่นหลังเพิ่มไฟล์ใหม่ 1 ไฟล์) โดยไม่เช็คว่า chunk ไหน embed ไปแล้วบ้าง ทำให้ chunk เดิมถูก embed และเก็บซ้ำ ทำให้ผลการค้นหามี chunk เดิมโผล่ซ้ำหลายรอบ แย่งที่ผลลัพธ์อื่นที่เกี่ยวข้องกว่า

**วิธีตรวจสอบ/แก้ไข:** `VectorStore.add()` คำนวณ SHA-256 hash ของเนื้อหาแต่ละ chunk แล้วข้าม chunk ที่ hash ตรงกับที่มีอยู่แล้ว ยืนยันด้วยการรันจริงใน `eval/evaluate.py dedup`:
```
vectors after first add: 28, after re-adding identical chunks: 28
PASS: re-adding identical chunks did not grow the index.
```

**ข้อจำกัดที่ทราบและยังไม่แก้:** วิธีนี้จับได้เฉพาะ **duplicate แบบตรงตัวทุกตัวอักษร** เท่านั้น ไฟล์ `data/docs/leave_policy.txt` และ `leave_policy_copy_2025.txt` ในชุดข้อมูลตัวอย่างมีเนื้อหาซ้ำกันเกือบทั้งหมดแต่คำไม่ตรงกัน 100% (เป็นไฟล์เก่าที่ฝ่าย HR อัปโหลดซ้ำโดยไม่ตั้งใจ) ระบบปัจจุบัน**ไม่**ตรวจจับ near-duplicate แบบนี้ ยังต้องอาศัยการตรวจสอบด้วยคนหรือทำ similarity-based dedup (เช่น เช็คว่า cosine similarity ระหว่างเอกสารสองไฟล์สูงเกิน threshold หรือไม่) เพิ่มเติมในอนาคต

### ปัญหา (c): Index ถูกสร้างใหม่ทุกครั้งที่รันโปรแกรม
**สาเหตุ:** ไม่มีการเก็บ index ลงดิสก์ ทำให้ทุกครั้งที่รัน `main.py` ต้อง embed เอกสารทั้งหมดใหม่ เสีย API quota และเวลาเริ่มโปรแกรมโดยไม่จำเป็น

**วิธีแก้ไข:** `VectorStore.save()` / `VectorStore.load()` เก็บ FAISS index ไปที่ `data/index/faiss.index` และ metadata ไปที่ `data/index/metadata.json` — `pipeline.py` จะโหลด index เดิมถ้ามีอยู่แล้ว และ build ใหม่เฉพาะตอนไม่พบไฟล์ index หรือเรียกด้วย `--rebuild` เท่านั้น

---

## 5. Retrieval (`src/retriever.py`)

**ปัญหาที่พบ:** ใช้ vector search อย่างเดียวกับคำถามสั้น ๆ ที่มีคำเฉพาะ เช่น "ยกเลิก Pro Plan ต้องทำอย่างไร" บางครั้ง chunk เกี่ยวกับแอปมือถือถูกจัดอันดับสูงกว่า chunk ที่ตอบคำถามจริง เพราะ embedding วางประโยคที่พูดถึง "account settings" ไว้ใกล้กันในปริภูมิเวกเตอร์ ทั้งที่มีแค่ chunk เดียวที่ตอบคำถามได้ตรง — เป็นช่องว่างเชิงคำศัพท์ (vocabulary gap) ที่พบบ่อยกับคำถามสั้นแบบผสมภาษาไทย-อังกฤษ

นอกจากนี้ เมื่อ corpus มีแค่ 5 เอกสาร การตั้ง `top_k=4` ตายตัวโดยไม่กรอง relevance ทำให้บางครั้งผลลัพธ์อันดับ 3-4 แทบไม่เกี่ยวข้องกับคำถามเลย แต่ก็ยังถูกส่งเข้า context ของ LLM อยู่ดี ซึ่งเพิ่มความเสี่ยงที่คำตอบจะเจือจางหรือหลอน (hallucinate)

**วิธีแก้ไข:**
1. **Hybrid search** — `HybridRetriever.search()` รวมคะแนน vector similarity กับคะแนน BM25 (lexical/keyword matching จาก `rank_bm25`) ด้วยสูตร `final_score = alpha * vector_score + (1-alpha) * bm25_norm` (`HYBRID_ALPHA=0.5` ปรับได้ใน `.env`) BM25 ช่วยจับคำเฉพาะที่ตรงตัว เช่น "ยกเลิก", "Pro Plan" ที่ embedding บางครั้งให้น้ำหนักน้อยเกินไป
2. **Similarity threshold** — ผลลัพธ์ที่ vector similarity ต่ำกว่า `SIMILARITY_THRESHOLD` (ค่าเริ่มต้น 0.15) จะถูกตัดทิ้งทันที ไม่ถูกส่งต่อไปยัง generation แม้จะติด top-k ก็ตาม

**วิธีตรวจสอบผลลัพธ์จริง (ทดสอบ plumbing ด้วย embedding จำลอง เพราะไม่ต้องใช้ API key):** โครงสร้างคะแนนที่ได้จริงจากการรัน:
```
product_faq.txt        vector=0.749  bm25=1.000  final=0.875
troubleshooting_guide.txt vector=0.749  bm25=0.695  final=0.722
```
ยืนยันว่าตัว blend คะแนนทำงานถูกต้องตามสูตร (ผลอันดับที่แท้จริงเมื่อใช้ embedding จริงจะแม่นกว่านี้ เพราะ vector คำนวณจากคำจริง ไม่ใช่เลขสุ่ม)

---

## 6. Generation (`src/generator.py`)

### ปัญหา (a): Hallucination เมื่อ context ไม่มีคำตอบ
**สาเหตุ:** prompt เวอร์ชันแรกบอกแค่ "ตอบคำถามจาก context ด้านล่าง" โดยไม่ได้สั่งว่าต้องทำอย่างไรถ้า context ไม่มีคำตอบ ผลคือถามคำถามที่เอกสารไม่ครอบคลุม (เช่น "บริษัทมีสาขาต่างประเทศไหม") LLM ก็ยังตอบแบบมั่นใจทั้งที่เป็นการเดา

**วิธีแก้ไข:** `SYSTEM_PROMPT` สั่งชัดเจนว่าให้ตอบประโยคคงที่ "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่..." เมื่อ context ไม่พอ และห้ามใช้ความรู้ภายนอกเด็ดขาด ทดสอบได้ผ่าน `eval/evaluate.py hallucination` (ต้องมี API key) นอกจากนี้ `generate_answer()` เช็คก่อนว่าถ้า retriever คืนค่า context ว่างเปล่า (ไม่มี chunk ผ่าน threshold เลย) จะตอบข้อความ "ไม่พบข้อมูล" ทันทีโดยไม่ยิง API เลย เพื่อประหยัด cost

### ปัญหา (b): Context ยาวเกินจำเป็นเมื่อปรับ top_k สูงขึ้นตอนทดสอบ
**สาเหตุ:** การต่อ chunk ทุกอันที่ retriever คืนมาเข้า prompt ตรง ๆ โดยไม่จำกัดความยาว ทำให้ prompt ยาวเกินความจำเป็นเมื่อทดลองปรับ `top_k` สูง เพิ่ม latency และค่าใช้จ่ายโดยไม่ได้ช่วยความแม่นยำ (chunk อันดับท้าย ๆ ส่วนใหญ่ไม่ช่วยตอบคำถาม)

**วิธีแก้ไข:** `build_prompt()` ต่อ context ตามลำดับคะแนนจนกว่าจะถึง `MAX_CONTEXT_CHARS` (ค่าเริ่มต้น 6000 ตัวอักษร) แล้วหยุด โดยหยุดที่ขอบเขต chunk เท่านั้น (ไม่ตัด chunk กลางคัน)

### ปัญหา (c): ตรวจสอบที่มาของคำตอบไม่ได้
**สาเหตุ:** ตอนทดสอบเอง ไม่มีทางรู้ว่าคำตอบมาจากเอกสารไหน ต้องไล่เปิดเอกสารทุกไฟล์เพื่อตรวจสอบความถูกต้อง

**วิธีแก้ไข:** ทุก chunk ที่ใส่เข้า prompt จะติด label `[source: <ชื่อไฟล์>]` และ system prompt สั่งให้ LLM ระบุ "อ้างอิง: <ชื่อไฟล์>" ท้ายคำตอบเสมอ

---

## สรุปรายการปัญหาและสถานะ

| # | ขั้นตอน | ปัญหา | สถานะ | ไฟล์ที่แก้ |
|---|---------|-------|-------|-----------|
| 1 | Ingestion | ไฟล์ภาษาไทยเป็น cp874 ไม่ใช่ UTF-8 | ✅ แก้แล้ว (ทดสอบผ่าน) | `src/ingest.py` |
| 2 | Chunking | ตัด chunk กลางขั้นตอนแก้ปัญหา | ✅ แก้แล้ว (ทดสอบผ่าน) | `src/chunking.py` |
| 3 | Embedding | Rate limit ทำ job ล้มทั้งหมด | ✅ แก้แล้ว (batch + retry) | `src/embedding_store.py` |
| 4 | Vector Store | เอกสารซ้ำ → vector ซ้ำ | ✅ แก้แล้วสำหรับ exact-duplicate / ⚠️ near-duplicate ยังไม่แก้ | `src/embedding_store.py` |
| 4 | Vector Store | Index ถูกสร้างใหม่ทุกครั้ง | ✅ แก้แล้ว (save/load) | `src/embedding_store.py` |
| 5 | Retrieval | vector-only พลาดคำตอบที่ตรงคำ | ✅ แก้แล้ว (hybrid BM25+vector) | `src/retriever.py` |
| 6 | Generation | Hallucination เมื่อไม่มีคำตอบในเอกสาร | ✅ แก้แล้ว (strict prompt + fixed fallback) | `src/generator.py` |
| 6 | Generation | Context ยาวเกินจำเป็น | ✅ แก้แล้ว (truncate ที่ขอบ chunk) | `src/generator.py` |
| 6 | Generation | ตรวจสอบที่มาคำตอบไม่ได้ | ✅ แก้แล้ว (source citation) | `src/generator.py` |

## วิธีรันการทดสอบซ้ำ

```bash
python eval/evaluate.py all
```
คำสั่งนี้จะรันการทดสอบข้อ 1, 2, 4 (dedup) จริงโดยไม่ต้องใช้ API key ส่วนข้อ hallucination guard ต้องตั้งค่า `OPENAI_API_KEY` ก่อน
