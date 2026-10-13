"""
config.py
---------
Central configuration for the RAG system. All tunable parameters live here
so that experiments (e.g. changing chunk size, top-k) don't require digging
through pipeline code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider ---
# "ollama" = free, runs on your own machine, no API key needed.
# "openai" = paid, needs OPENAI_API_KEY.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- OpenAI (used only when LLM_PROVIDER=openai) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = 1536
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# --- Ollama (used only when LLM_PROVIDER=ollama) ---
# Ollama exposes an OpenAI-compatible endpoint at /v1, so we reuse the
# same `openai` Python client for both providers -- only base_url,
# api_key, model names, and embedding dimension differ.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
# LAB03: switched from nomic-embed-text (English-centric) to bge-m3
# (multilingual, supports Thai). bge-m3 outputs 1024-dim vectors.
OLLAMA_EMBEDDING_DIM = int(os.getenv("OLLAMA_EMBEDDING_DIM", 1024))
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

if LLM_PROVIDER == "openai":
    EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
    EMBEDDING_DIM = OPENAI_EMBEDDING_DIM
    CHAT_MODEL = OPENAI_CHAT_MODEL
    LLM_BASE_URL = None  # use OpenAI's default
    LLM_API_KEY = OPENAI_API_KEY
else:
    EMBEDDING_MODEL = OLLAMA_EMBEDDING_MODEL
    EMBEDDING_DIM = OLLAMA_EMBEDDING_DIM
    CHAT_MODEL = OLLAMA_CHAT_MODEL
    LLM_BASE_URL = OLLAMA_BASE_URL
    LLM_API_KEY = "ollama"  # Ollama ignores the key but the client requires one

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# --- Chunking ---
# LAB03: chunks are one constitution section each; CHUNK_SIZE is only the
# limit above which a very long section gets sub-split.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "legal")  # "legal" or "smart"

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.15))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))  # weight: vector vs BM25
# LAB03: when a question does not name a year, search this constitution.
DEFAULT_YEAR = os.getenv("DEFAULT_YEAR", "2560")
BM25_TOKENIZER = os.getenv("BM25_TOKENIZER", "thai")  # "thai" or "whitespace"

# --- Generation ---
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 6000))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", 0.0))

# --- Embedding batching / retry ---
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 16))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", 1.0))
