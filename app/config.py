"""
Central configuration for ResearchPilot AI.
Keeping every tunable in one place makes the eval story in the README
reproducible: change one constant, rerun eval, compare numbers.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
INDEX_DIR = BASE_DIR / "data" / "index"
EVAL_DIR = BASE_DIR / "app" / "eval"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM models ----------------------------------------------------------
# Route by task complexity. Extraction/claim-matching need strong reasoning;
# simple Q&A can use a cheaper/faster model. This split is one of the things
# worth calling out in an interview: it's a deliberate cost/quality tradeoff,
# not "call the same model for everything."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 100% Free Groq Models (Powered by Groq Llama / Compound AI)
MODEL_EXTRACTION = os.environ.get("MODEL_EXTRACTION", "groq/compound")      # structured extraction, claim matching
MODEL_QA = os.environ.get("MODEL_QA", "groq/compound")              # Q&A / report writing
MODEL_FAST = os.environ.get("MODEL_FAST", "groq/compound")            # fast classification calls


# --- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, no API cost

# --- Chunking --------------------------------------------------------------
# Section-aware chunking, not fixed token windows. Each chunk stays inside a
# single detected section (Abstract, Methods, Results...) so retrieval never
# returns a chunk that straddles two unrelated topics.
MAX_CHUNK_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 40

# --- Retrieval -------------------------------------------------------------
TOP_K_DENSE = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 6          # after RRF fusion + (optional) rerank
RRF_K = 60                # reciprocal rank fusion constant

# --- Claim matching ---------------------------------------------------------
MAX_CLAIMS_PER_PAPER = 8
CLAIM_STANCE_LABELS = ["supports", "contradicts", "not_addressed"]
