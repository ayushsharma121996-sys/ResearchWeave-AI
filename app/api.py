"""
FastAPI app for ResearchPilot AI.

Endpoints:
  POST /papers/upload      -> upload PDFs, builds/rebuilds & persists the index
  POST /ask                -> grounded Q&A against the current index
  POST /compare             -> run full extraction + claim matrix + report
  GET  /health              -> liveness check
"""
from contextlib import asynccontextmanager
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import PAPERS_DIR, INDEX_DIR
from app.pipeline import ingest_papers, build_index, run_full_pipeline, answer_question
from app.retrieval.index import HybridIndex

# In-memory session state backed by disk persistence.
_state = {"index": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = INDEX_DIR / "index.pkl"
    if path.exists():
        try:
            index = HybridIndex()
            index.load("index")
            _state["index"] = index
            print(f"Loaded existing index from {path} ({len(index.chunks)} chunks).")
        except Exception as e:
            print(f"Warning: Failed to load index from {path}: {e}")
    yield


app = FastAPI(title="ResearchPilot AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class CompareRequest(BaseModel):
    topic: str = "the uploaded papers"


@app.get("/health")
def health():
    indexed = _state["index"] is not None
    chunks_count = len(_state["index"].chunks) if indexed else 0
    return {"status": "ok", "index_loaded": indexed, "total_chunks": chunks_count}


@app.post("/papers/upload")
async def upload_papers(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files uploaded.")

    saved_paths = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename} is not a PDF.")
        dest = PAPERS_DIR / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dest)

    try:
        chunks = ingest_papers(saved_paths)
        if not chunks:
            raise HTTPException(400, "No text chunks could be extracted from uploaded PDFs.")
        index = build_index(chunks)
        index.save("index")
        _state["index"] = index

        paper_ids = sorted({c.paper_id for c in chunks})
        return {
            "papers_ingested": paper_ids,
            "total_chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(500, f"Error indexing papers: {str(e)}")


@app.post("/ask")
def ask(req: AskRequest):
    if _state["index"] is None:
        raise HTTPException(400, "No papers indexed yet. Upload papers via /papers/upload first.")
    if not req.question.strip():
        raise HTTPException(400, "Question string cannot be empty.")
    try:
        return answer_question(_state["index"], req.question)
    except Exception as e:
        raise HTTPException(500, f"Error answering question: {str(e)}")


@app.post("/compare")
def compare(req: CompareRequest):
    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(400, "No papers found in paper repository. Upload PDFs first.")

    try:
        result = run_full_pipeline(pdf_paths, topic=req.topic)
        result["index"].save("index")
        _state["index"] = result["index"]

        return {
            "report_markdown": result["report"],
            "num_papers": len({e.paper_id for e in result["extractions"]}),
            "num_claims_analyzed": len(result["claim_matrix"]),
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating comparison report: {str(e)}")

