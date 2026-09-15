"""
Orchestrates the full ResearchPilot pipeline:
  PDFs -> parse -> chunk -> index -> extract -> claim-match -> report

Kept as plain functions (not a class) so each stage can be called and
inspected independently -- useful both for debugging and for the eval
harness, which needs to measure retrieval and extraction quality separately.
"""
from pathlib import Path
from typing import List

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_paragraphs, Chunk
from app.retrieval.index import HybridIndex
from app.extraction.extractor import extract_all, PaperExtraction
from app.claims.matcher import build_claim_matrix, ClaimMatrixEntry
from app.report.generator import generate_report


def ingest_papers(pdf_paths: List[Path]) -> List[Chunk]:
    """Parse + chunk every PDF. paper_id is derived from filename stem."""
    all_chunks: List[Chunk] = []
    for path in pdf_paths:
        paper_id = Path(path).stem
        paragraphs = parse_pdf(path, paper_id)
        chunks = chunk_paragraphs(paragraphs)
        all_chunks.extend(chunks)
    return all_chunks


def build_index(chunks: List[Chunk]) -> HybridIndex:
    index = HybridIndex()
    index.build(chunks)
    return index


def run_full_pipeline(pdf_paths: List[Path], topic: str = "the uploaded papers"):
    """
    Runs everything end to end and returns all intermediate artifacts, so a
    caller (API, CLI, eval script) can inspect any stage without re-running
    the whole thing.
    """
    chunks = ingest_papers(pdf_paths)
    index = build_index(chunks)

    paper_ids = sorted({c.paper_id for c in chunks})
    extractions: List[PaperExtraction] = extract_all(paper_ids, index)
    claim_matrix: List[ClaimMatrixEntry] = build_claim_matrix(extractions, index)
    report = generate_report(extractions, claim_matrix, topic=topic)

    return {
        "index": index,
        "extractions": extractions,
        "claim_matrix": claim_matrix,
        "report": report,
    }


def answer_question(index: HybridIndex, question: str, model=None) -> dict:
    """Simple grounded Q&A path (not the full comparison pipeline) --
    used for ad hoc questions against an already-built index."""
    from app.llm import call_llm
    from app.config import MODEL_QA

    chunks = index.search(question, top_k=6)
    context = "\n\n".join(
        f"[{c.paper_id} | {c.section} | p.{c.page_start}] {c.text}" for c in chunks
    )
    system = (
        "You are a research assistant. Answer ONLY using the provided excerpts. "
        "Cite the paper_id for every claim. If the excerpts don't contain the "
        "answer, say so explicitly rather than guessing."
    )
    prompt = f"Question: {question}\n\nExcerpts:\n---\n{context}\n---"
    answer = call_llm(system, prompt, model=model or MODEL_QA, max_tokens=800)
    return {
        "answer": answer,
        "context": context,
        "sources": [{"paper_id": c.paper_id, "section": c.section, "page": c.page_start} for c in chunks],
    }
