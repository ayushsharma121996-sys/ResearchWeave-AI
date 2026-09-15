"""
Structured extraction: turn a paper's raw chunks into a normalized schema
that can be compared across papers.

This is the piece that makes ResearchPilot more than "chat with your PDFs":
generic RAG answers questions well, but it can't produce a comparison TABLE
because nothing forces the model to extract the same fields consistently
across papers. Forcing a fixed schema is what makes comparison possible.
"""
from dataclasses import dataclass, field
from typing import List

from app.llm import call_llm_json
from app.config import MODEL_EXTRACTION, MAX_CLAIMS_PER_PAPER
from app.retrieval.index import HybridIndex

EXTRACTION_SYSTEM = """You are a meticulous research paper analyst. You extract \
structured information from academic paper excerpts. You never invent \
information that is not present in the given text -- if a field is not \
addressed in the excerpts, use null or an empty list, and do not guess.
Every claim you extract must be traceable to the excerpt it came from."""

EXTRACTION_SCHEMA_PROMPT = """Given the following excerpts from a research paper \
(paper_id: {paper_id}), extract the following as a JSON object:

{{
  "title_guess": "best guess at the paper's title, or null",
  "methods": ["short names/descriptions of the core method(s) proposed or used"],
  "datasets": ["dataset names used for evaluation or training"],
  "metrics": ["evaluation metrics reported, e.g. accuracy, F1, BLEU"],
  "key_results": ["1-3 sentence factual statements of headline results, with numbers if given"],
  "limitations": ["limitations the paper itself acknowledges"],
  "claims": [
    {{
      "claim_id": "c1",
      "text": "a short, atomic, checkable statement the paper makes (max ~25 words)",
      "source_section": "the section this came from, e.g. results, limitations"
    }}
  ]
}}

Extract at most {max_claims} of the most important/comparable claims -- prefer \
claims about method effectiveness, generalization, dataset properties, or \
limitations, since those are the ones most useful to compare across papers.

Excerpts:
---
{excerpts}
---

Respond with ONLY the JSON object."""


@dataclass
class PaperExtraction:
    paper_id: str
    title_guess: str | None
    methods: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    key_results: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    claims: List[dict] = field(default_factory=list)  # {claim_id, text, source_section}


def extract_paper(paper_id: str, index: HybridIndex) -> PaperExtraction:
    """
    Pull the most informative chunks for this paper (methods/results/
    limitations sections weighted first) and run structured extraction.
    """
    paper_chunks = [c for c in index.chunks if c.paper_id == paper_id]
    priority_sections = {"methods", "method", "methodology", "approach",
                          "results", "limitations", "abstract", "conclusion"}
    prioritized = sorted(
        paper_chunks,
        key=lambda c: 0 if c.section in priority_sections else 1,
    )
    # cap excerpt length sent to the model to control cost
    excerpt_chunks = prioritized[:20]
    excerpts = "\n\n".join(f"[{c.section} | p.{c.page_start}] {c.text}" for c in excerpt_chunks)

    prompt = EXTRACTION_SCHEMA_PROMPT.format(
        paper_id=paper_id, excerpts=excerpts, max_claims=MAX_CLAIMS_PER_PAPER
    )
    data = call_llm_json(EXTRACTION_SYSTEM, prompt, model=MODEL_EXTRACTION, max_tokens=2000)

    return PaperExtraction(
        paper_id=paper_id,
        title_guess=data.get("title_guess"),
        methods=data.get("methods", []),
        datasets=data.get("datasets", []),
        metrics=data.get("metrics", []),
        key_results=data.get("key_results", []),
        limitations=data.get("limitations", []),
        claims=data.get("claims", []),
    )


def extract_all(paper_ids: List[str], index: HybridIndex) -> List[PaperExtraction]:
    return [extract_paper(pid, index) for pid in paper_ids]


def to_comparison_table(extractions: List[PaperExtraction]) -> List[dict]:
    """Normalize extractions into rows for a paper x field comparison table."""
    rows = []
    for e in extractions:
        rows.append({
            "paper_id": e.paper_id,
            "title": e.title_guess or e.paper_id,
            "methods": "; ".join(e.methods) or "—",
            "datasets": "; ".join(e.datasets) or "—",
            "metrics": "; ".join(e.metrics) or "—",
            "key_results": " | ".join(e.key_results) or "—",
            "limitations": "; ".join(e.limitations) or "—",
        })
    return rows
