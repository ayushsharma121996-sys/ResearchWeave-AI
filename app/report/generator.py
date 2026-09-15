"""
Assemble extraction + claim matrix into a structured literature-review report.

Deliberately template-driven rather than "ask the LLM to write a report from
everything" -- the comparison table and claim matrix are already reliable
structured data at this point, so only the connective prose (background,
narrative framing of contradictions) needs generation. This keeps the report
grounded: numbers/table cells come straight from extraction, not from the
model re-summarizing under time pressure to sound coherent.
"""
from typing import List

from app.llm import call_llm
from app.config import MODEL_QA
from app.extraction.extractor import PaperExtraction, to_comparison_table
from app.claims.matcher import ClaimMatrixEntry, summarize_matrix

BACKGROUND_SYSTEM = """You write concise, neutral academic background \
paragraphs for literature reviews. You only use the information given to you \
-- never introduce outside claims about the field."""


def _generate_background(extractions: List[PaperExtraction]) -> str:
    titles = [e.title_guess or e.paper_id for e in extractions]
    all_methods = sorted({m for e in extractions for m in e.methods})
    prompt = (
        f"Papers being reviewed: {', '.join(titles)}.\n"
        f"Methods appearing across them: {', '.join(all_methods)}.\n\n"
        "Write a 3-4 sentence background paragraph introducing the shared "
        "research area these papers belong to, based only on this information."
    )
    return call_llm(BACKGROUND_SYSTEM, prompt, model=MODEL_QA, max_tokens=300)


def _render_comparison_table(extractions: List[PaperExtraction]) -> str:
    rows = to_comparison_table(extractions)
    header = "| Paper | Methods | Datasets | Metrics | Key Results | Limitations |\n"
    header += "|---|---|---|---|---|---|\n"
    body = "\n".join(
        f"| {r['title']} | {r['methods']} | {r['datasets']} | {r['metrics']} | "
        f"{r['key_results']} | {r['limitations']} |"
        for r in rows
    )
    return header + body


def _render_claim_matrix(matrix: List[ClaimMatrixEntry]) -> str:
    lines = []
    for entry in matrix:
        contradictions = [j for j in entry.judgments if j.stance == "contradicts"]
        supports = [j for j in entry.judgments if j.stance == "supports"]
        if not contradictions and not supports:
            continue  # skip claims nothing else engages with, keep report tight
        lines.append(f"**Claim** ({entry.source_paper}): {entry.claim_text}")
        for j in supports:
            lines.append(f"  - ✅ *Supported* by {j.paper_id}: {j.evidence_quote} — {j.reasoning}")
        for j in contradictions:
            lines.append(f"  - ⚠️ *Contradicted* by {j.paper_id}: {j.evidence_quote} — {j.reasoning}")
        lines.append("")
    return "\n".join(lines) if lines else "_No cross-paper agreement or contradiction detected among extracted claims._"


def generate_report(
    extractions: List[PaperExtraction],
    claim_matrix: List[ClaimMatrixEntry],
    topic: str = "the reviewed papers",
) -> str:
    background = _generate_background(extractions)
    comparison_table = _render_comparison_table(extractions)
    claims_section = _render_claim_matrix(claim_matrix)
    stats = summarize_matrix(claim_matrix)

    all_limitations = sorted({l for e in extractions for l in e.limitations})
    gaps_bullets = "\n".join(f"- {l}" for l in all_limitations[:10]) or "- None extracted."

    report = f"""# Literature Review: {topic}

## Background
{background}

## Methods & Results Comparison
{comparison_table}

## Cross-Paper Claim Analysis
_{stats['total_claims']} claims checked across papers — \
{stats['corroborated_claims']} corroborated, \
{stats['contested_claims']} contested, \
{stats['unaddressed_claims']} not addressed elsewhere._

{claims_section}

## Acknowledged Gaps & Limitations (aggregated from source papers)
{gaps_bullets}

## Open Questions
Based on the contested claims and unaddressed gaps above, the following \
warrant further investigation: areas where papers disagree, and limitations \
acknowledged by multiple papers independently.

---
*Every claim and table cell in this report is traceable to a specific paper \
and section extracted from the uploaded PDFs. See the claim matrix above for \
supporting/contradicting evidence citations.*
"""
    return report
