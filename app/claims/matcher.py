"""
Cross-paper claim matching: for every claim extracted from paper A, check
whether papers B..N support it, contradict it, or don't address it.

This is deliberately NOT a single "compare these papers" prompt dumped with
all text at once. That approach degrades badly past ~4-5 papers because the
model has to hold every comparison in its head simultaneously and citations
get unreliable. Instead:

  1. For each claim, retrieve only the chunks from OTHER papers that are
     actually relevant to that specific claim (hybrid search scoped to
     paper_ids != source paper).
  2. Ask the model to judge stance using only those retrieved chunks, one
     claim at a time.
  3. Every stance judgment is required to cite the exact chunk it's based on.

This keeps each LLM call small, grounded, and independently checkable --
which is also why it's straightforward to eval (see app/eval/).
"""
from dataclasses import dataclass, field
from typing import List

from app.llm import call_llm_json
from app.config import MODEL_EXTRACTION, CLAIM_STANCE_LABELS
from app.retrieval.index import HybridIndex
from app.extraction.extractor import PaperExtraction

STANCE_SYSTEM = """You are a careful research fact-checker. Given a claim from \
one paper and excerpts from OTHER papers, determine whether each excerpt \
supports the claim, contradicts it, or does not address it at all. Only mark \
'supports' or 'contradicts' if the excerpt is genuinely and specifically \
relevant -- default to 'not_addressed' when in doubt. Never invent evidence \
that is not in the excerpt."""

STANCE_PROMPT = """Claim (from paper {source_paper}): "{claim_text}"

Excerpts from other papers:
---
{excerpts}
---

For EACH excerpt, judge its stance toward the claim. Respond as JSON:
{{
  "judgments": [
    {{
      "paper_id": "...",
      "stance": "supports" | "contradicts" | "not_addressed",
      "evidence_quote": "short (<20 word) paraphrase of the relevant part of the excerpt, or null",
      "reasoning": "one sentence explaining the judgment"
    }}
  ]
}}"""


@dataclass
class ClaimJudgment:
    paper_id: str
    stance: str
    evidence_quote: str | None
    reasoning: str


@dataclass
class ClaimMatrixEntry:
    claim_id: str
    claim_text: str
    source_paper: str
    judgments: List[ClaimJudgment] = field(default_factory=list)


def build_claim_matrix(
    extractions: List[PaperExtraction],
    index: HybridIndex,
    top_k_per_claim: int = 4,
) -> List[ClaimMatrixEntry]:
    all_paper_ids = [e.paper_id for e in extractions]
    matrix: List[ClaimMatrixEntry] = []

    for extraction in extractions:
        other_paper_ids = [p for p in all_paper_ids if p != extraction.paper_id]
        if not other_paper_ids or not extraction.claims:
            continue

        claims_to_check = extraction.claims[:4]
        claims_with_excerpts = []
        for claim in claims_to_check:
            claim_id = claim.get("claim_id", "c1")
            claim_text = claim.get("text", "")
            if not claim_text:
                continue
            relevant_chunks = index.search(
                claim_text, top_k=top_k_per_claim, paper_ids=other_paper_ids
            )
            if not relevant_chunks:
                continue
            excerpts = "\n".join(
                f"    * [{c.paper_id} | {c.section} | p.{c.page_start}] {c.text}"
                for c in relevant_chunks
            )
            claims_with_excerpts.append((claim_id, claim_text, excerpts))

        if not claims_with_excerpts:
            continue

        claims_prompt_block = "\n\n".join(
            f"Claim [{cid}]: \"{ctext}\"\nExcerpts from other papers:\n{ex}"
            for cid, ctext, ex in claims_with_excerpts
        )

        prompt = (
            f"Source paper: {extraction.paper_id}\n\n"
            f"Below are key claims from {extraction.paper_id} and retrieved excerpts from other papers:\n\n"
            f"{claims_prompt_block}\n\n"
            "For EACH claim, judge stance toward other papers. Respond as a JSON object with key 'claim_evaluations':\n"
            "{\n"
            "  \"claim_evaluations\": [\n"
            "    {\n"
            "      \"claim_id\": \"c1\",\n"
            "      \"judgments\": [\n"
            "        {\n"
            "          \"paper_id\": \"...\",\n"
            "          \"stance\": \"supports\" | \"contradicts\" | \"not_addressed\",\n"
            "          \"evidence_quote\": \"short quote/paraphrase or null\",\n"
            "          \"reasoning\": \"brief reason\"\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        try:
            data = call_llm_json(STANCE_SYSTEM, prompt, model=MODEL_EXTRACTION, max_tokens=1500)
            evals = data.get("claim_evaluations", [])
            for cid, ctext, _ in claims_with_excerpts:
                matching_eval = next((item for item in evals if str(item.get("claim_id")) == str(cid)), None)
                raw_judgments = matching_eval.get("judgments", []) if isinstance(matching_eval, dict) else []
                judgments = [
                    ClaimJudgment(
                        paper_id=j.get("paper_id", "unknown"),
                        stance=j.get("stance") if j.get("stance") in CLAIM_STANCE_LABELS else "not_addressed",
                        evidence_quote=j.get("evidence_quote"),
                        reasoning=j.get("reasoning", ""),
                    )
                    for j in raw_judgments
                ]
                matrix.append(
                    ClaimMatrixEntry(
                        claim_id=f"{extraction.paper_id}::{cid}",
                        claim_text=ctext,
                        source_paper=extraction.paper_id,
                        judgments=judgments,
                    )
                )
        except Exception:
            continue

    return matrix


def summarize_matrix(matrix: List[ClaimMatrixEntry]) -> dict:
    """Quick aggregate stats: how many claims have contradictions, etc."""
    total = len(matrix)
    contested = sum(1 for e in matrix if any(j.stance == "contradicts" for j in e.judgments))
    corroborated = sum(
        1 for e in matrix
        if any(j.stance == "supports" for j in e.judgments)
        and not any(j.stance == "contradicts" for j in e.judgments)
    )
    unaddressed = total - contested - corroborated
    return {
        "total_claims": total,
        "contested_claims": contested,
        "corroborated_claims": corroborated,
        "unaddressed_claims": unaddressed,
    }
