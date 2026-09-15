"""
Evaluation harness.

Measures the two things that actually matter for a RAG system and that are
easy to fool yourself about by eyeballing outputs:

  1. Retrieval quality: for each labeled question, does the retrieved chunk
     set contain the chunk that actually has the answer? (recall@k)
  2. Answer faithfulness: does the generated answer only state things that
     are supported by the retrieved chunks, or does it hallucinate beyond
     them? Judged by a separate LLM call acting as a grader -- a common and
     reasonably reliable pattern (LLM-as-judge) as long as the grading
     prompt is strict and the judgments are spot-checked by hand.

Usage:
    python -m app.eval.run_eval --eval-set app/eval/eval_set.json

The eval set format is documented in eval_set.example.json. You populate it
by hand-reading a handful of papers and writing down real Q&A pairs with the
paper/section where the answer lives -- there's no shortcut for this step,
and that's the point: a self-generated eval set is one of the strongest
signals of rigor in a portfolio project.
"""
import argparse
import json
from pathlib import Path
from typing import List

from app.retrieval.index import HybridIndex
from app.pipeline import answer_question
from app.llm import call_llm_json
from app.config import MODEL_FAST

FAITHFULNESS_SYSTEM = """You are a strict grader. Given a set of source \
excerpts and a generated answer, determine if the answer is fully supported \
by the excerpts (no hallucinated facts, no unsupported numbers/claims). \
Respond with JSON: {"faithful": true|false, "unsupported_claims": ["..."]}"""


def eval_retrieval(index: HybridIndex, eval_items: List[dict], top_k: int = 6) -> dict:
    """recall@k: fraction of questions where the labeled source chunk's
    paper+section appears somewhere in the top_k retrieved results."""
    hits = 0
    for item in eval_items:
        results = index.search(item["question"], top_k=top_k)
        found = any(
            r.paper_id == item["expected_paper_id"] and r.section == item.get("expected_section")
            for r in results
        )
        hits += int(found)
    return {"recall_at_k": hits / len(eval_items) if eval_items else 0.0, "k": top_k}


def eval_faithfulness(index: HybridIndex, eval_items: List[dict]) -> dict:
    faithful_count = 0
    details = []
    for item in eval_items:
        result = answer_question(index, item["question"])
        context_text = result.get("context", "")
        grading_prompt = (
            f"Source excerpts:\n{context_text}\n\n"
            f"Generated answer: {result['answer']}\n\n"
            "Judge faithfulness as instructed."
        )
        try:
            verdict = call_llm_json(FAITHFULNESS_SYSTEM, grading_prompt, model=MODEL_FAST, max_tokens=300)
        except ValueError:
            verdict = {"faithful": False, "unsupported_claims": ["grading_failed"]}
        faithful_count += int(verdict.get("faithful", False))
        details.append({"question": item["question"], "verdict": verdict})

    return {
        "faithfulness_rate": faithful_count / len(eval_items) if eval_items else 0.0,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument("--index-name", type=str, default="index")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    eval_items = json.loads(Path(args.eval_set).read_text())

    index = HybridIndex()
    index.load(args.index_name)

    retrieval_results = eval_retrieval(index, eval_items, top_k=args.top_k)
    faithfulness_results = eval_faithfulness(index, eval_items)

    print(json.dumps({
        "retrieval": retrieval_results,
        "faithfulness": {
            "faithfulness_rate": faithfulness_results["faithfulness_rate"]
        },
    }, indent=2))

    # Save full details separately for manual spot-checking
    out_path = Path(args.eval_set).parent / "last_eval_details.json"
    out_path.write_text(json.dumps(faithfulness_results["details"], indent=2))
    print(f"\nFull faithfulness details written to {out_path}")


if __name__ == "__main__":
    main()
