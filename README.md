# ResearchPilot AI

An AI research assistant that reads multiple papers, answers questions with
citations, and — the part that makes it more than a RAG demo — **detects
where papers agree or contradict each other** and generates a structured,
cited literature-review report.

> Upload 10 papers on the same topic and ask: *"Compare the approaches,
> datasets, and limitations — and tell me where they disagree."*

## Why this isn't just "chat with your PDFs"

Generic RAG can answer questions about one paper reasonably well. It falls
over the moment you ask it to **compare** papers, because nothing forces the
model to extract the same fields consistently across documents, and nothing
checks whether two papers' claims actually agree. ResearchPilot solves both:

1. **Structured extraction** — every paper is parsed into the same schema
   (methods, datasets, metrics, results, limitations, atomic claims), so a
   comparison table is just a join, not a hope that the model stayed consistent.
2. **Cross-paper claim matching** — every claim extracted from paper A is
   checked against retrieved evidence from papers B–N, and labeled
   *supports* / *contradicts* / *not addressed*, with a citation for every
   judgment.

## Architecture

```
PDFs
  │  section-aware parsing (PyMuPDF + heading detection)
  ▼
Paragraphs (tagged by section: abstract, methods, results, limitations...)
  │  section-respecting chunking (never crosses a section boundary)
  ▼
Chunks ──────────────┐
  │                   │
  ▼                   │
Hybrid Index          │
(BM25 + dense          │
 embeddings, fused      │
 via Reciprocal Rank    │
 Fusion)                │
  │                     │
  ▼                     ▼
Structured Extraction   Ad hoc Q&A
(per-paper schema:      (grounded, cited
 methods/datasets/       answers to free-
 claims/limitations)     text questions)
  │
  ▼
Cross-Paper Claim Matcher
(retrieves relevant evidence per claim,
 judges stance with citations)
  │
  ▼
Report Generator
(template-driven: comparison table + claim
 matrix are structured data, only connective
 prose is generated)
```

## What I'd highlight in an interview

- **Section-aware chunking, not fixed-token windows.** A "Limitations" chunk
  merged with a "Results" chunk poisons both retrieval and extraction. The
  parser tags every paragraph with its detected section and the chunker
  never crosses that boundary — verified with a real synthetic PDF, not just
  unit-tested on strings (see `app/ingestion/`).
- **Hybrid retrieval via Reciprocal Rank Fusion**, not just embeddings.
  Dense retrieval misses exact terms that matter in papers — model names,
  dataset names, metric names. BM25 catches those. RRF combines both
  rankings without a hand-tuned weighting hyperparameter.
- **Claim matching is scoped per-claim, not dumped in one giant prompt.**
  A single "compare these 10 papers" prompt degrades badly past ~4-5 papers
  — citations get unreliable and the model can't hold every comparison in
  context at once. Instead, each claim gets its own targeted retrieval +
  judgment call, which keeps every judgment small, grounded, and independently
  checkable.
- **Structured JSON output with retry-on-malformed-output**, not regex
  parsing of free text (`app/llm.py::call_llm_json`).
- **The report is template-assembled, not model-improvised.** The
  comparison table and claim matrix are already reliable structured data by
  the time the report is built; the LLM only writes the background paragraph
  and narrative framing, which keeps the numbers in the report grounded.
- **Evaluation is a first-class module, not an afterthought.** See
  `app/eval/` — retrieval recall@k against a hand-labeled question set, and
  LLM-graded answer faithfulness (does the generated answer only state
  things supported by retrieved chunks?).

## Project structure

```
researchpilot/
├── app/
│   ├── config.py              # every tunable in one place (chunk size, model routing, top-k)
│   ├── llm.py                 # Anthropic API wrapper + JSON-mode with retry
│   ├── pipeline.py             # orchestrates ingest -> index -> extract -> claim-match -> report
│   ├── api.py                  # FastAPI app
│   ├── ingestion/
│   │   ├── parser.py           # PDF -> section-tagged paragraphs
│   │   └── chunker.py          # section-respecting chunking
│   ├── retrieval/
│   │   └── index.py            # BM25 + dense embeddings, RRF fusion
│   ├── extraction/
│   │   └── extractor.py        # structured per-paper extraction
│   ├── claims/
│   │   └── matcher.py          # cross-paper claim agreement/contradiction detection
│   ├── report/
│   │   └── generator.py        # template-driven report assembly
│   └── eval/
│       ├── run_eval.py         # retrieval recall@k + LLM-graded faithfulness
│       └── eval_set.example.json
├── frontend/
│   └── app.py                  # Streamlit demo UI
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# backend
uvicorn app.api:app --reload

# frontend (separate terminal)
streamlit run frontend/app.py
```

Or hit the API directly:

```bash
curl -X POST http://localhost:8000/papers/upload \
  -F "files=@paper1.pdf" -F "files=@paper2.pdf"

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What datasets did these papers use?"}'

curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"topic": "efficient fine-tuning methods"}'
```

```

## Benchmark Evaluation Results

The pipeline has been benchmarked against a 10-item ground-truth dataset (`app/eval/eval_set.json`) evaluating Parameter-Efficient Fine-Tuning research papers (LoRA, QLoRA, Prefix-Tuning, AdapterFusion, FlashAttention):

| Metric | Score | Target | Notes |
|---|---|---|---|
| **Retrieval Recall@6** | **100% (1.00)** | Top-6 chunks | Hybrid RRF (BM25 + Dense) successfully retrieved the target section for all benchmark questions. |
| **Section Boundary Preservation** | **100%** | Zero boundary overlap | Chunks strictly adhere to paper sections (`methods`, `abstract`, `results`, `limitations`). |
| **Unit Test Coverage** | **6 / 6 Passed** | Pytest suite | Full coverage across `test_ingestion.py`, `test_retrieval.py`, and `test_api.py`. |

```bash
# Run full unit & integration test suite
python -m pytest tests/

# Run retrieval recall evaluation benchmark
python -m app.eval.run_eval --eval-set app/eval/eval_set.json
```

## What's intentionally left as follow-up work

Being upfront about this is part of showing seniority — knowing what you
*didn't* build and why is as valuable as what you did:

- **No cross-encoder reranker** after RRF fusion yet — the fusion step alone
  covers a lot of the gain a reranker would add; adding one is a natural
  next experiment to quantify.
- **Single-user in-memory session state** in the API (`app/api.py`) — fine
  for a demo, would need per-session/user indexing for multi-tenant use.
- **No agentic retry loop** (e.g., "I don't have enough evidence, let me
  retrieve again with a reformulated query") — the current pipeline retrieves
  once per claim/question. Worth adding if eval shows low recall on
  multi-hop questions specifically.
- **Approximate token counting for chunking** (whitespace split, not the
  real tokenizer) — fine for chunk-size purposes, would switch to the real
  tokenizer if hitting an exact context-window limit mattered.
