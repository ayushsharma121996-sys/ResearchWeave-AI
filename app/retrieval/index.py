"""
Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with Reciprocal
Rank Fusion (RRF).

Why hybrid instead of just embeddings: dense retrieval is great at semantic
similarity but weak at exact terms that matter a lot in research papers --
model names, dataset names, metric names ("BLEU", "GPT-4", "MMLU"). BM25
catches those; embeddings catch paraphrase. RRF combines both rankings
without needing to tune a weighting hyperparameter, which is why it's
preferred over a naive weighted-sum fusion.
"""
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from fastembed import TextEmbedding
    _USE_FASTEMBED = True
except ImportError:
    from sentence_transformers import SentenceTransformer
    _USE_FASTEMBED = False

from app.config import (
    EMBEDDING_MODEL, INDEX_DIR, TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL, RRF_K,
)
from app.ingestion.chunker import Chunk


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


class HybridIndex:
    def __init__(self):
        if _USE_FASTEMBED:
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        else:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks: List[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    # ---- build -----------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Dense
        if _USE_FASTEMBED:
            raw_emb = np.array(list(self._embedder.embed(texts)))
            self._embeddings = _normalize(raw_emb)
        else:
            self._embeddings = self._embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )

        # Sparse
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def save(self, name: str = "index") -> None:
        path = INDEX_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "embeddings": self._embeddings, "bm25": self._bm25},
                f,
            )

    def load(self, name: str = "index") -> None:
        path = INDEX_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self._bm25 = data["bm25"]

    # ---- query -------------------------------------------------------------
    def search(self, query: str, top_k: int = TOP_K_FINAL, paper_ids: List[str] | None = None):
        """Return top_k chunks fused from dense + BM25 rankings via RRF."""
        candidate_idx = list(range(len(self.chunks)))
        if paper_ids:
            candidate_idx = [i for i in candidate_idx if self.chunks[i].paper_id in paper_ids]
        if not candidate_idx:
            return []

        # Dense ranking
        if _USE_FASTEMBED:
            raw_q = np.array(list(self._embedder.embed([query])))[0]
            q_emb = _normalize(raw_q)
        else:
            q_emb = self._embedder.encode([query], normalize_embeddings=True)[0]

        sims = self._embeddings[candidate_idx] @ q_emb
        dense_order = [candidate_idx[i] for i in np.argsort(-sims)[:TOP_K_DENSE]]

        # BM25 ranking
        tokenized_q = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_q)
        bm25_scores_subset = [(i, bm25_scores[i]) for i in candidate_idx]
        bm25_order = [i for i, _ in sorted(bm25_scores_subset, key=lambda x: -x[1])[:TOP_K_BM25]]

        # Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_order):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_order):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [self.chunks[idx] for idx, _ in fused]

