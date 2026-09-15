import pytest
from pathlib import Path
from app.ingestion.chunker import Chunk
from app.retrieval.index import HybridIndex
from app.config import INDEX_DIR

def test_hybrid_index_search_and_persistence(tmp_path):
    chunks = [
        Chunk(chunk_id="p1::chunk_0", paper_id="p1", section="methods", text="LoRA uses low rank matrix decomposition with rank r=8.", page_start=1, page_end=1),
        Chunk(chunk_id="p2::chunk_0", paper_id="p2", section="methods", text="QLoRA introduces 4-bit NormalFloat NF4 quantization.", page_start=1, page_end=1),
        Chunk(chunk_id="p3::chunk_0", paper_id="p3", section="abstract", text="Prefix Tuning prepends continuous vectors to transformer layers.", page_start=1, page_end=1),
    ]

    index = HybridIndex()
    index.build(chunks)
    assert len(index.chunks) == 3

    results = index.search("low rank decomposition", top_k=2)
    assert len(results) > 0
    assert results[0].paper_id == "p1"

    # Test scoped paper search
    scoped = index.search("quantization", top_k=2, paper_ids=["p2"])
    assert len(scoped) == 1
    assert scoped[0].paper_id == "p2"

    # Test save and load
    index.save("test_index")
    test_pkl = INDEX_DIR / "test_index.pkl"
    assert test_pkl.exists()

    new_index = HybridIndex()
    new_index.load("test_index")
    assert len(new_index.chunks) == 3
    assert new_index.chunks[0].paper_id == "p1"
