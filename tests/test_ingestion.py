import pytest
from app.ingestion.parser import Paragraph, _looks_like_heading, _merge_adjacent_lines
from app.ingestion.chunker import chunk_paragraphs, _approx_tokens, Chunk

def test_looks_like_heading():
    assert _looks_like_heading("1. Introduction", font_size=12.0, body_font_size=10.0) == "introduction"
    assert _looks_like_heading("ABSTRACT", font_size=10.0, body_font_size=10.0) == "abstract"
    assert _looks_like_heading("3. Methodology", font_size=14.0, body_font_size=10.0) == "methodology"
    assert _looks_like_heading("This is a normal paragraph sentence.", font_size=10.0, body_font_size=10.0) is None

def test_merge_adjacent_lines():
    paras = [
        Paragraph(paper_id="paper1", section="abstract", text="First line.", page=1),
        Paragraph(paper_id="paper1", section="abstract", text="Second line.", page=1),
        Paragraph(paper_id="paper1", section="methods", text="Method line.", page=1),
    ]
    merged = _merge_adjacent_lines(paras)
    assert len(merged) == 2
    assert merged[0].text == "First line. Second line."
    assert merged[0].section == "abstract"
    assert merged[1].section == "methods"

def test_chunk_paragraphs_never_crosses_sections():
    paras = [
        Paragraph(paper_id="p1", section="abstract", text="Abstract sentence one.", page=1),
        Paragraph(paper_id="p1", section="methods", text="Method sentence one.", page=1),
    ]
    chunks = chunk_paragraphs(paras)
    assert len(chunks) == 2
    assert chunks[0].section == "abstract"
    assert chunks[1].section == "methods"
    assert chunks[0].paper_id == "p1"
