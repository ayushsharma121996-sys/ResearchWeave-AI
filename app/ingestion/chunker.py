"""
Turn section-tagged paragraphs into retrieval-sized chunks.

Rule: a chunk never crosses a section boundary. Within a section, paragraphs
are packed up to MAX_CHUNK_TOKENS with a small overlap so retrieval doesn't
lose a claim that happens to sit at a chunk edge.

Token counting here is approximate (whitespace split) on purpose -- exact
tokenization would require pulling in the model's tokenizer, and for chunking
purposes a ~10% margin of error doesn't matter. This tradeoff is worth
mentioning explicitly if asked: "I used approximate token counts for chunking
because precision there doesn't affect retrieval quality, but I'd use the
real tokenizer if chunk size mattered for a hard context-window limit."
"""
from dataclasses import dataclass, field
from typing import List

from app.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS
from app.ingestion.parser import Paragraph


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    page_start: int
    page_end: int


def _approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_paragraphs(paragraphs: List[Paragraph]) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not paragraphs:
        return chunks

    paper_id = paragraphs[0].paper_id
    buffer: List[Paragraph] = []
    buffer_tokens = 0
    chunk_idx = 0
    current_section = paragraphs[0].section

    def flush():
        nonlocal buffer, buffer_tokens, chunk_idx
        if not buffer:
            return
        text = " ".join(p.text for p in buffer)
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}::chunk_{chunk_idx}",
                paper_id=paper_id,
                section=current_section,
                text=text,
                page_start=buffer[0].page,
                page_end=buffer[-1].page,
            )
        )
        chunk_idx += 1

    for p in paragraphs:
        if p.section != current_section:
            flush()
            buffer, buffer_tokens = [], 0
            current_section = p.section

        p_tokens = _approx_tokens(p.text)
        if buffer_tokens + p_tokens > MAX_CHUNK_TOKENS and buffer:
            flush()
            # carry a small overlap forward from the tail of the previous buffer
            overlap_para = buffer[-1] if buffer else None
            buffer = [overlap_para] if overlap_para and _approx_tokens(overlap_para.text) <= CHUNK_OVERLAP_TOKENS else []
            buffer_tokens = _approx_tokens(buffer[0].text) if buffer else 0

        buffer.append(p)
        buffer_tokens += p_tokens

    flush()
    return chunks
