"""
PDF -> structured text.

The naive approach (dump every page into one blob, split by token count)
loses the one thing that makes academic papers easy to reason about: their
section structure. A "Limitations" chunk and a "Results" chunk that get
merged together will poison both retrieval and extraction.

This module:
  1. Extracts raw text per page with PyMuPDF (keeps layout order reasonably).
  2. Detects section headers with a heuristic (font size + common heading
     vocabulary) and tags every paragraph with its section.
  3. Returns a list of (section, paragraph_text, page_number) tuples that
     downstream chunking can consume.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF (import name aliased for readability below)

# Common section headings in ML/CS papers. Matched case-insensitively,
# allowing for numbering like "3. Related Work" or "IV. Conclusion".
SECTION_HEADINGS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach",
    "experiments", "experimental setup", "results",
    "discussion", "limitations", "conclusion", "conclusions",
    "acknowledgments", "references", "appendix",
]
_HEADING_RE = re.compile(
    r"^\s*(?:[IVXLC0-9]+\.?\s+)?(" + "|".join(SECTION_HEADINGS) + r")\s*$",
    re.IGNORECASE,
)


@dataclass
class Paragraph:
    paper_id: str
    section: str
    text: str
    page: int


def _looks_like_heading(line: str, font_size: float, body_font_size: float) -> str | None:
    """Return the normalized section name if `line` looks like a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    m = _HEADING_RE.match(stripped)
    if m:
        return m.group(1).lower()
    # Fallback: noticeably larger font + short line + title case
    if font_size > body_font_size * 1.15 and len(stripped.split()) <= 6:
        candidate = stripped.lower()
        for heading in SECTION_HEADINGS:
            if heading in candidate:
                return heading
    return None


def parse_pdf(pdf_path: str | Path, paper_id: str) -> List[Paragraph]:
    """
    Parse a PDF into section-tagged paragraphs.

    Returns a flat list of Paragraph objects in reading order. Text before
    the first detected heading is tagged "front_matter" (title/authors/abstract
    lead-in); anything we can't classify falls back to "body".
    """
    doc = fitz.open(str(pdf_path))
    paragraphs: List[Paragraph] = []
    current_section = "front_matter"

    # Estimate body font size from the most common span size in the doc,
    # used as a baseline to detect "this line is a heading because it's bigger".
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(round(span["size"], 1))
    body_font_size = max(set(sizes), key=sizes.count) if sizes else 10.0

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                avg_size = sum(s["size"] for s in spans) / len(spans)

                heading = _looks_like_heading(line_text, avg_size, body_font_size)
                if heading:
                    current_section = heading
                    continue  # heading itself isn't paragraph content

                paragraphs.append(
                    Paragraph(
                        paper_id=paper_id,
                        section=current_section,
                        text=line_text,
                        page=page_num,
                    )
                )
    doc.close()
    return _merge_adjacent_lines(paragraphs)


def _merge_adjacent_lines(paragraphs: List[Paragraph]) -> List[Paragraph]:
    """Merge consecutive same-section, same-page lines into real paragraphs."""
    if not paragraphs:
        return []
    merged: List[Paragraph] = [paragraphs[0]]
    for p in paragraphs[1:]:
        last = merged[-1]
        if p.section == last.section and p.page == last.page and len(last.text) < 800:
            last.text = f"{last.text} {p.text}"
        else:
            merged.append(p)
    return merged
