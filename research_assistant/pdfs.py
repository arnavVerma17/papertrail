"""PDF text extraction and page-aware chunking."""

from dataclasses import dataclass
import re

import fitz


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    page: int


_QUESTION_HEADING = re.compile(
    r"^\s*\d{1,3}[.)]\s+(?:explain|define|discuss|what|how|compare|differentiate|"
    r"describe|list|state|write|give|illustrate|analyze|analyse|identify|"
    r"distinguish|elaborate|show|mention|outline|why|when|where|which|who|"
    r"summari[sz]e|examine|evaluate|demonstrate)\b",
    re.IGNORECASE,
)


def _page_sections(text: str) -> list[str]:
    """Keep numbered question-and-answer sections together when a PDF has them."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    headings = [index for index, line in enumerate(lines) if _QUESTION_HEADING.match(line)]
    if not headings:
        return [" ".join(lines)]

    starts = ([0] if headings[0] > 0 else []) + headings
    sections = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        section = " ".join(lines[start:end]).strip()
        if section:
            sections.append(section)
    return sections


def extract_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_words: int = 180,
    overlap: int = 35,
    ocr: bool = False,
    ocr_language: str = "eng",
) -> list[TextChunk]:
    """Extract page text, optionally OCRing pages with little selectable text."""
    chunks: list[TextChunk] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if ocr and len(text) < 60:
                text_page = page.get_textpage_ocr(
                    language=ocr_language,
                    dpi=200,
                    full=not bool(text),
                )
                text = page.get_text("text", textpage=text_page).strip()
            step = max(1, chunk_words - overlap)
            for section in _page_sections(text):
                words = section.split()
                for start in range(0, len(words), step):
                    piece = " ".join(words[start : start + chunk_words]).strip()
                    if piece:
                        chunks.append(TextChunk(piece, filename, page_number))
                    if start + chunk_words >= len(words):
                        break
    return chunks
