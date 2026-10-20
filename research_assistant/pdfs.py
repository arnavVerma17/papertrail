"""PDF text extraction and page-aware chunking."""

from dataclasses import dataclass
import re

import fitz


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    page: int


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
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            words = text.split()
            step = max(1, chunk_words - overlap)
            for start in range(0, len(words), step):
                piece = " ".join(words[start : start + chunk_words]).strip()
                if piece:
                    chunks.append(TextChunk(piece, filename, page_number))
                if start + chunk_words >= len(words):
                    break
    return chunks
