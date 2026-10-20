"""Small retrieval metrics for manually labeled Papertrail queries."""

from dataclasses import dataclass

from .pdfs import TextChunk


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    source: str
    page: int
    answerable: bool = True


def score_case(results: list[tuple[TextChunk, float]], case: EvaluationCase) -> tuple[bool, float]:
    """Return hit@k and reciprocal rank for one expected source page."""
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.source == case.source and chunk.page == case.page:
            return True, 1 / rank
    return False, 0.0
