"""Small retrieval metrics for manually labeled Papertrail queries."""

import csv
import io
from dataclasses import dataclass

from .pdfs import TextChunk


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    source: str | None
    page: int | None
    answerable: bool = True


EVALUATION_CSV_HEADERS = ("query", "source", "page", "answerable")


def serialize_evaluation_cases(cases: list[EvaluationCase]) -> str:
    """Return labeled cases as a portable UTF-8 CSV string."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(EVALUATION_CSV_HEADERS)
    for case in cases:
        writer.writerow((case.query, case.source or "", case.page or "", str(case.answerable).lower()))
    return output.getvalue()


def parse_evaluation_cases(csv_text: str) -> list[EvaluationCase]:
    """Parse labeled cases from CSV, accepting common boolean spellings."""
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    fieldnames = reader.fieldnames or []
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("CSV contains duplicate column names.")
    headers = set(fieldnames)
    missing = set(EVALUATION_CSV_HEADERS) - headers
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}.")

    cases = []
    boolean_values = {
        "true": True,
        "yes": True,
        "1": True,
        "answerable": True,
        "false": False,
        "no": False,
        "0": False,
        "unanswerable": False,
    }
    for line_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"Row {line_number} has more values than the CSV header.")
        if not any((value or "").strip() for value in row.values()):
            continue
        query = (row.get("query") or "").strip()
        source = (row.get("source") or "").strip()
        answerable_text = (row.get("answerable") or "").strip().lower()
        if answerable_text not in boolean_values:
            raise ValueError(
                f"Row {line_number} answerable must be true/false, yes/no, 1/0, or answerable/unanswerable."
            )
        answerable = boolean_values[answerable_text]
        page_text = (row.get("page") or "").strip()
        if not query:
            raise ValueError(f"Row {line_number} needs a query.")
        if answerable and not source:
            raise ValueError(f"Row {line_number} needs a source filename for an answerable question.")
        if bool(source) != bool(page_text):
            raise ValueError(f"Row {line_number} must include both source and page, or leave both blank.")
        page = None
        if page_text:
            try:
                page = int(page_text)
            except ValueError as error:
                raise ValueError(f"Row {line_number} must have a whole-number page.") from error
            if page < 1:
                raise ValueError(f"Row {line_number} page must be 1 or greater.")
        cases.append(EvaluationCase(query, source or None, page, answerable))
    if not cases:
        raise ValueError("CSV contains no evaluation cases.")
    return cases


def score_case(results: list[tuple[TextChunk, float]], case: EvaluationCase) -> tuple[bool, float]:
    """Return hit@k and reciprocal rank for one expected source page."""
    if not case.answerable or not case.source or case.page is None:
        return False, 0.0
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.source == case.source and chunk.page == case.page:
            return True, 1 / rank
    return False, 0.0
