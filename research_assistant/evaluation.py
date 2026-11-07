"""Small retrieval metrics for manually labeled Papertrail queries."""

import csv
import io
import math
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
    query_answerability = {}
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
        previous_answerability = query_answerability.get(query)
        if previous_answerability is not None and previous_answerability != answerable:
            raise ValueError(f"Query on row {line_number} has conflicting answerable labels.")
        query_answerability[query] = answerable
        cases.append(EvaluationCase(query, source or None, page, answerable))
    if not cases:
        raise ValueError("CSV contains no evaluation cases.")
    return cases


def group_evaluation_cases(cases: list[EvaluationCase]) -> dict[str, list[EvaluationCase]]:
    """Group source-location labels into one relevance set per query."""
    groups: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        group = groups.setdefault(case.query, [])
        if group and group[0].answerable != case.answerable:
            raise ValueError(f"Query {case.query!r} has conflicting answerable labels.")
        group.append(case)
    return groups


def score_query(
    results: list[tuple[TextChunk, float]], cases: list[EvaluationCase]
) -> tuple[bool, float]:
    """Score a query as relevant when any labeled source page is retrieved."""
    relevant_pages = {
        (case.source, case.page)
        for case in cases
        if case.answerable and case.source and case.page is not None
    }
    if not relevant_pages:
        return False, 0.0
    for rank, (chunk, _) in enumerate(results, start=1):
        if (chunk.source, chunk.page) in relevant_pages:
            return True, 1 / rank
    return False, 0.0


def score_query_metrics(
    results: list[tuple[TextChunk, float]], cases: list[EvaluationCase], k: int
) -> dict[str, float]:
    """Score a ranked list at unique source-location level with binary relevance."""
    if k < 1:
        raise ValueError("Evaluation k must be at least 1.")

    relevant_locations = {
        (case.source, case.page)
        for case in cases
        if case.answerable and case.source and case.page is not None
    }
    ranked_locations = []
    seen_locations = set()
    for chunk, _ in results:
        location = (chunk.source, chunk.page)
        if location not in seen_locations:
            seen_locations.add(location)
            ranked_locations.append(location)
        if len(ranked_locations) == k:
            break

    relevant_ranks = [
        rank for rank, location in enumerate(ranked_locations, start=1)
        if location in relevant_locations
    ]
    precision = len(relevant_ranks) / k
    recall = len(relevant_ranks) / len(relevant_locations) if relevant_locations else 0.0
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_relevant = min(k, len(relevant_locations))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return {
        "hit": float(bool(relevant_ranks)),
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate_retrieval_cases(
    cases, strategies, k: int, max_k: int = 10
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return summary, per-query, and depth-curve metrics for answerable labels."""
    if k < 1:
        raise ValueError("Evaluation k must be at least 1.")
    if max_k < k:
        raise ValueError("Maximum evaluation depth must be at least k.")
    query_groups = {
        query: query_cases
        for query, query_cases in group_evaluation_cases(cases).items()
        if query_cases[0].answerable
    }
    detail_rows = []
    summary_rows = []
    sweep_rows = []
    for strategy_name, retrieve in strategies:
        strategy_rows = []
        sweep_metrics = {depth: [] for depth in range(1, max_k + 1)}
        for query, query_cases in query_groups.items():
            results = retrieve(query)
            metrics = score_query_metrics(results, query_cases, k)
            for depth in sweep_metrics:
                sweep_metrics[depth].append(score_query_metrics(results, query_cases, depth))
            ranked_locations = []
            location_labels = {}
            seen_locations = set()
            for chunk, _ in results:
                location = (chunk.source, chunk.page)
                if location in seen_locations:
                    continue
                seen_locations.add(location)
                ranked_locations.append(location)
                location_labels[location] = chunk.citation_location
                if len(ranked_locations) == k:
                    break
            row = {
                "Retrieval": strategy_name,
                "Question": query,
                "Hit": bool(metrics["hit"]),
                "Precision@k": metrics["precision"],
                "Recall@k": metrics["recall"],
                "MRR": metrics["mrr"],
                "nDCG@k": metrics["ndcg"],
                "Top source locations": "; ".join(
                    f"{source} · {location_labels[(source, page)]}"
                    for source, page in ranked_locations
                ),
            }
            detail_rows.append(row)
            strategy_rows.append(metrics)
        if strategy_rows:
            summary_rows.append({
                "Retrieval": strategy_name,
                "Hit@k": sum(row["hit"] for row in strategy_rows) / len(strategy_rows),
                "Precision@k": sum(row["precision"] for row in strategy_rows) / len(strategy_rows),
                "Recall@k": sum(row["recall"] for row in strategy_rows) / len(strategy_rows),
                "MRR": sum(row["mrr"] for row in strategy_rows) / len(strategy_rows),
                "nDCG@k": sum(row["ndcg"] for row in strategy_rows) / len(strategy_rows),
                "Questions": len(strategy_rows),
            })
            for depth, depth_rows in sweep_metrics.items():
                sweep_rows.append({
                    "Retrieval": strategy_name,
                    "k": depth,
                    "Hit rate": sum(row["hit"] for row in depth_rows) / len(depth_rows),
                    "Recall": sum(row["recall"] for row in depth_rows) / len(depth_rows),
                })
    return summary_rows, detail_rows, sweep_rows
