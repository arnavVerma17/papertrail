"""Saved, source-linked research findings and Markdown export."""

from datetime import datetime, timezone
from uuid import uuid4


MAX_SAVED_FINDINGS = 100
MAX_SOURCES_PER_FINDING = 10
MAX_SOURCE_EXCERPT = 2_000


def create_finding(
    title: str,
    query: str,
    answer: str,
    sources: list[dict[str, object]],
    notes: str = "",
) -> dict[str, object]:
    """Create one validated notebook entry with stable source details."""
    finding = {
        "id": uuid4().hex,
        "title": title.strip() or query.strip(),
        "query": query.strip(),
        "answer": answer.strip(),
        "sources": sources,
        "notes": notes.strip(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return validate_saved_findings([finding])[0]


def validate_saved_findings(value: object) -> list[dict[str, object]]:
    """Validate notebook data before it is restored from an archive."""
    if not isinstance(value, list) or len(value) > MAX_SAVED_FINDINGS:
        raise ValueError(f"A workspace can contain at most {MAX_SAVED_FINDINGS} saved findings.")

    findings = []
    seen_ids = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Workspace contains an invalid notebook entry.")
        string_fields = ("id", "title", "query", "answer", "notes", "saved_at")
        if any(not isinstance(item.get(field), str) for field in string_fields):
            raise ValueError("Workspace notebook entry is missing required text fields.")
        if any(len(item[field]) > limit for field, limit in {
            "id": 64,
            "title": 500,
            "query": 5_000,
            "answer": 10_000,
            "notes": 10_000,
            "saved_at": 64,
        }.items()):
            raise ValueError("Workspace notebook entry exceeds a text size limit.")
        if not item["title"].strip() or not item["query"].strip() or not item["answer"].strip():
            raise ValueError("Workspace notebook entry needs a title, question, and finding.")
        finding_id = item["id"]
        if not finding_id or finding_id in seen_ids:
            raise ValueError("Workspace notebook contains a missing or duplicate entry ID.")
        seen_ids.add(finding_id)

        sources_value = item.get("sources")
        if (
            not isinstance(sources_value, list)
            or not sources_value
            or len(sources_value) > MAX_SOURCES_PER_FINDING
        ):
            raise ValueError("Workspace notebook entry has an invalid number of sources.")
        sources = []
        for source_number, source in enumerate(sources_value, start=1):
            if not isinstance(source, dict) or any(
                not isinstance(source.get(field), str)
                for field in ("filename", "location", "excerpt")
            ):
                raise ValueError("Workspace notebook contains an invalid source citation.")
            if (
                len(source["filename"]) > 500
                or len(source["location"]) > 500
                or len(source["excerpt"]) > MAX_SOURCE_EXCERPT
            ):
                raise ValueError("Workspace notebook citation exceeds a text size limit.")
            citation_number = source.get("citation_number", source_number)
            if (
                not isinstance(citation_number, int)
                or isinstance(citation_number, bool)
                or not 1 <= citation_number <= 10
            ):
                raise ValueError("Workspace notebook contains an invalid citation number.")
            sources.append({
                "filename": source["filename"],
                "location": source["location"],
                "excerpt": source["excerpt"],
                "citation_number": citation_number,
            })

        findings.append({
            "id": finding_id,
            "title": item["title"],
            "query": item["query"],
            "answer": item["answer"],
            "sources": sources,
            "notes": item["notes"],
            "saved_at": item["saved_at"],
        })
    return findings


def render_notebook_markdown(findings: list[dict[str, object]]) -> str:
    """Render saved findings as a portable Markdown research notebook."""
    validated = validate_saved_findings(findings)
    lines = ["# Papertrail Research Notebook", ""]
    if not validated:
        lines.extend(["No findings have been saved yet.", ""])
        return "\n".join(lines)

    for index, finding in enumerate(validated, start=1):
        lines.extend([
            f"## {index}. {finding['title']}",
            "",
            f"**Question:** {finding['query']}",
            f"**Saved:** {finding['saved_at']}",
            "",
            "### Finding",
            "",
            str(finding["answer"]),
            "",
            "### Sources",
            "",
        ])
        sources = finding["sources"]
        if sources:
            for source_number, source in enumerate(sources, start=1):
                lines.extend([
                    f"{source['citation_number']}. **{source['filename']} · {source['location']}**",
                    f"  > {source['excerpt']}",
                ])
        else:
            lines.append("No source passages were attached.")
        lines.extend(["", "### Notes", "", str(finding["notes"]) or "_No notes._", "", "---", ""])
    return "\n".join(lines)
