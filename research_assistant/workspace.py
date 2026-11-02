"""Portable, validated workspace backup archives."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import PurePosixPath
import zipfile

from .evaluation import EvaluationCase, parse_evaluation_cases, serialize_evaluation_cases


WORKSPACE_FORMAT = "papertrail-workspace"
WORKSPACE_VERSION = 1
MAX_WORKSPACE_BYTES = 100 * 1024 * 1024
MAX_WORKSPACE_DOCUMENTS = 100
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}


@dataclass(frozen=True)
class WorkspaceBundle:
    document_files: dict[str, bytes]
    evaluation_cases: list[EvaluationCase]
    answer_reviews: dict[str, str]
    extraction_settings: dict[str, dict[str, object]]


def _safe_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    name = path.name
    if not name or name in {".", ".."} or name != normalized:
        raise ValueError(f"Invalid document filename: {filename!r}.")
    if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {filename}.")
    return name


def build_workspace_archive(
    document_files: dict[str, bytes],
    evaluation_cases: list[EvaluationCase],
    answer_reviews: dict[str, str],
    extraction_settings: dict[str, dict[str, object]] | None = None,
) -> bytes:
    """Create an in-memory ZIP containing documents and portable review data."""
    if not document_files:
        raise ValueError("Add at least one document before saving a workspace.")
    if len(document_files) > MAX_WORKSPACE_DOCUMENTS:
        raise ValueError(f"A workspace can contain at most {MAX_WORKSPACE_DOCUMENTS} documents.")

    normalized_files: list[tuple[str, bytes]] = []
    seen_names = set()
    total_bytes = 0
    for original_name, contents in document_files.items():
        name = _safe_filename(original_name)
        if name.casefold() in seen_names:
            raise ValueError(f"Workspace contains duplicate filenames: {name}.")
        seen_names.add(name.casefold())
        if not isinstance(contents, bytes):
            raise ValueError(f"The original file data for {name} is unavailable; upload it again first.")
        total_bytes += len(contents)
        if total_bytes > MAX_WORKSPACE_BYTES:
            raise ValueError("Workspace documents exceed the 100 MB backup limit.")
        normalized_files.append((name, contents))

    manifest_documents = []
    archive_stream = BytesIO()
    with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, (name, contents) in enumerate(normalized_files, start=1):
            archive_path = f"documents/{index:04d}{PurePosixPath(name).suffix.casefold()}"
            archive.writestr(archive_path, contents)
            manifest_documents.append(
                {
                    "name": name,
                    "path": archive_path,
                    "size": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                    "extraction": (extraction_settings or {}).get(name, {}),
                }
            )

        manifest = {
            "format": WORKSPACE_FORMAT,
            "version": WORKSPACE_VERSION,
            "documents": manifest_documents,
            "evaluation_csv": serialize_evaluation_cases(evaluation_cases) if evaluation_cases else "",
            "answer_reviews": {
                query: review
                for query, review in answer_reviews.items()
                if review in {"Supported", "Unsupported"}
            },
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    bundle = archive_stream.getvalue()
    if len(bundle) > MAX_WORKSPACE_BYTES:
        raise ValueError("Compressed workspace exceeds the 100 MB backup limit.")
    return bundle


def load_workspace_archive(archive_bytes: bytes) -> WorkspaceBundle:
    """Validate and read a workspace ZIP without extracting files to disk."""
    if len(archive_bytes) > MAX_WORKSPACE_BYTES:
        raise ValueError("Workspace backup must be 100 MB or smaller.")

    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise ValueError("This file is not a valid Papertrail workspace ZIP.") from error

    with archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Workspace ZIP contains duplicate file entries.")
        if "manifest.json" not in names:
            raise ValueError("This ZIP does not contain a Papertrail workspace manifest.")
        if archive.getinfo("manifest.json").file_size > 1_000_000:
            raise ValueError("Workspace manifest is too large.")
        total_uncompressed = sum(entry.file_size for entry in entries)
        if total_uncompressed > MAX_WORKSPACE_BYTES:
            raise ValueError("Workspace contents exceed the 100 MB restore limit.")

        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Workspace manifest is missing or invalid.") from error
        if manifest.get("format") != WORKSPACE_FORMAT or manifest.get("version") != WORKSPACE_VERSION:
            raise ValueError("This workspace was created by an unsupported Papertrail version.")

        document_entries = manifest.get("documents")
        if not isinstance(document_entries, list) or not document_entries:
            raise ValueError("Workspace manifest does not list any documents.")
        if len(document_entries) > MAX_WORKSPACE_DOCUMENTS:
            raise ValueError(f"A workspace can contain at most {MAX_WORKSPACE_DOCUMENTS} documents.")

        document_files: dict[str, bytes] = {}
        extraction_settings: dict[str, dict[str, object]] = {}
        seen_names = set()
        referenced_paths = {"manifest.json"}
        for item in document_entries:
            if not isinstance(item, dict):
                raise ValueError("Workspace manifest contains an invalid document entry.")
            name = _safe_filename(str(item.get("name", "")))
            if name.casefold() in seen_names:
                raise ValueError(f"Workspace contains duplicate filenames: {name}.")
            seen_names.add(name.casefold())

            archive_path = item.get("path")
            path = PurePosixPath(archive_path) if isinstance(archive_path, str) else None
            if (
                path is None
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 2
                or path.parts[0] != "documents"
                or archive_path not in names
            ):
                raise ValueError(f"Workspace contains an unsafe path for {name}.")
            referenced_paths.add(archive_path)

            contents = archive.read(archive_path)
            if len(contents) != item.get("size"):
                raise ValueError(f"Workspace document failed its size check: {name}.")
            digest = hashlib.sha256(contents).hexdigest()
            if digest != item.get("sha256"):
                raise ValueError(f"Workspace document failed its integrity check: {name}.")
            document_files[name] = contents
            settings = item.get("extraction", {})
            if not isinstance(settings, dict):
                settings = {}
            language = settings.get("language", "eng")
            if (
                not isinstance(language, str)
                or not language
                or len(language) > 32
                or any(not (character.isalnum() or character in "+_-.") for character in language)
            ):
                language = "eng"
            extraction_settings[name] = {
                "ocr": settings.get("ocr") is True,
                "language": language,
            }

        if set(names) != referenced_paths:
            raise ValueError("Workspace ZIP contains files that are not listed in its manifest.")

        evaluation_csv = manifest.get("evaluation_csv", "")
        if not isinstance(evaluation_csv, str):
            raise ValueError("Workspace evaluation data is invalid.")
        evaluation_cases = parse_evaluation_cases(evaluation_csv) if evaluation_csv.strip() else []

        reviews = manifest.get("answer_reviews", {})
        if not isinstance(reviews, dict):
            raise ValueError("Workspace answer reviews are invalid.")
        answer_reviews = {
            str(query): review
            for query, review in reviews.items()
            if isinstance(query, str)
            and isinstance(review, str)
            and review in {"Supported", "Unsupported"}
        }

    return WorkspaceBundle(document_files, evaluation_cases, answer_reviews, extraction_settings)
