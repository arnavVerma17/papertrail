import csv
from collections import Counter
import hashlib
from pathlib import Path

import streamlit as st

from research_assistant.pdfs import TextChunk, extract_document_chunks
from research_assistant.search import (
    diversify_by_source,
    embed_chunks,
    extract_grounded_answer,
    load_embedding_model,
    search_chunks,
    semantic_search_chunks,
)
from research_assistant.evaluation import (
    EvaluationCase,
    group_evaluation_cases,
    parse_evaluation_cases,
    score_query,
    serialize_evaluation_cases,
)
from research_assistant.generation import generate_grounded_answer, load_generator
from research_assistant.notebook import (
    MAX_SAVED_FINDINGS,
    create_finding,
    render_notebook_markdown,
)
from research_assistant.workspace import build_workspace_archive, load_workspace_archive


st.set_page_config(page_title="Papertrail | Research workspace", page_icon=":material/menu_book:", layout="wide")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "processed_names" not in st.session_state:
    st.session_state.processed_names = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "evaluation_cases" not in st.session_state:
    st.session_state.evaluation_cases = []
if "answer_reviews" not in st.session_state:
    st.session_state.answer_reviews = {}
if "answer_outcomes" not in st.session_state:
    st.session_state.answer_outcomes = {}
if "generated_answers" not in st.session_state:
    st.session_state.generated_answers = {}
if "index_version" not in st.session_state:
    st.session_state.index_version = 0
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = {
        name: [chunk for chunk in st.session_state.chunks if chunk.source == name]
        for name in st.session_state.processed_names
    }
if "document_files" not in st.session_state:
    st.session_state.document_files = {}
if "document_extraction_settings" not in st.session_state:
    st.session_state.document_extraction_settings = {}
if "saved_findings" not in st.session_state:
    st.session_state.saved_findings = []
if "library_notice" not in st.session_state:
    st.session_state.library_notice = ""
if "active_search" not in st.session_state:
    st.session_state.active_search = {
        "query": "",
        "retrieval": "Semantic",
        "answer_style": "Extractive",
        "limit": 5,
    }


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_answer_generator():
    return load_generator()


def rebuild_library_index():
    """Rebuild the in-session chunk and embedding indexes after document changes."""
    st.session_state.processed_names = list(st.session_state.document_chunks)
    st.session_state.chunks = [
        chunk
        for document_chunks in st.session_state.document_chunks.values()
        for chunk in document_chunks
    ]
    st.session_state.index_version += 1
    st.session_state.evaluation_cases = []
    st.session_state.answer_reviews = {}
    st.session_state.answer_outcomes = {}
    st.session_state.generated_answers = {}
    st.session_state.embeddings = None
    if not st.session_state.chunks:
        return "Library is empty."
    try:
        st.session_state.embeddings = embed_chunks(st.session_state.chunks, get_embedding_model())
        return f"Indexed {len(st.session_state.chunks)} passages from {len(st.session_state.processed_names)} document(s)."
    except Exception as error:
        return f"Keyword search is ready, but semantic indexing failed: {error}"

with st.sidebar:
    st.title("Papertrail")
    st.caption("Your source-grounded research library")
    st.divider()
    with st.expander("Add research documents", expanded=not bool(st.session_state.processed_names)):
        uploads = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "pptx", "txt", "md"],
            accept_multiple_files=True,
        )
        has_pdf_upload = any(Path(upload.name).suffix.casefold() == ".pdf" for upload in uploads or [])
        use_ocr = st.checkbox(
            "OCR scanned PDFs",
            value=False,
            disabled=not has_pdf_upload,
            help="For image-only PDFs. Requires Tesseract OCR and its language data.",
        )
        ocr_language = st.text_input(
            "OCR language code(s)",
            value="eng",
            disabled=not use_ocr,
            help="Use Tesseract language codes; combine languages with +, such as eng+spa.",
        )
    if st.button("Add / re-index selected documents", type="primary", width="stretch", disabled=not uploads):
        updated_documents = dict(st.session_state.document_chunks)
        updated_files = dict(st.session_state.document_files)
        updated_settings = dict(st.session_state.document_extraction_settings)
        successful = 0
        with st.spinner("Reading selected documents and rebuilding the library index..."):
            for upload in uploads or []:
                try:
                    file_bytes = upload.getvalue()
                    extracted = extract_document_chunks(
                        file_bytes,
                        upload.name,
                        ocr=use_ocr,
                        ocr_language=ocr_language.strip() or "eng",
                    )
                    if extracted:
                        updated_documents[upload.name] = extracted
                        updated_files[upload.name] = file_bytes
                        updated_settings[upload.name] = {
                            "ocr": use_ocr and Path(upload.name).suffix.casefold() == ".pdf",
                            "language": ocr_language.strip() or "eng",
                        }
                        successful += 1
                    else:
                        if Path(upload.name).suffix.casefold() == ".pdf":
                            st.warning(f"No selectable text found in {upload.name}; scanned PDFs need OCR.")
                        else:
                            st.warning(f"No searchable text found in {upload.name}.")
                except Exception as error:
                    if use_ocr and Path(upload.name).suffix.casefold() == ".pdf":
                        st.error(f"Could not read {upload.name}: {error}. Check that Tesseract OCR and the '{ocr_language}' language data are installed.")
                    else:
                        st.error(f"Could not read {upload.name}: {error}")
            st.session_state.document_chunks = updated_documents
            if successful:
                st.session_state.document_files = updated_files
                st.session_state.document_extraction_settings = updated_settings
            message = rebuild_library_index() if successful else None
        if message:
            if st.session_state.embeddings is not None:
                st.success(message)
            else:
                st.warning(message)
    st.divider()
    st.subheader("In this session")
    if st.session_state.processed_names:
        for index, name in enumerate(st.session_state.processed_names):
            document_col, action_col = st.columns([4, 1], vertical_alignment="center")
            with document_col:
                st.caption(f"✓ {name}")
            with action_col:
                if st.button(":material/delete:", key=f"remove_document_{index}", help=f"Remove {name}"):
                    with st.spinner(f"Removing {name} and rebuilding the index..."):
                        st.session_state.document_chunks.pop(name, None)
                        st.session_state.document_files.pop(name, None)
                        st.session_state.document_extraction_settings.pop(name, None)
                        rebuild_library_index()
                    st.session_state.library_notice = f"Removed {name} from this session's library."
                    st.rerun()
    else:
        st.caption("No documents indexed yet.")

    with st.expander("Workspace backup", expanded=False):
        st.caption("Save your source files, evaluation labels, answer reviews, and research notebook together. The search index is rebuilt when you restore it.")
        if st.session_state.document_files:
            try:
                workspace_archive = build_workspace_archive(
                    st.session_state.document_files,
                    st.session_state.evaluation_cases,
                    st.session_state.answer_reviews,
                    st.session_state.document_extraction_settings,
                    st.session_state.saved_findings,
                )
            except ValueError as error:
                st.warning(str(error))
            else:
                st.download_button(
                    "Download workspace backup",
                    data=workspace_archive,
                    file_name="papertrail-workspace.zip",
                    mime="application/zip",
                    width="stretch",
                )
        elif st.session_state.document_chunks:
            st.info("The current session has indexed passages but no saved originals. Re-upload the documents to include them in a backup.")
        else:
            st.caption("Index a document first to create a workspace backup.")

        workspace_upload = st.file_uploader(
            "Restore a workspace backup",
            type=["zip"],
            key="workspace_restore_upload",
        )
        has_current_library = bool(st.session_state.document_chunks)
        confirm_replace = True
        if has_current_library:
            confirm_replace = st.checkbox(
                "Replace the current session library and review data",
                value=False,
                key="confirm_workspace_restore",
            )
        if st.button(
            "Restore workspace",
            disabled=not workspace_upload or not confirm_replace,
            width="stretch",
        ):
            try:
                bundle = load_workspace_archive(workspace_upload.getvalue())
                restored_chunks = {}
                for name, file_bytes in bundle.document_files.items():
                    settings = bundle.extraction_settings.get(name, {})
                    extracted = extract_document_chunks(
                        file_bytes,
                        name,
                        ocr=settings.get("ocr") is True,
                        ocr_language=str(settings.get("language", "eng")),
                    )
                    if not extracted:
                        raise ValueError(f"No searchable text could be restored from {name}.")
                    restored_chunks[name] = extracted

                st.session_state.document_files = bundle.document_files
                st.session_state.document_extraction_settings = bundle.extraction_settings
                st.session_state.document_chunks = restored_chunks
                index_message = rebuild_library_index()
                indexed_locations = {(chunk.source, chunk.page) for chunk in st.session_state.chunks}
                restored_cases = [
                    case for case in bundle.evaluation_cases
                    if not case.answerable
                    or (case.source, case.page) in indexed_locations
                ]
                st.session_state.evaluation_cases = restored_cases
                st.session_state.answer_reviews = bundle.answer_reviews
                for finding in st.session_state.saved_findings:
                    st.session_state.pop(f"notebook_notes_{finding['id']}", None)
                st.session_state.saved_findings = bundle.saved_findings
                st.success(
                    f"Restored {len(restored_chunks)} document(s) and "
                    f"{len(st.session_state.chunks)} passages. {index_message}"
                )
                if len(restored_cases) < len(bundle.evaluation_cases):
                    st.warning("Some saved evaluation labels no longer match the restored document locations and were skipped.")
            except Exception as error:
                st.error(f"Could not restore this workspace: {error}")
    if st.session_state.library_notice:
        st.success(st.session_state.library_notice)
        st.session_state.library_notice = ""

st.title("Find the evidence in your papers")
st.markdown("Search by idea, method, or finding. Every result points back to its page, slide, or document passage.")
st.caption("Local retrieval · session-based documents · no API key required")

chunks = st.session_state.chunks
counts = Counter(chunk.source for chunk in chunks)
if chunks:
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(counts))
    m2.metric("Source locations indexed", len({(c.source, c.page) for c in chunks}))
    m3.metric("Passages ready", len(chunks))
else:
    with st.container(border=True):
        st.subheader("Start with your research library")
        st.write("Add PDFs, Word documents, PowerPoint files, or plain text in the sidebar. Papertrail links every passage back to its source location.")
        step1, step2, step3 = st.columns(3)
        step1.markdown("**1 · Add documents**\n\nChoose PDF, Word, PowerPoint, TXT, or Markdown files.")
        step2.markdown("**2 · Build the index**\n\nChoose whether scanned pages need OCR.")
        step3.markdown("**3 · Ask a question**\n\nCompare keyword and semantic evidence.")

st.subheader("Ask your library")
with st.form("paper_search_form"):
    search_input = st.text_input("Research question", placeholder="For example: What methods improved model calibration?")
    retrieval_input = st.segmented_control(
        "Retrieval method",
        ["Semantic", "TF-IDF", "Compare"],
        default="Semantic",
        disabled=not chunks,
    )
    answer_styles = ["Extractive", "Abstractive"]
    if len(counts) >= 2 and st.session_state.embeddings is not None:
        answer_styles.append("Compare sources")
    answer_style_input = st.segmented_control(
        "Answer style",
        answer_styles,
        default="Extractive",
        disabled=not chunks,
    )
    if len(counts) < 2:
        st.caption("Compare sources is available after at least two documents are indexed.")
    elif st.session_state.embeddings is None:
        st.caption("Compare sources requires a working semantic index.")
    limit_input = st.slider("Passages to show", 1, 10, 5, disabled=not chunks)
    search_submitted = st.form_submit_button("Search papers", type="primary", width="stretch")
if search_submitted:
    st.session_state.active_search = {
        "query": search_input.strip(),
        "retrieval": retrieval_input or "Semantic",
        "answer_style": answer_style_input or "Extractive",
        "limit": limit_input,
    }

query = st.session_state.active_search["query"]
search_mode = st.session_state.active_search["retrieval"]
answer_style = st.session_state.active_search["answer_style"]
limit = st.session_state.active_search["limit"]
notebook_answer = ""
notebook_sources: list[dict[str, str]] = []

if query.strip():
    if not chunks:
        st.info("Add documents in the sidebar and index them to start searching.")
    else:
        candidate_limit = len(chunks) if answer_style == "Compare sources" else limit
        semantic_candidates = []
        keyword_candidates = []
        if search_mode == "Semantic":
            if st.session_state.embeddings is None:
                st.warning("Semantic index is unavailable. Re-index the documents after installing the embedding dependency, or switch to TF-IDF.")
            else:
                semantic_candidates = semantic_search_chunks(
                    chunks, query, st.session_state.embeddings, get_embedding_model(), candidate_limit
                )
            result_sets = [("Semantic matches", semantic_candidates[:limit])]
        elif search_mode == "TF-IDF":
            keyword_candidates = search_chunks(chunks, query, candidate_limit)
            result_sets = [("Keyword matches", keyword_candidates[:limit])]
        else:
            keyword_candidates = search_chunks(chunks, query, candidate_limit)
            semantic_candidates = (
                semantic_search_chunks(
                    chunks, query, st.session_state.embeddings, get_embedding_model(), candidate_limit
                )
                if st.session_state.embeddings is not None
                else []
            )
            result_sets = [
                ("Keyword matches · TF-IDF", keyword_candidates[:limit]),
                ("Semantic matches · sentence embeddings", semantic_candidates[:limit]),
            ]
        if not any(result for _, result in result_sets):
            st.info("No matching text found. Try different terms or check whether the document contains searchable text.")
        else:
            answer_evidence = semantic_candidates or keyword_candidates
            if answer_style == "Compare sources":
                answer_evidence = diversify_by_source(answer_evidence, max(limit, 2))
            if st.session_state.embeddings is not None and answer_evidence:
                answer_heading = "### Cross-source comparison" if answer_style == "Compare sources" else "### Evidence-based answer"
                st.markdown(answer_heading)
                source_count = len({chunk.source for chunk, _ in answer_evidence})
                if answer_style == "Compare sources" and source_count < 2:
                    st.info("I couldn’t find relevant evidence in at least two different documents. Try a broader question or add another source.")
                    review_available = False
                elif answer_style == "Compare sources":
                    st.caption(f"Evidence is balanced across {source_count} documents. Check these passages before generating a comparison.")
                    with st.expander("Source comparison evidence", expanded=True):
                        for source_number, (chunk, _) in enumerate(answer_evidence[:4], start=1):
                            st.caption(f"[{source_number}] {chunk.source} · {chunk.citation_location}")
                            st.write(chunk.text)
                    comparison_mode = True
                    answer_key = f"{query.strip()}::{search_mode}::{answer_style}::{st.session_state.index_version}"
                    if st.button("Generate source comparison", type="primary"):
                        try:
                            with st.spinner("Loading the local answer model and comparing sources..."):
                                tokenizer, generator = get_answer_generator()
                                generated_text, cited_passages = generate_grounded_answer(
                                    query,
                                    answer_evidence[:4],
                                    tokenizer,
                                    generator,
                                    comparison=True,
                                )
                            st.session_state.generated_answers[answer_key] = (generated_text, cited_passages)
                            st.session_state.answer_outcomes[query.strip()] = bool(generated_text)
                        except Exception as error:
                            st.error(f"Local source comparison failed: {error}")
                    generated = st.session_state.generated_answers.get(answer_key)
                    if generated and generated[0]:
                        generated_text, cited_passages = generated
                        notebook_answer = generated_text
                        notebook_sources = [
                            {
                                "citation_number": passage_number,
                                "filename": chunk.source,
                                "location": chunk.citation_location,
                                "excerpt": chunk.text[:2_000],
                            }
                            for passage_number in cited_passages[:10]
                            for chunk in [answer_evidence[passage_number - 1][0]]
                        ]
                        st.caption("Locally generated from balanced passages across different documents. Review each citation before relying on the comparison.")
                        st.write(generated_text)
                        for passage_number in cited_passages:
                            chunk = answer_evidence[passage_number - 1][0]
                            with st.expander(f"[{passage_number}] {chunk.source} · {chunk.citation_location}"):
                                st.write(chunk.text)
                        review_available = True
                    else:
                        st.info("Generate a comparison to see a cited draft. Papertrail will abstain if the sources do not support one.")
                        review_available = False
                elif answer_style == "Extractive":
                    answer = extract_grounded_answer(answer_evidence, query, get_embedding_model())
                    if answer:
                        notebook_answer = "\n\n".join(
                            f"{sentence} [{source_number}]"
                            for source_number, (sentence, _, _) in enumerate(answer[:10], start=1)
                        )
                        notebook_sources = [
                            {
                                "citation_number": source_number,
                                "filename": chunk.source,
                                "location": chunk.citation_location,
                                "excerpt": sentence[:2_000],
                            }
                            for source_number, (sentence, chunk, _) in enumerate(answer[:10], start=1)
                        ]
                        st.caption("Each sentence is copied from a retrieved passage and linked to its source location.")
                        for sentence, chunk, _ in answer:
                            st.markdown(f"> {sentence}  \n> — *[{chunk.source}, {chunk.citation_location}]*")
                        review_available = True
                    else:
                        st.info("I couldn’t find a passage that clearly supports an answer. Try rephrasing the question or adding more papers.")
                        review_available = False
                else:
                    answer_key = f"{query.strip()}::{search_mode}::{answer_style}::{st.session_state.index_version}"
                    if st.button("Generate grounded answer", type="primary"):
                        try:
                            with st.spinner("Loading the local answer model and drafting..."):
                                tokenizer, generator = get_answer_generator()
                                generated_text, cited_passages = generate_grounded_answer(
                                    query,
                                    answer_evidence[:4],
                                    tokenizer,
                                    generator,
                                )
                            st.session_state.generated_answers[answer_key] = (generated_text, cited_passages)
                            st.session_state.answer_outcomes[query.strip()] = bool(generated_text)
                        except Exception as error:
                            st.error(f"Local answer generation failed: {error}")
                    generated = st.session_state.generated_answers.get(answer_key)
                    if generated and generated[0]:
                        generated_text, cited_passages = generated
                        notebook_answer = generated_text
                        notebook_sources = [
                            {
                                "filename": chunk.source,
                                "location": chunk.citation_location,
                                "excerpt": chunk.text[:2_000],
                            }
                            for passage_number in cited_passages[:10]
                            for chunk in [answer_evidence[passage_number - 1][0]]
                        ]
                        st.caption("Locally generated from retrieved passages. Review the linked evidence before relying on it.")
                        st.write(generated_text)
                        for passage_number in cited_passages:
                            chunk = answer_evidence[passage_number - 1][0]
                            with st.expander(f"[{passage_number}] {chunk.source} · {chunk.citation_location}"):
                                st.write(chunk.text)
                        review_available = True
                    else:
                        st.info("Generate an answer to see a cited draft. If the passages don’t support one, Papertrail will abstain.")
                        review_available = False
                if review_available:
                    review_key = query.strip()
                    with st.form(f"answer_review_{review_key}"):
                        review_value = st.segmented_control(
                            "Does this draft stay supported by its cited passages?",
                            ["Supported", "Unsupported"],
                        )
                        review_saved = st.form_submit_button("Save review")
                    if review_saved:
                        if review_value:
                            st.session_state.answer_reviews[review_key] = review_value
                            st.success(f"Review saved: {review_value.lower()}.")
                        else:
                            st.warning("Choose whether the draft is supported before saving the review.")
                    elif review_key in st.session_state.answer_reviews:
                        st.caption(f"Your saved review for this question: {st.session_state.answer_reviews[review_key].lower()}.")
                    if len(st.session_state.saved_findings) >= MAX_SAVED_FINDINGS:
                        st.info(f"The notebook is full ({MAX_SAVED_FINDINGS} findings). Remove one before saving another.")
                    else:
                        query_key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
                        with st.form(f"save_finding_{st.session_state.index_version}_{query_key}", clear_on_submit=True):
                            finding_title = st.text_input("Notebook title", value=query, max_chars=500)
                            finding_notes = st.text_area("Your research notes (optional)", height=90, max_chars=10_000)
                            save_finding = st.form_submit_button("Save finding to notebook")
                        if save_finding:
                            st.session_state.saved_findings.insert(
                                0,
                                create_finding(
                                    finding_title,
                                    query,
                                    notebook_answer,
                                    notebook_sources,
                                    finding_notes,
                                ),
                            )
                            st.success("Saved this finding and its source passages to the research notebook.")
            else:
                st.info("Answer drafting needs the semantic index. Re-index documents after the embedding model is available.")
            for heading, results in result_sets:
                if not results:
                    if heading.startswith("Semantic") and st.session_state.embeddings is None:
                        st.warning("Semantic search is unavailable until document indexing succeeds.")
                    else:
                        st.info(f"No relevant passages were found for {heading.lower()}. Try a different query or upload a source that covers it.")
                    continue
                st.markdown(f"#### {heading}")
                is_semantic = heading.startswith("Semantic")
                ranking_note = (
                    "Ranked by semantic similarity and query-term relevance."
                    if is_semantic
                    else "Ranked locally by TF-IDF cosine similarity."
                )
                score_label = "Relevance score" if is_semantic else "Cosine similarity"
                st.caption(f"Showing {len(results)} passages. {ranking_note}")
                for chunk, score in results:
                    with st.container(border=True):
                        st.markdown(f"**{chunk.source} · {chunk.citation_location}**  \n{score_label} · {score:.3f}")
                        st.write(chunk.text)
else:
    st.info("Start by adding a few papers in the sidebar, then search for a concept, method, or finding.")

st.subheader(f"Research notebook · {len(st.session_state.saved_findings)} saved finding(s)")
st.caption("Keep useful answers with their cited evidence and personal notes. Download the Markdown notebook or include it in a workspace ZIP backup.")
if st.session_state.saved_findings:
    notebook_markdown = render_notebook_markdown(st.session_state.saved_findings)
    st.download_button(
        "Download research notebook (.md)",
        data=notebook_markdown.encode("utf-8"),
        file_name="papertrail-research-notebook.md",
        mime="text/markdown",
    )
    for finding in st.session_state.saved_findings:
        with st.expander(str(finding["title"]), expanded=False):
            st.markdown(f"**Question:** {finding['query']}")
            st.markdown("**Saved finding**")
            st.write(finding["answer"])
            sources = finding["sources"]
            if sources:
                st.markdown("**Cited source passages**")
                for source_number, source in enumerate(sources, start=1):
                    st.caption(f"[{source['citation_number']}] {source['filename']} · {source['location']}")
                    st.write(source["excerpt"])
            note_key = f"notebook_notes_{finding['id']}"
            notes = st.text_area(
                "Your notes",
                value=str(finding["notes"]),
                key=note_key,
                height=100,
                max_chars=10_000,
            )
            save_col, remove_col = st.columns(2)
            with save_col:
                if st.button("Save note", key=f"save_note_{finding['id']}", width="stretch"):
                    finding["notes"] = notes
                    st.success("Notebook note saved.")
            with remove_col:
                if st.button("Remove finding", key=f"remove_finding_{finding['id']}", width="stretch"):
                    st.session_state.saved_findings = [
                        item for item in st.session_state.saved_findings
                        if item["id"] != finding["id"]
                    ]
                    st.rerun()
else:
    st.info("Save a supported extractive or generated answer to start your notebook.")

with st.expander("Evaluate retrieval"):
    st.caption("Label a relevant source location for each query. Locations are PDF pages, PowerPoint slides, or numbered document passages; labels stay in this app session.")
    if not chunks:
        st.info("Index documents first to create evaluation cases.")
    else:
        available_pages = sorted({(chunk.source, chunk.page) for chunk in chunks})
        location_labels = {
            (chunk.source, chunk.page): chunk.citation_location
            for chunk in chunks
        }
        with st.form("evaluation_case_form", clear_on_submit=True):
            eval_query = st.text_input("Test question")
            expected_page = st.selectbox(
                "Expected relevant source location",
                [None, *available_pages],
                format_func=lambda item: "No supporting source (unanswerable)" if item is None else f"{item[0]} · {location_labels[item]}",
            )
            expected_answerable = st.selectbox(
                "Should these papers support an answer?",
                [True, False],
                format_func=lambda value: "Yes, answerable" if value else "No, should abstain",
            )
            add_case = st.form_submit_button("Add evaluation case")
        if add_case and eval_query.strip():
            if expected_answerable and expected_page is None:
                st.warning("Choose the expected source location for an answerable question.")
            else:
                source, page = expected_page if expected_page is not None else (None, None)
                new_case = EvaluationCase(eval_query.strip(), source, page, expected_answerable)
                same_query = [
                    case
                    for case in st.session_state.evaluation_cases
                    if case.query == new_case.query
                ]
                if same_query and any(case.answerable != new_case.answerable for case in same_query):
                    st.warning("This query already has a different answerability label.")
                elif any(
                    (case.source, case.page, case.answerable)
                    == (new_case.source, new_case.page, new_case.answerable)
                    for case in same_query
                ):
                    st.info("That query and source location are already labeled.")
                else:
                    st.session_state.evaluation_cases.append(new_case)
                    st.success("Evidence location added to the evaluation set.")

        st.download_button(
            "Download evaluation CSV template",
            data=serialize_evaluation_cases(st.session_state.evaluation_cases),
            file_name="papertrail-evaluation.csv",
            mime="text/csv",
            help="The legacy 'page' column stores a source location ID (PDF page, slide, or document passage).",
        )
        with st.form("evaluation_import_form"):
            uploaded_cases = st.file_uploader("Import labeled evaluation cases", type=["csv"])
            import_cases = st.form_submit_button("Add cases from CSV")
        if import_cases:
            if uploaded_cases is None:
                st.warning("Choose a CSV file first.")
            elif uploaded_cases.size > 1_000_000:
                st.error("Evaluation CSV must be smaller than 1 MB.")
            else:
                try:
                    imported_cases = parse_evaluation_cases(
                        uploaded_cases.getvalue().decode("utf-8-sig")
                    )
                    indexed_pages = {(chunk.source, chunk.page) for chunk in chunks}
                    missing_pages = sorted(
                        {
                            (case.source, case.page)
                            for case in imported_cases
                            if case.answerable and case.source and case.page is not None
                        }
                        - indexed_pages
                    )
                    if missing_pages:
                        missing_label = ", ".join(
                            f"{source} · location {page}" for source, page in missing_pages[:5]
                        )
                        extra = " and more" if len(missing_pages) > 5 else ""
                        st.error(
                            "These source locations are not in the current indexed library: "
                            f"{missing_label}{extra}. Check filenames and location IDs."
                        )
                    else:
                        known_cases = {
                            (case.query, case.source, case.page, case.answerable)
                            for case in st.session_state.evaluation_cases
                        }
                        new_cases = []
                        for case in imported_cases:
                            case_key = (case.query, case.source, case.page, case.answerable)
                            if case_key not in known_cases:
                                new_cases.append(case)
                                known_cases.add(case_key)
                        st.session_state.evaluation_cases.extend(new_cases)
                        st.success(
                            f"Added {len(new_cases)} new case(s); "
                            f"skipped {len(imported_cases) - len(new_cases)} duplicate(s)."
                        )
                except (UnicodeDecodeError, csv.Error, ValueError) as error:
                    st.error(f"Could not import evaluation CSV: {error}")

        cases = st.session_state.evaluation_cases
        reviews = list(st.session_state.answer_reviews.values())
        if reviews:
            supported = sum(review == "Supported" for review in reviews)
            support_rate = supported / len(reviews)
            st.caption(f"Answer review: {support_rate:.0%} supported · {1 - support_rate:.0%} marked unsupported ({len(reviews)} question(s)).")
        if cases:
            case_groups = group_evaluation_cases(cases)
            st.write(
                f"{len(case_groups)} labeled question(s) across "
                f"{len(cases)} evidence label(s) in this session"
            )
            for query, query_cases in case_groups.items():
                expectation = "answerable" if query_cases[0].answerable else "should abstain"
                expected_pages = (
                    ", ".join(
                        f"{case.source}, {location_labels.get((case.source, case.page)) or 'location ' + str(case.page)}"
                        for case in query_cases
                        if case.source and case.page is not None
                    )
                    if query_cases[0].answerable
                    else "no supporting source expected"
                )
                st.caption(f"“{query}” → {expected_pages} · {expectation}")
            if st.button("Clear evaluation cases"):
                st.session_state.evaluation_cases = []
                st.rerun()
            eval_k = st.slider("Evaluate top results", 1, 10, 5, key="eval_k")
            if st.button("Run retrieval evaluation", type="primary"):
                case_groups = group_evaluation_cases(cases)
                answerable_cases = {
                    query: query_cases
                    for query, query_cases in case_groups.items()
                    if query_cases[0].answerable
                }
                strategies = [("TF-IDF", lambda q: search_chunks(chunks, q, eval_k))]
                if st.session_state.embeddings is not None:
                    model = get_embedding_model()
                    strategies.append(("Semantic", lambda q: semantic_search_chunks(chunks, q, st.session_state.embeddings, model, eval_k)))
                for strategy_name, retrieve in strategies:
                    if answerable_cases:
                        outcomes = [
                            score_query(retrieve(query), query_cases)
                            for query, query_cases in answerable_cases.items()
                        ]
                        hit_rate = sum(hit for hit, _ in outcomes) / len(outcomes)
                        mrr = sum(reciprocal_rank for _, reciprocal_rank in outcomes) / len(outcomes)
                        st.markdown(f"**{strategy_name}** · Hit@{eval_k}: {hit_rate:.0%} · MRR: {mrr:.3f}")
                unanswerable_queries = {
                    query for query, query_cases in case_groups.items() if not query_cases[0].answerable
                }
                judged_unanswerable = unanswerable_queries & set(st.session_state.answer_outcomes)
                if judged_unanswerable:
                    false_answers = sum(st.session_state.answer_outcomes[query] for query in judged_unanswerable)
                    st.markdown(f"**Abstention check** · Unsupported answer rate on unanswerable questions: {false_answers / len(judged_unanswerable):.0%} ({len(judged_unanswerable)} evaluated)")
                elif unanswerable_queries:
                    st.info("Run the abstractive answer option on your unanswerable questions to measure whether it abstains.")
                judged_answerable = set(answerable_cases) & set(st.session_state.answer_outcomes)
                if judged_answerable:
                    answered = sum(st.session_state.answer_outcomes[query] for query in judged_answerable)
                    st.markdown(f"**Answer coverage** · Answerable questions answered: {answered / len(judged_answerable):.0%} ({len(judged_answerable)} evaluated)")
                reviewed_queries = set(case_groups) & set(st.session_state.answer_reviews)
                if reviewed_queries:
                    supported_reviews = sum(st.session_state.answer_reviews[query] == "Supported" for query in reviewed_queries)
                    st.markdown(f"**Citation support** · {supported_reviews / len(reviewed_queries):.0%} supported ({len(reviewed_queries)} reviewed answers)")
                if st.session_state.embeddings is None:
                    st.info("Semantic evaluation is unavailable until semantic indexing succeeds.")
        else:
            st.caption("No evaluation cases yet.")

with st.expander("How this first milestone works"):
    st.write("Papertrail extracts text from PDFs, Word and PowerPoint documents, and plain text files, then creates source-linked passages. TF-IDF finds direct keyword overlap; semantic search uses a local sentence embedding model to find related meaning. Extractive answers copy cited source sentences. Optional abstractive answers use a local model on retrieved passages and request citations; review those citations against the source. The evaluator reports retrieval ranking, answer coverage, abstention on unanswerable questions, and manual citation support reviews.")
