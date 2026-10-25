import csv
from collections import Counter
from pathlib import Path

import streamlit as st

from research_assistant.pdfs import TextChunk, extract_document_chunks
from research_assistant.search import embed_chunks, extract_grounded_answer, load_embedding_model, search_chunks, semantic_search_chunks
from research_assistant.evaluation import (
    EvaluationCase,
    group_evaluation_cases,
    parse_evaluation_cases,
    score_query,
    serialize_evaluation_cases,
)
from research_assistant.generation import generate_grounded_answer, load_generator


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
        successful = 0
        with st.spinner("Reading selected documents and rebuilding the library index..."):
            for upload in uploads or []:
                try:
                    extracted = extract_document_chunks(
                        upload.getvalue(),
                        upload.name,
                        ocr=use_ocr,
                        ocr_language=ocr_language.strip() or "eng",
                    )
                    if extracted:
                        updated_documents[upload.name] = extracted
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
                        rebuild_library_index()
                    st.session_state.library_notice = f"Removed {name} from this session's library."
                    st.rerun()
    else:
        st.caption("No documents indexed yet.")
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
    answer_style_input = st.segmented_control(
        "Answer style",
        ["Extractive", "Abstractive"],
        default="Extractive",
        disabled=not chunks,
    )
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

if query.strip():
    if not chunks:
        st.info("Add documents in the sidebar and index them to start searching.")
    else:
        if search_mode == "Semantic":
            if st.session_state.embeddings is None:
                st.warning("Semantic index is unavailable. Re-index the documents after installing the embedding dependency, or switch to TF-IDF.")
                results = []
            else:
                results = semantic_search_chunks(chunks, query, st.session_state.embeddings, get_embedding_model(), limit)
            result_sets = [("Semantic matches", results)]
        elif search_mode == "TF-IDF":
            results = search_chunks(chunks, query, limit)
            result_sets = [("Keyword matches", results)]
        else:
            keyword_results = search_chunks(chunks, query, limit)
            semantic_results = semantic_search_chunks(chunks, query, st.session_state.embeddings, get_embedding_model(), limit) if st.session_state.embeddings is not None else []
            result_sets = [("Keyword matches · TF-IDF", keyword_results), ("Semantic matches · sentence embeddings", semantic_results)]
        if not any(result for _, result in result_sets):
            st.info("No matching text found. Try different terms or check whether the document contains searchable text.")
        else:
            semantic_results = next((items for title, items in result_sets if title.startswith("Semantic")), None)
            answer_evidence = semantic_results if semantic_results is not None else result_sets[0][1]
            if st.session_state.embeddings is not None and answer_evidence:
                st.markdown("### Evidence-based answer")
                if answer_style == "Extractive":
                    answer = extract_grounded_answer(answer_evidence, query, get_embedding_model())
                    if answer:
                        st.caption("Each sentence is copied from a retrieved passage and linked to its source location.")
                        for sentence, chunk, _ in answer:
                            st.markdown(f"> {sentence}  \n> — *[{chunk.source}, {chunk.citation_location}]*")
                        review_available = True
                    else:
                        st.info("I couldn’t find a passage that clearly supports an answer. Try rephrasing the question or adding more papers.")
                        review_available = False
                else:
                    answer_key = f"{query.strip()}::{search_mode}::{st.session_state.index_version}"
                    if st.button("Generate grounded answer", type="primary"):
                        try:
                            with st.spinner("Loading the local answer model and drafting..."):
                                tokenizer, generator = get_answer_generator()
                                generated_text, cited_passages = generate_grounded_answer(
                                    query, answer_evidence[:4], tokenizer, generator
                                )
                            st.session_state.generated_answers[answer_key] = (generated_text, cited_passages)
                            st.session_state.answer_outcomes[query.strip()] = bool(generated_text)
                        except Exception as error:
                            st.error(f"Local answer generation failed: {error}")
                    generated = st.session_state.generated_answers.get(answer_key)
                    if generated and generated[0]:
                        generated_text, cited_passages = generated
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
            else:
                st.info("Answer drafting needs the semantic index. Re-index documents after the embedding model is available.")
            for heading, results in result_sets:
                if not results:
                    st.warning(f"{heading} is unavailable until semantic indexing succeeds.")
                    continue
                st.markdown(f"#### {heading}")
                st.caption(f"Showing {len(results)} passages, ranked locally by cosine similarity.")
                for chunk, score in results:
                    with st.container(border=True):
                        st.markdown(f"**{chunk.source} · {chunk.citation_location}**  \nCosine similarity · {score:.3f}")
                        st.write(chunk.text)
else:
    st.info("Start by adding a few papers in the sidebar, then search for a concept, method, or finding.")

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
