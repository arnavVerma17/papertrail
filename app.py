from collections import Counter

import streamlit as st

from research_assistant.pdfs import TextChunk, extract_chunks
from research_assistant.search import embed_chunks, extract_grounded_answer, load_embedding_model, search_chunks, semantic_search_chunks
from research_assistant.evaluation import EvaluationCase, score_case
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
        return f"Indexed {len(st.session_state.chunks)} passages from {len(st.session_state.processed_names)} PDF(s)."
    except Exception as error:
        return f"Keyword search is ready, but semantic indexing failed: {error}"

with st.sidebar:
    st.title("Papertrail")
    st.caption("Your source-grounded research library")
    st.divider()
    with st.expander("Add research PDFs", expanded=not bool(st.session_state.processed_names)):
        uploads = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
        use_ocr = st.checkbox(
            "OCR scanned pages",
            value=False,
            help="Requires Tesseract OCR and its language data to be installed on this computer.",
        )
        ocr_language = st.text_input(
            "OCR language code(s)",
            value="eng",
            disabled=not use_ocr,
            help="Use Tesseract language codes; combine languages with +, such as eng+spa.",
        )
    if st.button("Add / re-index selected PDFs", type="primary", width="stretch", disabled=not uploads):
        updated_documents = dict(st.session_state.document_chunks)
        successful = 0
        with st.spinner("Reading selected documents and rebuilding the library index..."):
            for upload in uploads or []:
                try:
                    extracted = extract_chunks(
                        upload.getvalue(),
                        upload.name,
                        ocr=use_ocr,
                        ocr_language=ocr_language.strip() or "eng",
                    )
                    if extracted:
                        updated_documents[upload.name] = extracted
                        successful += 1
                    else:
                        st.warning(f"No selectable text found in {upload.name}; scanned PDFs need OCR.")
                except Exception as error:
                    if use_ocr:
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
st.markdown("Search by idea, method, or finding. Every result points back to its PDF page.")
st.caption("Local retrieval · session-based documents · no API key required")

chunks = st.session_state.chunks
counts = Counter(chunk.source for chunk in chunks)
if chunks:
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(counts))
    m2.metric("Pages indexed", len({(c.source, c.page) for c in chunks}))
    m3.metric("Passages ready", len(chunks))
else:
    with st.container(border=True):
        st.subheader("Start with your research library")
        st.write("Add one or more PDFs in the sidebar. Papertrail will index selectable text and keep each passage linked to its page.")
        step1, step2, step3 = st.columns(3)
        step1.markdown("**1 · Add papers**\n\nChoose PDFs from your computer.")
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
        st.info("Add PDFs in the sidebar and index them to start searching.")
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
            st.info("No matching text found. Try different terms or check whether the PDF contains selectable text.")
        else:
            semantic_results = next((items for title, items in result_sets if title.startswith("Semantic")), None)
            answer_evidence = semantic_results if semantic_results is not None else result_sets[0][1]
            if st.session_state.embeddings is not None and answer_evidence:
                st.markdown("### Evidence-based answer")
                if answer_style == "Extractive":
                    answer = extract_grounded_answer(answer_evidence, query, get_embedding_model())
                    if answer:
                        st.caption("Each sentence is copied from a retrieved passage and linked to its source page.")
                        for sentence, chunk, _ in answer:
                            st.markdown(f"> {sentence}  \n> — *[{chunk.source}, page {chunk.page}]*")
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
                            with st.expander(f"[{passage_number}] {chunk.source} · page {chunk.page}"):
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
                        st.markdown(f"**{chunk.source} · Page {chunk.page}**  \nCosine similarity · {score:.3f}")
                        st.write(chunk.text)
else:
    st.info("Start by adding a few papers in the sidebar, then search for a concept, method, or finding.")

with st.expander("Evaluate retrieval"):
    st.caption("Add queries with a source page you expect to be relevant. These labels stay in this app session.")
    if not chunks:
        st.info("Index PDFs first to create evaluation cases.")
    else:
        available_pages = sorted({(chunk.source, chunk.page) for chunk in chunks})
        with st.form("evaluation_case_form", clear_on_submit=True):
            eval_query = st.text_input("Test question")
            expected_page = st.selectbox(
                "Expected relevant page",
                available_pages,
                format_func=lambda item: f"{item[0]} · page {item[1]}",
            )
            expected_answerable = st.selectbox(
                "Should these papers support an answer?",
                [True, False],
                format_func=lambda value: "Yes, answerable" if value else "No, should abstain",
            )
            add_case = st.form_submit_button("Add evaluation case")
        if add_case and eval_query.strip():
            st.session_state.evaluation_cases.append(
                EvaluationCase(eval_query.strip(), expected_page[0], expected_page[1], expected_answerable)
            )
            st.success("Evaluation case added.")

        cases = st.session_state.evaluation_cases
        reviews = list(st.session_state.answer_reviews.values())
        if reviews:
            supported = sum(review == "Supported" for review in reviews)
            support_rate = supported / len(reviews)
            st.caption(f"Answer review: {support_rate:.0%} supported · {1 - support_rate:.0%} marked unsupported ({len(reviews)} question(s)).")
        if cases:
            st.write(f"{len(cases)} labeled question(s) in this session")
            for case in cases:
                expectation = "answerable" if case.answerable else "should abstain"
                st.caption(f"“{case.query}” → {case.source}, page {case.page} · {expectation}")
            if st.button("Clear evaluation cases"):
                st.session_state.evaluation_cases = []
                st.rerun()
            eval_k = st.slider("Evaluate top results", 1, 10, 5, key="eval_k")
            if st.button("Run retrieval evaluation", type="primary"):
                answerable_cases = [case for case in cases if case.answerable]
                strategies = [("TF-IDF", lambda q: search_chunks(chunks, q, eval_k))]
                if st.session_state.embeddings is not None:
                    model = get_embedding_model()
                    strategies.append(("Semantic", lambda q: semantic_search_chunks(chunks, q, st.session_state.embeddings, model, eval_k)))
                for strategy_name, retrieve in strategies:
                    if answerable_cases:
                        outcomes = [score_case(retrieve(case.query), case) for case in answerable_cases]
                        hit_rate = sum(hit for hit, _ in outcomes) / len(outcomes)
                        mrr = sum(reciprocal_rank for _, reciprocal_rank in outcomes) / len(outcomes)
                        st.markdown(f"**{strategy_name}** · Hit@{eval_k}: {hit_rate:.0%} · MRR: {mrr:.3f}")
                unanswerable_cases = [case for case in cases if not case.answerable]
                judged_unanswerable = [case for case in unanswerable_cases if case.query in st.session_state.answer_outcomes]
                if judged_unanswerable:
                    false_answers = sum(st.session_state.answer_outcomes[case.query] for case in judged_unanswerable)
                    st.markdown(f"**Abstention check** · Unsupported answer rate on unanswerable questions: {false_answers / len(judged_unanswerable):.0%} ({len(judged_unanswerable)} evaluated)")
                elif unanswerable_cases:
                    st.info("Run the abstractive answer option on your unanswerable questions to measure whether it abstains.")
                judged_answerable = [case for case in answerable_cases if case.query in st.session_state.answer_outcomes]
                if judged_answerable:
                    answered = sum(st.session_state.answer_outcomes[case.query] for case in judged_answerable)
                    st.markdown(f"**Answer coverage** · Answerable questions answered: {answered / len(judged_answerable):.0%} ({len(judged_answerable)} evaluated)")
                reviewed_cases = [case for case in cases if case.query in st.session_state.answer_reviews]
                if reviewed_cases:
                    supported_reviews = sum(st.session_state.answer_reviews[case.query] == "Supported" for case in reviewed_cases)
                    st.markdown(f"**Citation support** · {supported_reviews / len(reviewed_cases):.0%} supported ({len(reviewed_cases)} reviewed answers)")
                if st.session_state.embeddings is None:
                    st.info("Semantic evaluation is unavailable until semantic indexing succeeds.")
        else:
            st.caption("No evaluation cases yet.")

with st.expander("How this first milestone works"):
    st.write("Papertrail extracts selectable text page by page and creates overlapping passages. TF-IDF finds direct keyword overlap; semantic search uses a local sentence embedding model to find related meaning. Extractive answers copy cited source sentences. Optional abstractive answers use a local model on retrieved passages and request citations; review those citations against the source. The evaluator reports retrieval ranking, answer coverage, abstention on unanswerable questions, and manual citation support reviews.")
