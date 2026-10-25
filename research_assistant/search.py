"""Local keyword and semantic search over extracted document chunks."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

from .pdfs import TextChunk


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SEMANTIC_RELEVANCE = 0.28


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load a compact sentence embedding model (downloaded on first use)."""
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[TextChunk], model: SentenceTransformer):
    """Encode passages once when documents are indexed."""
    if not chunks:
        return None
    return model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def semantic_search_chunks(chunks: list[TextChunk], query: str, embeddings, model: SentenceTransformer, limit: int = 5) -> list[tuple[TextChunk, float]]:
    """Rank semantic matches with an exact-term boost and discard weak hits."""
    if not chunks or not query.strip() or embeddings is None:
        return []
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    semantic_scores = embeddings @ query_embedding
    query_terms = [
        token for token in re.findall(r"\w+", query.casefold())
        if len(token) > 1 and token not in ENGLISH_STOP_WORDS
    ]
    normalized_query = " ".join(query_terms)
    scores = []
    for semantic_score, chunk in zip(semantic_scores, chunks):
        text_terms = re.findall(r"\w+", chunk.text.casefold())
        text_set = set(text_terms)
        term_coverage = (
            sum(term in text_set for term in query_terms) / len(query_terms)
            if query_terms else 0.0
        )
        normalized_text = " ".join(text_terms)
        phrase_match = bool(normalized_query and normalized_query in normalized_text)
        relevance = 0.60 * float(semantic_score) + 0.25 * term_coverage + 0.15 * phrase_match
        scores.append(relevance)

    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranked = [index for index in ranked if scores[index] >= MIN_SEMANTIC_RELEVANCE]
    return [(chunks[i], float(scores[i])) for i in ranked[:limit]]


def extract_grounded_answer(results: list[tuple[TextChunk, float]], query: str, model: SentenceTransformer, max_sentences: int = 3, min_score: float = 0.28) -> list[tuple[str, TextChunk, float]]:
    """Select source sentences that semantically support a query, with provenance."""
    candidates: list[tuple[str, TextChunk]] = []
    for chunk, _ in results:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            sentence = sentence.strip()
            if len(sentence.split()) >= 6:
                candidates.append((sentence, chunk))
    if not candidates:
        return []
    vectors = model.encode(
        [query] + [sentence for sentence, _ in candidates],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores = vectors[1:] @ vectors[0]
    ranked = scores.argsort()[::-1]
    selected: list[tuple[str, TextChunk, float]] = []
    seen: set[str] = set()
    for index in ranked:
        sentence, chunk = candidates[int(index)]
        normalized = sentence.casefold()
        if scores[index] < min_score:
            break
        if normalized in seen:
            continue
        selected.append((sentence, chunk, float(scores[index])))
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break
    return selected


def search_chunks(chunks: list[TextChunk], query: str, limit: int = 5) -> list[tuple[TextChunk, float]]:
    """Return the most relevant chunks and their cosine similarity scores."""
    if not chunks or not query.strip():
        return []
    corpus = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = scores.argsort()[::-1]
    return [(chunks[int(i)], float(scores[i])) for i in ranked[:limit] if scores[i] > 0]
