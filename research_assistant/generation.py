"""Local abstractive answer generation from retrieved passages."""

import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-small"


def load_generator(model_name: str = MODEL_NAME):
    """Load the small local text generation model (downloaded on first use)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_grounded_answer(query: str, results, tokenizer, model) -> tuple[str, list[int]]:
    """Generate an answer constrained to retrieved context and require citations."""
    if not results or results[0][1] < 0.28:
        return "", []

    passages = []
    for index, (chunk, _) in enumerate(results[:4], start=1):
        passages.append(f"[{index}] {chunk.source}, {chunk.citation_location}: {chunk.text[:1400]}")
    prompt = (
        "Answer the question using only the passages below. Do not add facts from memory. "
        "Put a citation like [1] after every factual claim, using only passage numbers shown. "
        "If the passages do not support an answer, reply exactly: INSUFFICIENT EVIDENCE.\n\n"
        f"Question: {query}\n\nPassages:\n" + "\n\n".join(passages)
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=140, num_beams=3, early_stopping=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if "INSUFFICIENT EVIDENCE" in answer.upper():
        return "", []
    citations = sorted({int(value) for value in re.findall(r"\[(\d+)\]", answer) if 1 <= int(value) <= min(4, len(results))})
    if not citations:
        return "", []
    return answer, citations
