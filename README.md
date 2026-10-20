# Papertrail — grounded research assistant

Papertrail helps you search a set of research PDFs and inspect the page behind each result. It extracts PDF text, keeps page numbers, and ranks passages with TF-IDF or a local sentence embedding model. It can also draft extractive or locally generated answers with source citations.

## Live demo

[Open Papertrail](https://papertrail-lowkey.streamlit.app/)

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run with Docker

Install Docker Desktop, then run:

```powershell
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501). The image includes Tesseract with English language data for optional OCR. A named volume keeps downloaded model files between container rebuilds. The first semantic index or generated answer downloads its model. Stop the app with `docker compose down`; uploaded documents remain in memory for the app session and are not saved in the volume.

The Compose port is bound to localhost. For a remote deployment, put Papertrail behind an access-controlled reverse proxy and forward traffic to port 8501; the app itself does not provide user authentication. To use another OCR language in Docker, add its Tesseract language package to the Dockerfile and rebuild.

Upload PDFs from the sidebar and click **Add / re-index selected PDFs**. New files join the current session library; a selected file with the same name replaces its earlier version. Use **×** beside a document to remove it. Updating the library rebuilds its index and clears evaluation labels and answer reviews tied to the previous document set. Choose semantic search, TF-IDF keyword search, or compare both. Semantic search uses `sentence-transformers/all-MiniLM-L6-v2`; it downloads the model the first time the index is built, then runs locally. Choose **Abstractive (local model)** and click **Generate grounded answer** to draft a cited answer with `google/flan-t5-small`; its model weights download on first use. Files are processed in memory for the current app session. For scanned pages, turn on **OCR scanned pages** before indexing; Tesseract and the selected language data must be installed locally.

## Current features

- Page-aware PDF text extraction
- Overlapping passages to preserve nearby context
- Local TF-IDF and sentence embedding retrieval, with no API key required
- Side-by-side retrieval comparison
- Extractive answer drafts with page citations and an abstain option
- Optional abstractive answer drafts from retrieved passages, with passage citations
- Session based evaluation with Hit@k, MRR, answer coverage, abstention, and citation reviews
- Downloadable and importable CSV evaluation sets tied to exact source filenames and pages
- Manual supported/unsupported answer reviews summarized in-session
- Native, branded light and dark themes
- Batched search form and guided empty-library experience
- Add, replace, and remove documents in the current session library
- Optional OCR for scanned pages through the local Tesseract installation
- Source filename, page number, and relevance score on every result

## Roadmap

1. Add semantic retrieval with sentence embeddings and compare it against TF-IDF. *(Complete)*
2. Add abstractive answer generation from retrieved passages with citations and abstention. *(First local version complete; review outputs against cited sources.)*
3. Expand the evaluation set with more labeled papers and questions. *(Reusable CSV workflow complete; add and review labels from real papers.)*
4. Polish the portfolio demo and deployment workflow. *(Hosted deployment complete.)*

See [DEMO.md](DEMO.md) for a one-minute product walkthrough, architecture diagram, and presentation notes.

## Project layout

```text
app.py                    Streamlit interface
research_assistant/pdfs.py PDF extraction and chunking
research_assistant/search.py Local retrieval
research_assistant/generation.py Local answer generation
research_assistant/evaluation.py Retrieval and answer evaluation
DEMO.md                   Portfolio demo walkthrough
```
