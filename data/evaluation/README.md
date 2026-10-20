# Starter evaluation set

`transformer_starter.csv` contains eight unique questions: six answerable and two that should be unanswerable from the two listed papers. One answerable question has relevant evidence on two pages. The page labels use the PDF's one-based page numbers.

When a query appears on multiple CSV rows, each row marks another relevant page. Papertrail groups those labels into one question for Hit@k and MRR.

## Source PDFs

Download these papers from arXiv and rename the files exactly as shown before uploading them to Papertrail:

- `Sentence-BERT.pdf` — [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084)
- `Attention-Is-All-You-Need.pdf` — [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

The PDFs are not included in this repository. In Papertrail, upload and index both files, then import `transformer_starter.csv` in **Evaluate retrieval**. The false-answer rows intentionally have blank source and page fields.

These are starter labels for comparing retrieval rankings. Review the cited pages and adjust any labels if the arXiv PDF edition or page extraction differs.
