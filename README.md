# SIT Student Policy Advisor (HacktoberFest 2A)

A Gradio retrieval-augmented generation app for student policy questions. Upload at least two text-based official PDFs; the app indexes their pages, retrieves relevant passages, and asks Google's Gemma model to answer from those passages with document and page citations.

## Features

- Upload and index multiple PDFs, with document names and passage counts shown in the status panel.
- Cite the source PDF and page in answers, and display the retrieved passages and similarity scores.
- Refuse questions when the best retrieval score is below the configurable MIN_SIMILARITY threshold.
- Provide a 15-row evaluation worksheet for recording expected answers, app answers, correctness, and failure notes. The worksheet does not claim an accuracy result until the team fills and reviews it.
- Explain the pipeline and current design settings in the application.
- Use the black, white, yellow, and orange interface theme.

## Run locally

1. Install Python 3.10 or newer.
2. Create a virtual environment and install dependencies: python -m venv .venv, activate it, then run pip install -r requirements.txt.
3. Set GOOGLE_API_KEY in your environment to a Google AI Studio key. Never commit the key. Without it, the app runs in retrieval-only mode and shows passages rather than generated answers.
4. Start the app with python app.py and open the local URL printed by Gradio. The first run downloads the embedding model.

## Pipeline settings

The app reads selectable text from PDFs with pypdf, splits each page into overlapping 120-word chunks with 30 words of overlap, embeds chunks using sentence-transformers/all-MiniLM-L6-v2, and retrieves the top 3 passages by cosine similarity. MIN_SIMILARITY defaults to 0.28 and should be checked against the team's own test set and documents.

Scanned or image-only PDFs need OCR before they can be indexed. The 15-question worksheet starts blank by design: use the documents selected for the showcase, manually verify each answer, and document the main failure cases before reporting accuracy.

## Security

Keep API keys in environment variables or a local .env file excluded by .gitignore. If a key has ever been committed or shared, revoke it and create a replacement. Do not upload private or copyrighted documents to the public repository; upload test PDFs directly in the running app when permitted.

## Based on

The pipeline and model choices follow the [HacktoberFest HackDay RAG starter kit](https://github.com/mithun-srinivas/HacktoberFest-HackDay/blob/main/templates/02_rag_starter_kit.ipynb).
