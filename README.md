# Document Question Answering (2A)

A small Retrieval-Augmented Generation (RAG) app for asking questions about PDF documents. It splits pages into overlapping text chunks, embeds them with `all-MiniLM-L6-v2`, retrieves the closest passages, then asks Google's Gemma model to answer from those passages with page citations.

## Features

- Upload a text-based PDF or load the included sample syllabus.
- Retrieve the three most relevant chunks using cosine similarity.
- Generate concise, document-grounded answers with page citations.
- Show retrieved passages and similarity scores for transparency.
- Run in demo mode without an API key; retrieval still works and displays a relevant passage.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. Set your Google AI Studio key as an environment variable (never commit it):

   ```bash
   export GOOGLE_API_KEY="your-key"  # Windows PowerShell: $env:GOOGLE_API_KEY="your-key"
   ```

   The app can also start without a key in demo mode.

4. Start the app:

   ```bash
   python app.py
   ```

Open the local URL printed by Gradio. The first run downloads the embedding model. Scanned/image-only PDFs need OCR before this app can read them.

## Configuration

`GEMMA_MODEL` can optionally override the default `gemma-4-26b-a4b-it` model. Chunk size, overlap, and retrieval count are set near the top of `app.py`.

## Security

Keep API keys in environment variables or a local `.env` file that is excluded by `.gitignore`. If a key has ever been committed or shared, revoke it and create a replacement.

## Based on

The flow and model choices follow the [HacktoberFest HackDay RAG starter kit](https://github.com/mithun-srinivas/HacktoberFest-HackDay/blob/main/templates/02_rag_starter_kit.ipynb). This repository packages the working 2A PDF-question-answering path as a standalone Gradio app.
