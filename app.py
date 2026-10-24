"""Student policy question answering with transparent retrieval-augmented generation."""

import os
from pathlib import Path

import gradio as gr
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 30
TOP_K = 3
MIN_SIMILARITY = 0.28
REFUSAL = "The answer could not be found in the documents."
ANSWER_RULES = """Answer questions using ONLY the supplied document passages.
Each passage is labeled with its document filename and PDF page number.
Cite every factual claim using [filename, p. N].
If the answer is not supported by the passages, reply exactly:
The answer could not be found in the documents.
Keep the answer concise, in 2 to 4 sentences."""
CSS = """
body, .gradio-container { background: #050505 !important; color: #f5f5f5 !important; }
.gradio-container { max-width: 1220px !important; }
.app-header { background: linear-gradient(120deg, #111, #252018); border: 1px solid #ffd43b; border-left: 6px solid #ff7619; border-radius: 18px; padding: 25px 30px; margin-bottom: 16px; }
.app-header h1 { color: #ffd43b; margin: 0 0 8px; font-size: 30px; }
.app-header p { color: #eee; margin: 0; }
.panel { background: #121212; border: 1px solid #514314; border-radius: 14px; padding: 16px; }
button.primary { background: #ff7619 !important; color: #111 !important; border: 0 !important; font-weight: 700 !important; }
button { border-radius: 10px !important; }
label, h2, h3 { color: #ffd43b !important; }
input, textarea { background: #1a1a1a !important; color: #fff !important; border: 1px solid #a27b00 !important; }
footer { display: none !important; }
"""

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) if os.getenv("GOOGLE_API_KEY") else None
knowledge_base = {"documents": [], "chunks": [], "vectors": None}


def split_into_chunks(document_name, pages):
    """Split PDF pages into overlapping word chunks while preserving citations."""
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for page_number, text in enumerate(pages, start=1):
        words = (text or "").split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if len(chunk_words) > 5:
                chunks.append({"document": document_name, "page": page_number, "text": " ".join(chunk_words)})
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


def build_knowledge_base(file_paths):
    """Read and embed two or more text-based PDFs."""
    if not file_paths or len(file_paths) < 2:
        raise ValueError("Upload at least two official PDF documents before indexing.")
    all_chunks, document_names = [], []
    for file_path in file_paths:
        path = Path(file_path)
        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        chunks = split_into_chunks(path.name, pages)
        if not chunks:
            raise ValueError(f"No selectable text was found in {path.name}. Scanned PDFs need OCR.")
        document_names.append(path.name)
        all_chunks.extend(chunks)
    vectors = embedder.encode([chunk["text"] for chunk in all_chunks], normalize_embeddings=True)
    knowledge_base.update(documents=document_names, chunks=all_chunks, vectors=vectors)
    return (
        f"### Documents ready\nIndexed **{len(document_names)} PDFs** and **{len(all_chunks)} passages**.\n\n"
        + "\n".join(f"- {name}" for name in document_names)
    )


def retrieve(question):
    if knowledge_base["vectors"] is None:
        raise gr.Error("Upload and index at least two PDFs first.")
    question_vector = embedder.encode([question], normalize_embeddings=True)[0]
    similarities = knowledge_base["vectors"] @ question_vector
    best = np.argsort(similarities)[::-1][:TOP_K]
    return [{**knowledge_base["chunks"][i], "score": float(similarities[i])} for i in best]


def answer(question):
    found = retrieve(question)
    if not found or found[0]["score"] < MIN_SIMILARITY:
        return REFUSAL, found
    context = "\n\n".join(
        f"[Document: {item['document']} | Page: {item['page']}] {item['text']}" for item in found
    )
    if client is None:
        reply = ("Gemma answers are disabled because GOOGLE_API_KEY is not set. "
                 "The highest-ranked passage is shown below for retrieval-only mode.")
    else:
        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=ANSWER_RULES, temperature=0.2),
        )
        reply = (response.text or "").strip() or REFUSAL
    return reply, found


def on_upload(file_paths):
    try:
        if not file_paths or len(file_paths) < 2:
            return "Upload at least two official PDFs, then select **Index documents**."
        return build_knowledge_base(file_paths)
    except Exception as exc:
        return f"### Could not index the documents\n{exc}"


def on_ask(question):
    if not question or not question.strip():
        return "Enter a question to get started.", "No passages retrieved."
    if knowledge_base["vectors"] is None:
        return "Upload and index your PDFs first.", "No passages retrieved."
    try:
        reply, found = answer(question.strip())
        evidence = "\n\n---\n\n".join(
            f"### {item['document']} · page {item['page']} · similarity {item['score']:.2f}\n\n> {item['text']}"
            for item in found
        ) or "No passages were retrieved."
        return reply, evidence
    except Exception as exc:
        return f"The request failed: {exc}", ""


EVALUATION_QUESTIONS = [
    "What is the annual tuition fee?",
    "How many tuition installments are available?",
    "When is the first tuition installment due?",
    "When is the second tuition installment due?",
    "What late fee applies to overdue tuition?",
    "What scholarships and eligibility conditions are listed?",
    "What key dates appear in the academic calendar?",
    "What is the application fee?",
    "What attendance requirement applies to exam eligibility?",
    "What should a student do after missing an examination?",
    "What grading scale is specified?",
    "How can a student request re-evaluation, and by when?",
    "How should a student raise a grievance?",
    "What examination conduct rules are stated?",
    "Who won the 2022 FIFA World Cup?",
]


def initial_evaluation_rows():
    return [[question, "", "", "Pending", ""] for question in EVALUATION_QUESTIONS]


def generate_evaluation_answers(rows):
    """Generate answers and citations, leaving correctness for human review."""
    if knowledge_base["vectors"] is None:
        return rows, "Index at least two PDFs before generating evaluation answers."
    updated = []
    for row in rows or []:
        values = list(row or []) + [""] * (5 - len(row or []))
        question, expected, app_answer, verdict, notes = [
            str(value or "") for value in values[:5]
        ]
        if question.strip() and not app_answer.strip():
            reply, found = answer(question.strip())
            evidence = "\n".join(
                f"Retrieved: {item['document']}, p. {item['page']} "
                f"(similarity {item['score']:.2f}) — {item['text']}"
                for item in found
            )
            app_answer = reply + (f"\n\n{evidence}" if evidence else "")
        updated.append([question, expected, app_answer, verdict or "Pending", notes])
    return updated, (
        "Generated answers where possible. Add expected answers from the PDFs, "
        "inspect each answer and its retrieved evidence, then mark Correct? Yes or No. "
        "Record a failure note for every No."
    )


def score_evaluation(rows):
    """Report accuracy only after 15 answers have been reviewed against gold answers."""
    rows = rows or []
    pending = []
    reviewed = []
    for index, row in enumerate(rows[:15], start=1):
        values = list(row or []) + [""] * (5 - len(row or []))
        question, expected, actual, verdict, notes = [
            str(value or "").strip() for value in values[:5]
        ]
        verdict_key = verdict.casefold()
        if (
            not question
            or not expected
            or not actual
            or verdict_key not in {"yes", "no"}
            or (verdict_key == "no" and not notes)
        ):
            pending.append(index)
        else:
            reviewed.append((question, verdict_key == "yes", notes))

    if len(rows) != 15 or pending:
        return (
            f"### Evaluation pending\nReviewed **{len(reviewed)}/15** questions. "
            "Complete expected answers, app answers, human correctness decisions, "
            "and failure notes for every incorrect answer before reporting accuracy."
        )

    correct = sum(is_correct for _, is_correct, _ in reviewed)
    failures = [
        f"- {question}: {note}"
        for question, is_correct, note in reviewed
        if not is_correct
    ]
    return (
        f"### Reviewed evaluation\n**{correct}/15 correct "
        f"({correct / 15 * 100:.1f}%).**\n\nFailure cases:\n"
        + ("\n".join(failures) if failures else "No failures recorded.")
    )

with gr.Blocks(title="SIT Student Policy Advisor",
               theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange", neutral_hue="slate")) as demo:
    gr.HTML("<div class='app-header'><h1>🎓 SIT Student Policy Advisor</h1><p>Ask questions across official student policy PDFs. Answers cite their source; retrieved passages stay visible for verification.</p></div>")
    with gr.Tabs():
        with gr.Tab("Ask the policy assistant"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=320):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("## 1 · Add official documents")
                        pdf_input = gr.File(label="Select at least two text-based PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
                        index_btn = gr.Button("Index documents", variant="primary")
                        status_md = gr.Markdown("Upload at least two PDFs, then select **Index documents**.")
                        index_btn.click(on_upload, inputs=pdf_input, outputs=status_md)
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("## 2 · Ask a question")
                        question_box = gr.Textbox(label="Your question", placeholder="For example: What eligibility requirement does the policy state?", lines=3)
                        ask_btn = gr.Button("Get cited answer", variant="primary")
                        gr.Examples(examples=[["What eligibility requirement does the document state?"], ["What deadline or date does the policy give?"], ["Who should a student contact about this rule?"], ["Who won the 2022 FIFA World Cup?"]], inputs=question_box, label="Try a policy question or an unsupported question")
                with gr.Column(scale=1.35, min_width=380):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("## Answer")
                        answer_md = gr.Markdown("Your cited answer will appear here.")
                    with gr.Accordion("Retrieved evidence · passages used for this answer", open=True):
                        sources_md = gr.Markdown("Retrieved excerpts will appear here with filename, page, and similarity.")
                    ask_btn.click(on_ask, inputs=question_box, outputs=[answer_md, sources_md])
                    question_box.submit(on_ask, inputs=question_box, outputs=[answer_md, sources_md])
        with gr.Tab("15-question evaluation"):
            gr.Markdown("""## Evaluation worksheet
Create a document-specific set of 15 questions. Enter the expected answer and the app's answer, then mark each result. Include an unsupported question to review refusal behavior. The app does not invent a score.""")
            eval_table = gr.Dataframe(value=initial_evaluation_rows(), headers=["Question", "Expected answer (from PDF)", "App answer and retrieved evidence", "Correct? (Yes/No)", "Failure notes (required for No)"], datatype=["str", "str", "str", "str", "str"], interactive=True, wrap=True)
            with gr.Row():
                generate_eval_button = gr.Button("Generate app answers", variant="primary")
                score_eval_button = gr.Button("Calculate reviewed accuracy")
            eval_result = gr.Markdown("Evaluation pending: fill expected answers from the PDFs, generate responses, and review all 15 rows.")
            generate_eval_button.click(generate_evaluation_answers, inputs=eval_table, outputs=[eval_table, eval_result])
            score_eval_button.click(score_evaluation, inputs=eval_table, outputs=eval_result)
        with gr.Tab("How it works"):
            gr.Markdown("""## Pipeline and design choices
1. Read selectable text from each uploaded PDF and preserve its filename and page number.
2. Split pages into overlapping chunks of **120 words** with **30 words** of overlap.
3. Embed chunks locally with **all-MiniLM-L6-v2**.
4. Retrieve the **top 3** passages by cosine similarity.
5. Refuse when the best similarity is below **0.28**; otherwise ask Gemma to answer from the retrieved text and cite each factual claim.

**Limitations:** scanned PDFs need OCR. The threshold is a starting point and should be reviewed against the team's real 15-question evaluation set. Without an API key, the app displays passages but does not generate an answer.""")

if __name__ == "__main__":
    demo.launch(css=CSS)
