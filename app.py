"""Document question answering with retrieval augmented generation."""

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
ANSWER_RULES = """Answer questions about the document using ONLY the context passages provided.
Each passage begins with its page number. Cite every fact with a page, e.g. (p. 2).
If the answer is absent, reply exactly: The answer could not be found in the document.
Keep answers concise, in 2 to 4 sentences."""
SAMPLE_PAGES = [
    """CS 101: Introduction to Programming (Sample Syllabus). Instructor: Dr. A. Rao. Email: arao@example.edu.
Lectures are held Monday and Wednesday, 10:00 to 11:30 AM, in Room B-204. Labs are on Friday from 2:00 to 4:00 PM
in the Computer Lab (Room C-110). Office hours: Tuesday 3:00 to 5:00 PM, or by appointment. This course introduces
programming using Python. No prior coding experience is required.""",
    """Grading. Weekly lab exercises: 20%. Two mini-projects: 30% (15% each). Midterm exam: 20%. Final exam: 25%.
Class participation: 5%. A score of 90% or above earns an A, 80 to 89% a B, 70 to 79% a C, and 60 to 69% a D.
Late submissions lose 10% per day, up to a maximum of 3 days, after which they receive zero. Each student has
two grace days per semester that may be used to submit any lab late without penalty.""",
    """Schedule. Weeks 1-2: variables, data types, and input/output. Weeks 3-4: conditions and loops. Week 5: functions.
Week 6: lists and dictionaries. Week 7: Midterm exam (in class, closed book). Weeks 8-9: files and error handling.
Weeks 10-11: an introduction to APIs and working with data. Weeks 12-13: mini-project 2, a team project of up to
3 students. Week 14: review. The final exam is a 2-hour written exam during the examination period.""",
    """Policies. Attendance: students must attend at least 75% of lectures to be eligible for the final exam.
Academic integrity: students may discuss ideas with classmates, but all submitted code must be their own. AI
tools may be used for learning and debugging, but their use must be stated in a comment at the top of the file.
Laptops: bring a laptop to every lab. Students without a laptop may reserve a machine in the Computer Lab.
Textbook: Automate the Boring Stuff with Python (available free online).""",
]

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY")) if os.getenv("GOOGLE_API_KEY") else None
knowledge_base = {"name": None, "chunks": [], "vectors": None}


def read_pdf(path):
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def split_into_chunks(pages):
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for page_number, text in enumerate(pages, start=1):
        words = text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_SIZE]
            if len(chunk_words) > 5:
                chunks.append({"page": page_number, "text": " ".join(chunk_words)})
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


def build_knowledge_base(pages, name):
    chunks = split_into_chunks(pages)
    if not chunks:
        raise ValueError("No text was found. The PDF may contain scanned images only.")
    vectors = embedder.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True)
    knowledge_base.update(name=name, chunks=chunks, vectors=vectors)
    return f"Loaded **{name}**: {len(pages)} pages, {len(chunks)} chunks."


def retrieve(question):
    if knowledge_base["vectors"] is None:
        raise gr.Error("Upload a PDF or choose the sample document first.")
    question_vector = embedder.encode([question], normalize_embeddings=True)[0]
    similarities = knowledge_base["vectors"] @ question_vector
    best = np.argsort(similarities)[::-1][:TOP_K]
    return [{**knowledge_base["chunks"][i], "score": float(similarities[i])} for i in best]


def answer(question):
    found = retrieve(question)
    context = "\n\n".join(f"[Page {chunk['page']}] {chunk['text']}" for chunk in found)
    if client is None:
        reply = ("**Demo mode:** set GOOGLE_API_KEY to enable Gemma answers. Retrieval found this passage "
                 f"on page **{found[0]['page']}**:\n\n> {found[0]['text'][:400]}...")
    else:
        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=ANSWER_RULES, temperature=0.2),
        )
        reply = (response.text or "").strip()
    sources = "\n\n".join(
        f"**Page {chunk['page']}** (similarity {chunk['score']:.2f})\n> {chunk['text']}" for chunk in found
    )
    return reply, sources


def on_upload(file):
    if file is None:
        return "No file uploaded."
    try:
        return build_knowledge_base(read_pdf(file), Path(file).name)
    except Exception as exc:
        raise gr.Error(f"The PDF could not be read: {exc}") from exc


def on_ask(question):
    if not question or not question.strip():
        raise gr.Error("Enter a question first.")
    return answer(question.strip())


with gr.Blocks(title="Document Question Answering") as demo:
    gr.Markdown("# Document Question Answering\nUpload a PDF and ask questions. Answers use retrieved passages and include page references.")
    with gr.Row():
        pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
        with gr.Column():
            status_md = gr.Markdown("Choose a PDF or load the sample document.")
            sample_btn = gr.Button("Use the sample document")
    question_box = gr.Textbox(label="Question", placeholder="For example: How is the course graded?")
    ask_btn = gr.Button("Ask", variant="primary")
    gr.Examples(["How is the course graded?", "When are office hours?", "Can I use AI tools?"], inputs=question_box)
    answer_md = gr.Markdown()
    with gr.Accordion("Retrieved passages", open=False):
        sources_md = gr.Markdown()
    pdf_input.upload(on_upload, inputs=pdf_input, outputs=status_md)
    sample_btn.click(lambda: build_knowledge_base(SAMPLE_PAGES, "Sample CS 101 syllabus"), outputs=status_md)
    ask_btn.click(on_ask, inputs=question_box, outputs=[answer_md, sources_md])
    question_box.submit(on_ask, inputs=question_box, outputs=[answer_md, sources_md])


if __name__ == "__main__":
    demo.launch()
