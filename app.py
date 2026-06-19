from flask import Flask, render_template, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import faiss
import numpy as np
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chat_history = []

filename = ""
page_count = 0
word_count = 0
chunk_count = 0

document_text = ""
chunks = []
index = None


def clean_response(text):
    return (
        text.replace("**", "")
            .replace("```html", "")
            .replace("```python", "")
            .replace("```css", "")
            .replace("```javascript", "")
            .replace("```", "")
            .strip()
    )


def safe_gemini_call(prompt):

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    try:
        response = model.generate_content(prompt)
        return clean_response(response.text)

    except Exception:
        return "⚠️ Gemini API quota exceeded or temporarily unavailable. Please try again later."


def process_pdf(pdf_path):

    global document_text
    global chunks
    global index
    global page_count
    global word_count
    global chunk_count

    reader = PdfReader(pdf_path)

    document_text = ""

    page_count = len(reader.pages)

    for page in reader.pages:

        text = page.extract_text()

        if text:
            document_text += text

    word_count = len(document_text.split())

    chunks = [
        document_text[i:i + 1000]
        for i in range(0, len(document_text), 1000)
    ]

    chunk_count = len(chunks)

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )


def ask_document(question):

    global index

    if index is None:
        return "Please upload a PDF first."

    query_embedding = embedding_model.encode([question])

    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        k=min(3, len(chunks))
    )

    context = ""

    for idx in I[0]:
        context += chunks[idx] + "\n\n"

    return context[:2000]

def generate_summary():

    if not document_text:
        return "Please upload a PDF first."

    sentences = document_text.split(".")

    summary = ".".join(sentences[:10])

    return summary.strip()

def extract_skills():

    if not document_text:
        return "Please upload a PDF first."

    skills = [
        "Python", "Java", "SQL", "HTML", "CSS",
        "JavaScript", "React", "Node.js",
        "Express", "MongoDB", "MySQL",
        "Git", "GitHub", "Docker",
        "Flask", "Machine Learning",
        "Artificial Intelligence"
    ]

    found = []

    for skill in skills:
        if skill.lower() in document_text.lower():
            found.append(skill)

    return "\n".join(found)

def ats_score():

    if not document_text:
        return 0

    score = 60

    keywords = [
        "python", "java", "sql", "html", "css",
        "javascript", "react", "node", "express",
        "mongodb", "mysql", "git", "github",
        "machine learning", "artificial intelligence",
        "cloud", "docker", "flask", "api"
    ]

    text_lower = document_text.lower()

    for keyword in keywords:
        if keyword in text_lower:
            score += 2

    return min(score, 100)

def document_analysis():

    if not document_text:
        return "Please upload a PDF first."

    analysis = f"""
Document Analysis

Pages: {page_count}
Words: {word_count}
Chunks: {chunk_count}

Document Type:
PDF Document

Key Statistics:
- Total Pages: {page_count}
- Total Words: {word_count}
- Semantic Chunks: {chunk_count}

This analysis was generated locally.
"""

    return analysis


@app.route("/", methods=["GET", "POST"])
def home():

    global filename

    answer = ""
    summary = ""
    skills = ""
    ats = ""
    review = ""
    question = ""

    if request.method == "POST":

        pdf = request.files.get("pdf")

        if pdf and pdf.filename:

            filename = pdf.filename

            pdf_path = os.path.join(
                UPLOAD_FOLDER,
                pdf.filename
            )

            pdf.save(pdf_path)

            process_pdf(pdf_path)

        action = request.form.get("action")

        question = request.form.get(
            "question",
            ""
        )

        if action == "ask" and question:

            answer = ask_document(question)

            chat_history.append(("You", question))
            chat_history.append(("AI", answer))

        elif action == "summary":

            summary = generate_summary()

        elif action == "skills":

            skills = extract_skills()

        elif action == "ats":

            ats = ats_score()

        elif action == "review":

            review = document_analysis()

        elif action == "clear_chat":

            chat_history.clear()

    return render_template(
        "index.html",
        answer=answer,
        summary=summary,
        skills=skills,
        ats=ats,
        review=review,
        question=question,
        chat_history=chat_history,
        filename=filename,
        page_count=page_count,
        word_count=word_count,
        chunk_count=chunk_count
    )


if __name__ == "__main__":
    app.run()
