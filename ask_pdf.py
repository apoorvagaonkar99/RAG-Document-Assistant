from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

chunks = [text[i:i+500] for i in range(0, len(text), 500)]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("resume.index")

question = input("Ask a question: ")

query_embedding = embedding_model.encode([question])

D, I = index.search(
    np.array(query_embedding).astype("float32"),
    k=2
)

context = ""

for idx in I[0]:
    context += chunks[idx] + "\n"

prompt = f"""
Answer only using the given context.

Context:
{context}

Question:
{question}
"""

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(prompt)

print("\nAnswer:")
print(response.text)