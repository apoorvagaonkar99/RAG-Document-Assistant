from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# User Question
question = input("Ask a question: ")

# Prompt
prompt = f"""
Answer only using the information below.

Document:
{text}

Question:
{question}
"""

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(prompt)

print("\nAnswer:")
print(response.text)