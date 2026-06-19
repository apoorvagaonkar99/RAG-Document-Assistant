from pypdf import PdfReader

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print(text)