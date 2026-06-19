from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text into chunks
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "resume.index")

print("FAISS index created successfully!")