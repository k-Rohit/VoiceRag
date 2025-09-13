import os
import fitz  # PyMuPDF for PDF parsing
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY ")
# Connect to Qdrant
qdrant = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)

collection_name = "ncert_books"

embedding_dim = 1536  

# Create/recreate collection
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append((page_num, text))
    return pages

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def ingest_book(pdf_path, subject, book_title):
    pages = extract_text_from_pdf(pdf_path)

    for page_num, text in pages:
        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = embed_text(chunk)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "subject": subject,
                    "book_title": book_title,
                    "page": page_num,
                    "text": chunk
                }
            )
            qdrant.upsert(collection_name=collection_name, points=[point])

    print(f"✅ Ingested {book_title} ({subject}) into Qdrant")

ingest_book("ncert_science_class10.pdf", subject="Science", book_title="NCERT Science Class 10")
