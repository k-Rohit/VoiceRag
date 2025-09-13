from openai import OpenAI
import os
from qdrant_client.models import CollectionStatus, VectorParams, Distance
from constants import collection_name, embedding_dim
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

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

def embed_text(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def init_collection(qdrant, collection_name=collection_name, embedding_dim=embedding_dim):
    try:
        info = qdrant.get_collection(collection_name)
        if info.status == CollectionStatus.GREEN:
            print(f"✅ Collection '{collection_name}' already exists")
            return
    except:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"📚 Created new collection '{collection_name}'")

def ingest_chapter(text):
     chunks = chunk_text(text)

     for chunk in chunks:
          embedding = embed_text(chunk)
          point = PointStruct(
               id=str(uuid.uuid4()),
               vector=embedding,
               payload={
               "text": chunk
               }
          )
          qdrant.upsert(collection_name=collection_name, points=[point])

     print(f"✅ Ingested into Qdrant")