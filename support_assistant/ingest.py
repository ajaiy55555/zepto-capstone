"""Ingestion, embedding and retrieval for the Zepto policy corpus."""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"                       # folder holding doc_01.txt ... doc_08.txt
CHROMA_DIR = "chroma_db"                # folder where ChromaDB saves its data
COLLECTION_NAME = "zepto_policies"      # name of the vector collection

# Local open-source embedding model: no API key, no account needed.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # use cosine similarity for matching
)


def ingest_documents():
    """Read every doc file, embed it, and store it in ChromaDB (safe to re-run)."""
    ids, texts, metadatas = [], [], []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        ids.append(os.path.splitext(os.path.basename(path))[0])   # "doc_01", "doc_02", ...
        texts.append(text)                                         # one chunk per document
        metadatas.append({"source_file": os.path.basename(path)})

    embeddings = embedder.encode(texts).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def retrieve(query, k=3):
    """Embed the query and return the k most similar chunks."""
    query_vector = embedder.encode([query]).tolist()
    result = collection.query(query_embeddings=query_vector, n_results=k)

    chunks = []
    for chunk_id, text, distance in zip(result["ids"][0], result["documents"][0], result["distances"][0]):
        chunks.append({
            "id": chunk_id,
            "text": text,
            "similarity": round(1 - distance, 4),
        })
    return chunks                        # best match first


if __name__ == "__main__":
    print(f"Ingested {ingest_documents()} documents")