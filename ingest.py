import os
import uuid
import json

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------- CONFIG ----------

BASE_DIR = r"C:\Users\MattVeydt\OneDrive - AMEND Consulting\Desktop\kb_assistant"
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.json")

load_dotenv()  # not strictly needed here but harmless

# Local embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------- HELPERS ----------

def load_text_from_file(path: str) -> str:
    """
    Load text from .md, .txt, or .pdf files.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in [".md", ".txt"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return ""  # unsupported types


def embed_texts(texts):
    """
    Get embeddings for a list of texts using local SentenceTransformer.
    Returns list of Python lists (for JSON serialization).
    """
    if not texts:
        return []
    embeddings = embedding_model.encode(texts, batch_size=16, show_progress_bar=False)
    # convert to plain lists for JSON
    return [emb.tolist() for emb in embeddings]


def main():
    print("Looking for documents in:", DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = []
    metadatas = []

    # Walk data directory
    for root, dirs, files in os.walk(DATA_DIR):
        print("Scanning directory:", root)
        print("Files:", files)
        for fname in files:
            path = os.path.join(root, fname)
            _, ext = os.path.splitext(fname)
            if ext.lower() not in [".md", ".txt", ".pdf"]:
                continue

            print(f"Loading {path}")
            text = load_text_from_file(path)
            if not text.strip():
                continue

            file_chunks = splitter.split_text(text)
            print(f" - {len(file_chunks)} chunks")

            for i, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "id": str(uuid.uuid4()),
                    "source": fname,
                    "path": path,
                    "chunk_index": i,
                })

    if not chunks:
        print("No documents found to ingest.")
        return

    print(f"Embedding {len(chunks)} chunks locally with {EMBEDDING_MODEL_NAME}...")
    embeddings = embed_texts(chunks)

    index = []
    for chunk, meta, emb in zip(chunks, metadatas, embeddings):
        meta_copy = dict(meta)
        meta_copy["text"] = chunk
        meta_copy["embedding"] = emb
        index.append(meta_copy)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"Ingestion complete. Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()