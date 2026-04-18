import os
import sys
import hashlib
import tiktoken
import chromadb
from tqdm import tqdm
from pypdf import PdfReader
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DB_PATH, COLLECTION_NAME
)

enc = tiktoken.encoding_for_model("text-embedding-ada-002")


def load_pdf(filepath: str) -> list[dict]:
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": i + 1,
                "source": os.path.basename(filepath)
            })
    return pages


def load_txt(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"text": text.strip(), "page": 1, "source": os.path.basename(filepath)}]


def chunk_text(text: str, source: str, page: int) -> list[dict]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunk_id = hashlib.md5(f"{source}-{page}-{start}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text_str,
            "source": source,
            "page": page,
            "token_count": len(chunk_tokens)
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_file(filepath: str):
    print(f"\nIngesting: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        pages = load_pdf(filepath)
    elif ext == ".txt":
        pages = load_txt(filepath)
    else:
        print(f"Unsupported file type: {ext}")
        return

    print(f"  Loaded {len(pages)} page(s)")

    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data["text"], page_data["source"], page_data["page"])
        all_chunks.extend(chunks)
    print(f"  Created {len(all_chunks)} chunks")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["id"] for c in all_chunks]
    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "token_count": c["token_count"]} 
        for c in all_chunks
    ]

    print(f"  Embedding and storing {len(all_chunks)} chunks...")
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    print(f"  Stored {len(all_chunks)} chunks in ChromaDB")
    print(f"  Collection size: {collection.count()} total chunks")


def ingest_directory(directory: str):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".pdf", ".txt"))
    ]
    if not files:
        print(f"No PDF or TXT files found in {directory}")
        return
    for filepath in files:
        ingest_file(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_directory>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        ingest_directory(target)
    elif os.path.isfile(target):
        ingest_file(target)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)