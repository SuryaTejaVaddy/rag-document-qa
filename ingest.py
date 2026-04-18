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