import sys
import chromadb
from google import genai
from google.genai import types
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL,
    TOP_K, CHROMA_DB_PATH, COLLECTION_NAME, SYSTEM_PROMPT
)

client = genai.Client(api_key=GEMINI_API_KEY,
                      http_options=types.HttpOptions(api_version="v1"))


def embed_query(query: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def retrieve(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    )
    return response.text


def answer_query(query: str, verbose: bool = False) -> str: