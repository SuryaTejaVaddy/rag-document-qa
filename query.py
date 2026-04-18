import sys
import chromadb
import google.generativeai as genai
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL,
    TOP_K, CHROMA_DB_PATH, COLLECTION_NAME, SYSTEM_PROMPT
)

genai.configure(api_key=GEMINI_API_KEY)


def embed_query(query: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]


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
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text


def answer_query(query: str, verbose: bool = False) -> str:
    query_embedding = embed_query(query)
    chunks = retrieve(query_embedding)

    if verbose:
        print(f"\n--- Retrieved {len(chunks)} chunks ---")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['source']} p.{c['page']} | distance={c['distance']:.4f}")
            print(f"      {c['text'][:120]}...")

    context = build_context(chunks)
    answer = generate_answer(query, context)
    return answer


def interactive_mode():
    print("RAG Q&A System — type 'exit' to quit, 'verbose' to toggle debug output\n")
    verbose = False
    while True:
        try:
            query = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'on' if verbose else 'off'}")
            continue
        answer = answer_query(query, verbose=verbose)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(answer_query(query, verbose=True))
    else:
        interactive_mode()