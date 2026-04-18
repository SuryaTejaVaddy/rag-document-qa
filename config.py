import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I cannot find this in the
provided documents."
Always cite the source document and page number when available."""