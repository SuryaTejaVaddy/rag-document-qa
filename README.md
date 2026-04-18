# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your PDF and text documents using Google Gemini AI.

## How It Works

1. **Ingest** — PDFs/text files are chunked and stored in ChromaDB (local vector database)
2. **Retrieve** — Your question is matched against stored chunks using semantic search
3. **Generate** — Gemini AI reads the matched chunks and answers your question

## Prerequisites

- Python 3.11+
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com) (free tier)
- GitHub Codespaces (recommended) or local Python environment

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SuryaTejaVaddy/rag-document-qa.git
cd rag-document-qa
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your API Key

**Option A — GitHub Codespaces (recommended):**

Go to your GitHub repository → Settings → Secrets and variables → Codespaces → New secret:
- **Name:** `GEMINI_API_KEY`
- **Value:** your API key

Then restart your Codespace. If the key doesn't load automatically, run:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Option B — Local `.env` file:**

```bash
echo 'GEMINI_API_KEY=your-api-key-here' > .env
```

## Running the Pipeline

### Step 1 — Add your documents

Copy PDF or `.txt` files into the `data/` folder.

### Step 2 — Ingest documents

```bash
python ingest.py
```

You'll see output like:
Ingesting: data/yourfile.pdf
Loaded 4 page(s)
Created 8 chunks
Stored 8 chunks in ChromaDB
Collection size: 8 total chunks

### Step 3 — Ask a question

```bash
python query.py "What is this document about?"
```

You'll see the top 5 retrieved chunks, then the generated answer:
--- Retrieved 5 chunks ---
[1] yourfile.pdf p.1 | distance=0.69..
Answer: This document is about...

### Step 4 — Run evaluation

```bash
python evaluate.py
```

Runs predefined test questions and saves results to `eval_results.json`.

## Configuration

Edit `config.py` to change defaults:

| Setting | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `gemini-2.0-flash-lite` | Gemini model for generation |
| `CHUNK_SIZE` | `500` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHROMA_DB_PATH` | `./chroma_db` | Where ChromaDB stores data |

## Supported File Types

- PDF (`.pdf`)
- Plain text (`.txt`)

## Troubleshooting

**`ValueError: No API key was provided`**
```bash
export GEMINI_API_KEY="your-key-here"
```

**`429 RESOURCE_EXHAUSTED with limit: 0`**
Your API key's project doesn't have free tier access. Create a new key at [aistudio.google.com](https://aistudio.google.com) using "Create API key in new project".

**ChromaDB telemetry warnings**
Messages like `Failed to send telemetry event` are harmless and can be ignored.

**ONNX warnings on startup**
The `GetPciBusId` warning from ONNX Runtime is harmless — ChromaDB uses it for local embeddings.

## Project Structure
rag-document-qa/
├── config.py          # Configuration settings
├── ingest.py          # Document loading, chunking, and storage
├── query.py           # Retrieval and answer generation
├── evaluate.py        # Test suite with keyword-match scoring
├── requirements.txt   # Python dependencies
├── data/              # Place your documents here
├── chroma_db/         # Auto-created vector database
└── .devcontainer/     # GitHub Codespaces configuration
