"""RAG Document Q&A SystemA Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your PDF and text documents using Google Gemini AI.How It WorksIngest — PDFs/text files are chunked and stored in ChromaDB (local vector database)Retrieve — Your question is matched against stored chunks using semantic searchGenerate — Gemini AI reads the matched chunks and answers your questionPrerequisitesPython 3.11+A Gemini API key from aistudio.google.com (free tier)GitHub Codespaces (recommended) or local Python environmentSetup1. Clone the RepositoryBashgit clone https://github.com/SuryaTejaVaddy/rag-document-qa.git
cd rag-document-qa
2. Install DependenciesBashpip install -r requirements.txt
3. Set Your API KeyOption A — GitHub Codespaces (recommended):Go to your GitHub repository → Settings → Secrets and variables → Codespaces → New secret:Name: GEMINI_API_KEYValue: your API keyThen restart your Codespace. If the key doesn't load automatically, run:Bashexport GEMINI_API_KEY="your-api-key-here"
Option B — Local .env file:Bashecho 'GEMINI_API_KEY=your-api-key-here' > .env
Running the PipelineStep 1 — Add your documentsCopy PDF or .txt files into the data/ folder.Step 2 — Ingest documentsBashpython ingest.py data/
This chunks your documents and stores them in ChromaDB. You'll see output like:Ingesting: data/yourfile.pdfLoaded 4 page(s)Created 8 chunksStored 8 chunks in ChromaDBCollection size: 8 total chunksStep 3 — Ask a questionBashpython query.py "What is this document about?"
You'll see the top 5 retrieved chunks, then the generated answer:--- Retrieved 5 chunks ---[1] yourfile.pdf p.1 | distance=0.69...Answer: This document is about...Step 4 — Run evaluationBashpython evaluate.py
Runs predefined test questions and saves results to eval_results.json.ConfigurationEdit config.py to change defaults:SettingDefaultDescriptionCHAT_MODELgemini-2.0-flash-liteGemini model for generationCHUNK_SIZE500Tokens per chunkCHUNK_OVERLAP50Overlap between chunksTOP_K5Number of chunks to retrieveCHROMA_DB_PATH./chroma_dbWhere ChromaDB stores dataSupported File TypesPDF (.pdf)Plain text (.txt)TroubleshootingValueError: No API key was providedRun: export GEMINI_API_KEY="your-key-here"429 RESOURCE_EXHAUSTED with limit: 0Your API key's project doesn't have free tier access. Create a new key at aistudio.google.com using "Create API key in new project".ChromaDB telemetry warningsMessages like Failed to send telemetry event are harmless and can be ignored.ONNX warnings on startupThe GetPciBusId warning from ONNX Runtime is harmless — ChromaDB uses it for local embeddings.Project StructurePlaintextrag-document-qa/
├── config.py          # Configuration settings
├── ingest.py          # Document loading, chunking, and storage
├── query.py           # Retrieval and answer generation
├── evaluate.py        # Test suite with keyword-match scoring
├── requirements.txt   # Python dependencies
├── data/              # Place your documents here
├── chroma_db/         # Auto-created vector database
└── .devcontainer/     # GitHub Codespaces configuration
"""
