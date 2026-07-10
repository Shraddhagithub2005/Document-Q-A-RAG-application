# Document Q&A API — RAG Backend Service

A REST API that lets you upload PDF documents and ask natural-language questions
about them, with answers grounded in the source text and cited by page number.

Built with **FastAPI**, **ChromaDB** (vector store), and the **Anthropic Claude API**
for generation. This is a small, self-contained implementation of a Retrieval-Augmented
Generation (RAG) pipeline — the pattern behind most production "chat with your docs" tools.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  POST       │ PDF  │  parse_pdf   │ text │  chunk_text  │
│  /upload    │─────▶│  (pypdf)     │─────▶│ (sliding     │
└─────────────┘      └──────────────┘      │  window)     │
                                            └──────┬───────┘
                                                   │ chunks + page #s
                                                   ▼
                                          ┌──────────────────┐
                                          │  ChromaStore      │
                                          │  (embed + persist)│
                                          └──────────────────┘

┌─────────────┐                          ┌──────────────────┐
│  POST       │  question                │  ChromaStore      │
│  /ask       │─────────────────────────▶│  .query()         │
└─────────────┘                          │  (vector search)  │
                                          └────────┬──────────┘
                                                   │ top-k relevant chunks
                                                   ▼
                                          ┌──────────────────┐
                                          │  generate_answer  │
                                          │  (Claude API)      │
                                          └────────┬──────────┘
                                                   │
                                                   ▼
                                     { answer, sources: [{page, doc_id}] }
```

## Why this design

- **Injectable embedding function** (`ChromaStore(embedding_function=...)`) — production
  uses Chroma's built-in ONNX embedder (no GPU/torch dependency, keeps the Docker image
  small), while tests inject a deterministic offline stand-in so the test suite runs in
  under 3 seconds with no network calls and no API key.
- **Page-level citations** — every chunk keeps its source page number through the whole
  pipeline, so answers can point back to exactly where the information came from.
- **Optional API key auth** — set `APP_API_KEY` to require an `X-API-Key` header;
  leave it blank for local development.
- **Stateless generation, persistent retrieval** — the vector index survives restarts
  (`CHROMA_DIR`), so you don't re-embed documents every time the service redeploys.

## Project structure

```
app/
  main.py     — FastAPI routes (/health, /upload, /ask)
  rag.py      — PDF parsing, chunking, vector store, LLM generation
  config.py   — environment-based settings
tests/
  test_api.py — endpoint tests, including a full upload → ask flow
Dockerfile
requirements.txt
.env.example
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

### Try it with curl

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf" \
  -F "doc_id=my-doc"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "doc_id": "my-doc"}'
```

## Running tests

```bash
pip install -r requirements.txt   # includes test deps
pytest tests/ -v
```

5 tests cover: health check, file-type validation, 404 on unknown document,
the full upload → ask round trip, and API key auth enforcement.

## Deploying (free tier)

### Option A — Render
1. Push this repo to GitHub.
2. On [render.com](https://render.com), create a **New Web Service**, connect the repo,
   choose "Docker" as the environment.
3. Add environment variable `ANTHROPIC_API_KEY` (and optionally `APP_API_KEY`) in the
   Render dashboard.
4. Deploy — Render builds the Dockerfile automatically on every push.

### Option B — Railway
1. Push to GitHub, then "New Project" → "Deploy from GitHub repo" on
   [railway.app](https://railway.app).
2. Railway auto-detects the Dockerfile.
3. Add the same environment variables under the Variables tab.

### Option C — plain Docker anywhere
```bash
docker build -t doc-qa-api .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  doc-qa-api
```

## Possible extensions

- Swap Chroma for a hosted vector DB (Pinecone, Qdrant Cloud) for multi-instance deploys.
- Add streaming responses (`/ask/stream`) using Claude's streaming API.
- Support multi-file uploads and cross-document search.
- Add a `/documents` endpoint to list/delete ingested docs.

## License

MIT
