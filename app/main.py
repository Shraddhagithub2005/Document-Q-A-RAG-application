"""
Document Q&A API — a RAG (Retrieval-Augmented Generation) backend service.

Endpoints:
  GET  /health          -> liveness check
  POST /upload           -> upload a PDF, it gets chunked, embedded, and stored
  POST /ask               -> ask a question, get an answer grounded in your documents

Run locally:
  uvicorn app.main:app --reload

Auth:
  If APP_API_KEY is set in the environment, all endpoints except /health require
  the header  X-API-Key: <your key>
"""
import uuid

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from app.config import settings
from app.rag import ChromaStore, chunk_text, generate_answer, parse_pdf

app = FastAPI(
    title="Document Q&A API",
    description="Upload PDFs and ask questions answered with cited sources, powered by RAG + Claude.",
    version="1.0.0",
)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

store = ChromaStore()


# ---------- Auth ----------

def require_api_key(x_api_key: str = Header(default=None)):
    if settings.APP_API_KEY and x_api_key != settings.APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return True


# ---------- Schemas ----------

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages_parsed: int
    chunks_stored: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    doc_id: str | None = Field(default=None, description="Restrict search to this document.")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceRef(BaseModel):
    page: int
    doc_id: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_id: str | None = Form(default=None, description="Optional custom document ID"),
):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pages = parse_pdf(file_bytes)
    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in this PDF.")

    resolved_doc_id = doc_id or f"{uuid.uuid4().hex[:12]}"
    chunks = chunk_text(pages)
    stored = store.add_document(resolved_doc_id, chunks)

    return UploadResponse(
        doc_id=resolved_doc_id,
        filename=file.filename,
        pages_parsed=len(pages),
        chunks_stored=stored,
    )


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask_question(payload: AskRequest):
    if payload.doc_id and not store.document_exists(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"No document found with doc_id '{payload.doc_id}'.")

    hits = store.query(payload.question, doc_id=payload.doc_id, top_k=payload.top_k)

    try:
        result = generate_answer(payload.question, hits)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return AskResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
    )
