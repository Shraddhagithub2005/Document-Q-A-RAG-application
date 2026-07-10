"""
Core RAG (Retrieval-Augmented Generation) logic.

Pipeline:
  1. parse_pdf()   -> extract text per page
  2. chunk_text()  -> split into overlapping chunks (keeps page numbers for citations)
  3. ChromaStore   -> embed + store chunks, and later retrieve the most relevant ones
  4. generate_answer() -> call Claude with the retrieved context and return a grounded answer
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import chromadb
from pypdf import PdfReader
from anthropic import Anthropic, APIError

from app.config import settings


# ---------- PDF parsing ----------

def parse_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text from a PDF, page by page.

    Returns a list of {"page": int, "text": str} skipping blank pages.
    """
    reader = PdfReader.__new__(PdfReader)
    import io
    reader = PdfReader(io.BytesIO(file_bytes))

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# ---------- Chunking ----------

def chunk_text(pages: list[dict], chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Split page text into overlapping chunks, preserving the source page number.

    A simple sliding-window character splitter. Good enough for most documents
    without pulling in a heavier NLP dependency.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page["page"]})
            if end >= len(text):
                break
            start = end - overlap
    return chunks


# ---------- Vector store ----------

class ChromaStore:
    """Thin wrapper around a persistent Chroma collection.

    Uses Chroma's built-in default embedding function (ONNX MiniLM) so no
    extra ML dependency (like torch/sentence-transformers) is required —
    this keeps the Docker image small enough for free-tier hosting.
    """

    def __init__(self, persist_dir: str = None, embedding_function=None):
        """embedding_function is injectable so tests (and alternate deployments)
        can swap in a fast/offline embedder instead of Chroma's default, which
        downloads an ONNX model from the network on first use."""
        self.client = chromadb.PersistentClient(path=persist_dir or settings.CHROMA_DIR)
        self._embedding_function = embedding_function

    def _collection(self, name: str):
        kwargs = {"name": name}
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return self.client.get_or_create_collection(**kwargs)

    def add_document(self, doc_id: str, chunks: list[dict]) -> int:
        """Embed and store chunks under a collection named after doc_id's namespace.

        Every chunk is tagged with doc_id in its metadata so we can filter
        retrieval to a single document at query time.
        """
        collection = self._collection("documents")
        ids = [f"{doc_id}::{uuid.uuid4().hex[:8]}" for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"doc_id": doc_id, "page": c["page"]} for c in chunks]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def query(self, question: str, doc_id: Optional[str] = None, top_k: int = None) -> list[dict]:
        collection = self._collection("documents")
        top_k = top_k or settings.DEFAULT_TOP_K

        where = {"doc_id": doc_id} if doc_id else None
        result = collection.query(query_texts=[question], n_results=top_k, where=where)

        hits = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, distances):
            hits.append({"text": text, "page": meta.get("page"), "doc_id": meta.get("doc_id"), "distance": dist})
        return hits

    def document_exists(self, doc_id: str) -> bool:
        collection = self._collection("documents")
        result = collection.get(where={"doc_id": doc_id}, limit=1)
        return len(result.get("ids", [])) > 0


# ---------- Generation ----------

@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]


SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant. Answer the user's question using "
    "ONLY the provided context excerpts. If the context does not contain enough "
    "information to answer, say so plainly instead of guessing. Keep answers concise."
)


def generate_answer(question: str, context_chunks: list[dict]) -> AnswerResult:
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment to enable answer generation."
        )

    if not context_chunks:
        return AnswerResult(
            answer="No relevant content was found in the document(s) for this question.",
            sources=[],
        )

    context_block = "\n\n".join(
        f"[Source: page {c['page']}]\n{c['text']}" for c in context_chunks
    )

    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                }
            ],
        )
    except APIError as e:
        raise RuntimeError(f"LLM generation failed: {e}") from e

    answer_text = "".join(block.text for block in response.content if block.type == "text")

    sources = [{"page": c["page"], "doc_id": c["doc_id"]} for c in context_chunks]
    return AnswerResult(answer=answer_text, sources=sources)
