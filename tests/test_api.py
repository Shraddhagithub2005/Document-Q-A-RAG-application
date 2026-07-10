"""
Tests for the Document Q&A API.

The LLM call (generate_answer) is monkeypatched so tests run fast, free,
and without needing a real ANTHROPIC_API_KEY.
"""
import io
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Isolate Chroma storage per test run so tests don't pollute each other.
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma_test"))
    monkeypatch.setenv("APP_API_KEY", "")  # auth disabled for most tests

    # Reimport app fresh so it picks up the patched env vars.
    import importlib
    from app import config, rag, main

    importlib.reload(config)
    importlib.reload(rag)
    importlib.reload(main)

    # Swap in a deterministic, offline embedding function so tests don't
    # depend on downloading a model from the network.
    from tests.fake_embeddings import FakeEmbeddingFunction
    main.store = rag.ChromaStore(embedding_function=FakeEmbeddingFunction())

    with TestClient(main.app) as c:
        yield c, main


def _make_pdf_bytes(text: str) -> bytes:
    """Build a minimal single-page PDF containing the given text."""
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_rejects_non_pdf(client):
    c, _ = client
    resp = c.post("/upload", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert resp.status_code == 400


def test_ask_unknown_doc_id_returns_404(client):
    c, _ = client
    resp = c.post("/ask", json={"question": "What is this about?", "doc_id": "does-not-exist"})
    assert resp.status_code == 404


def test_upload_and_ask_flow(client, monkeypatch):
    c, main = client

    # Patch generate_answer so we don't need a real Anthropic API key in CI.
    def fake_generate_answer(question, context_chunks):
        from app.rag import AnswerResult
        assert context_chunks, "expected retrieved chunks to be passed to generation"
        return AnswerResult(
            answer=f"Fake answer for: {question}",
            sources=[{"page": c["page"], "doc_id": c["doc_id"]} for c in context_chunks],
        )

    monkeypatch.setattr(main, "generate_answer", fake_generate_answer)

    pdf_bytes = _make_pdf_bytes("The mitochondria is the powerhouse of the cell.")
    upload_resp = c.post(
        "/upload",
        files={"file": ("bio.pdf", pdf_bytes, "application/pdf")},
        data={"doc_id": "bio-doc"},
    )
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["doc_id"] == "bio-doc"
    assert body["chunks_stored"] >= 1

    ask_resp = c.post("/ask", json={"question": "What is the mitochondria?", "doc_id": "bio-doc"})
    assert ask_resp.status_code == 200
    ask_body = ask_resp.json()
    assert "Fake answer" in ask_body["answer"]
    assert ask_body["sources"][0]["doc_id"] == "bio-doc"


def test_api_key_auth_blocks_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma_test_auth"))
    monkeypatch.setenv("APP_API_KEY", "secret123")

    import importlib
    from app import config, rag, main
    importlib.reload(config)
    importlib.reload(rag)
    importlib.reload(main)

    from tests.fake_embeddings import FakeEmbeddingFunction
    main.store = rag.ChromaStore(embedding_function=FakeEmbeddingFunction())

    with TestClient(main.app) as c:
        resp = c.post("/ask", json={"question": "anything"})
        assert resp.status_code == 401

        resp2 = c.post("/ask", json={"question": "anything"}, headers={"X-API-Key": "secret123"})
        # 404/502 is fine here (no docs / no LLM key) — the point is auth let it through
        assert resp2.status_code != 401
