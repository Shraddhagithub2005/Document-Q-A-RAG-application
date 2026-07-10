FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal on purpose to keep the image small and free-tier friendly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Persisted vector store lives here; mount a volume to this path if you want
# data to survive container restarts on your host.
RUN mkdir -p /app/chroma_db
ENV CHROMA_DIR=/app/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
