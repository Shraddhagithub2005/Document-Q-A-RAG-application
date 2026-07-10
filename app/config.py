"""
Centralized configuration, loaded from environment variables.
Keeping all env access in one place makes the app easy to test and deploy.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Anthropic API key used for answer generation. Required for /ask to work.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Which Claude model to use for generation.
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Optional simple API key to protect this service's endpoints.
    # If left blank, auth is disabled (fine for local dev, NOT for public deploy).
    APP_API_KEY: str = os.getenv("APP_API_KEY", "")

    # Where Chroma persists its on-disk index.
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")

    # Chunking parameters.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # How many chunks to retrieve per question by default.
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "4"))


settings = Settings()
