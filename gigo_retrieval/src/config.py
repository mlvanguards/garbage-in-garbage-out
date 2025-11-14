from typing import Optional
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    BASE_DIR: Path = Path(__file__).parent.parent

    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "hybrid_collection"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    LLM_MODEL_NAME: str = "anthropic/claude-3-7-sonnet-20250219"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()