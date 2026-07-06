import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

class LLMSettings(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 1000
    max_retries: int = 3


class OpenAISettings(LLMSettings):

    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    default_model: str = "gpt-4.1-mini"

    embedding_model: str = "text-embedding-3-small"


class VectorStoreSettings(BaseModel):

    table_name: str = "documents"

    embedding_dimensions: int = 1536

    qdrant_host: str = "localhost"

    qdrant_port: int = 6333


class Settings(BaseModel):

    openai: OpenAISettings = Field(default_factory=OpenAISettings)

    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)


@lru_cache()
def get_settings() -> Settings:

    setup_logging()

    settings = Settings()

    if not settings.openai.api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please add it to your .env file."
        )

    return settings