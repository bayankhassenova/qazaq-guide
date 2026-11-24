"""Application settings.

All settings are read from environment variables (or from a local `.env` file),
so secrets such as API keys are never stored in the code or in Git.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Default model names for every supported provider.
PROVIDER_DEFAULTS = {
    # EPAM DIAL (Azure OpenAI compatible API)
    "dial": {
        "chat_model": "gpt-4o-mini-2024-07-18",
        "embedding_model": "text-embedding-3-small-1",
    },
    # OpenAI platform
    "openai": {
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    },
    # Google Gemini through its OpenAI compatible endpoint (has a free tier)
    "gemini": {
        "chat_model": "gemini-3.1-flash-lite",
        # Used automatically when the main model is overloaded or unavailable.
        "fallback_models": ["gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-3.5-flash-lite"],
        "embedding_model": "gemini-embedding-001",
    },
}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    dial_endpoint: str
    dial_api_version: str
    chat_model: str
    fallback_models: tuple
    embedding_model: str
    embedding_dimensions: int | None
    qdrant_url: str
    collection_name: str
    top_k: int
    temperature: float


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "dial").strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDER_DEFAULTS)}"
        )
    defaults = PROVIDER_DEFAULTS[provider]
    dims = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
    return Settings(
        provider=provider,
        api_key=os.getenv("API_KEY", "").strip(),
        dial_endpoint=os.getenv("DIAL_ENDPOINT", "https://ai-proxy.lab.epam.com").strip(),
        dial_api_version=os.getenv("DIAL_API_VERSION", "2024-02-01").strip(),
        chat_model=os.getenv("CHAT_MODEL", "").strip() or defaults["chat_model"],
        fallback_models=tuple(defaults.get("fallback_models", [])),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or defaults["embedding_model"],
        embedding_dimensions=int(dims) if dims else None,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        collection_name=os.getenv("COLLECTION_NAME", "qazaq_guide").strip(),
        top_k=int(os.getenv("TOP_K", "4")),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
    )


def make_openai_client(settings: Settings):
    """Create an OpenAI SDK client for the selected provider.

    DIAL exposes an Azure OpenAI compatible API, OpenAI and Gemini expose the
    standard OpenAI API, so one SDK covers all three providers.
    """
    from openai import AzureOpenAI, OpenAI

    if not settings.api_key:
        raise RuntimeError("API_KEY is empty. Copy .env.example to .env and put your key there.")
    if settings.provider == "dial":
        return AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.dial_endpoint,
            api_version=settings.dial_api_version,
            timeout=45, max_retries=0,
        )
    if settings.provider == "gemini":
        return OpenAI(api_key=settings.api_key, base_url=GEMINI_BASE_URL, timeout=45, max_retries=0)
    return OpenAI(api_key=settings.api_key, timeout=45, max_retries=0)
