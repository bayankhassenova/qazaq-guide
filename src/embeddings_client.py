"""Embeddings client: turns any text chunk into a searchable vector."""
from __future__ import annotations

from typing import Sequence

from .config import Settings, get_settings, make_openai_client


class EmbeddingsClient:
    """Thin wrapper around the embeddings endpoint of the chosen provider."""

    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions
        # `client` can be injected (for example a fake one in tests).
        self._client = client or make_openai_client(self.settings)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
        """Return one vector per text. Texts are sent in batches to save requests."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[start:start + batch_size]]
            kwargs = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            response = self._client.embeddings.create(**kwargs)
            data = list(response.data)
            # Keep the original order of the inputs. Some providers (e.g. Gemini)
            # do not fill the `index` field, then the order of the list is used.
            if all(getattr(d, "index", None) is not None for d in data):
                data.sort(key=lambda d: d.index)
            vectors.extend(list(d.embedding) for d in data)
        return vectors

    def embed(self, text: str) -> list[float]:
        """Return the vector for a single text (used for the user's question)."""
        return self.embed_batch([text])[0]
