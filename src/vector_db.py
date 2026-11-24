"""Vector database layer (Qdrant): create collection, upload vectors, search."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client import models

# Payload fields we filter on get an index for fast filtering.
KEYWORD_INDEXES = ["category", "region", "budget", "best_seasons", "doc_type"]


def point_id(doc_id: str) -> str:
    """Stable UUID from the record id, so re-running ingestion updates instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qazaq-guide/{doc_id}"))


class VectorStore:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "qazaq_guide",
                 client: QdrantClient | None = None):
        self.client = client or QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            current = self.client.get_collection(self.collection).config.params.vectors.size
            if current != vector_size:
                raise ValueError(
                    f"Collection has vector size {current}, but the model returns {vector_size}. "
                    "Run the ingestion with --recreate."
                )
            return
        self.client.create_collection(
            collection_name=self.collection,
            # Cosine similarity is the standard choice for text embeddings.
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(
                self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            models.PointStruct(id=point_id(i), vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    @staticmethod
    def build_filter(category: str | None = None, region: str | None = None,
                     season: str | None = None, budget: str | None = None) -> models.Filter | None:
        conditions = []
        if category:
            conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
        if region:
            conditions.append(models.FieldCondition(key="region", match=models.MatchValue(value=region)))
        if season:
            # best_seasons is a list; MatchValue matches if the list contains the value.
            conditions.append(models.FieldCondition(key="best_seasons", match=models.MatchValue(value=season)))
        if budget:
            conditions.append(models.FieldCondition(key="budget", match=models.MatchValue(value=budget)))
        return models.Filter(must=conditions) if conditions else None

    def search(self, query_vector: list[float], top_k: int = 4,
               query_filter: models.Filter | None = None, score_threshold: float | None = None) -> list[dict]:
        """Vector (semantic) search. Returns payloads with similarity scores."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in result.points]
