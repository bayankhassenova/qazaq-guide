"""Offline tests of the whole RAG pipeline.

They do NOT call a real AI provider: the embeddings API and the LLM are replaced
with small fakes, and Qdrant runs in memory. This checks our own code (dataset,
chunking, database, filters, prompt building) without spending API credits.

Run:  python -m pytest -q
"""
import hashlib
import math
import re
from types import SimpleNamespace

from qdrant_client import QdrantClient

from src.config import Settings
from src.dataset import load_records, record_to_chunk, record_to_payload
from src.embeddings_client import EmbeddingsClient
from src.llm_client import LLMClient
from src.rag import RAGPipeline, build_messages
from src.vector_db import VectorStore

DIM = 256
SETTINGS = Settings(
    provider="openai", api_key="test", dial_endpoint="", dial_api_version="",
    chat_model="fake-chat", fallback_models=(), embedding_model="fake-embed", embedding_dimensions=None,
    qdrant_url=":memory:", collection_name="test", top_k=4, temperature=0.0,
)


def fake_vector(text: str) -> list[float]:
    """Bag-of-words hashing vector: texts with common words get similar vectors."""
    v = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeOpenAI:
    """Mimics the parts of the OpenAI SDK that our clients use."""

    def __init__(self):
        self.last_messages = None
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _embed(self, model, input, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=fake_vector(t)) for i, t in enumerate(input)])

    def _chat(self, model, messages, temperature):
        self.last_messages = messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fake answer [1]"))])


def build_pipeline():
    fake = FakeOpenAI()
    records = load_records()
    embedder = EmbeddingsClient(SETTINGS, client=fake)
    store = VectorStore(collection="test", client=QdrantClient(":memory:"))
    vectors = embedder.embed_batch([record_to_chunk(r) for r in records])
    store.ensure_collection(DIM)
    store.upsert([r["id"] for r in records], vectors, [record_to_payload(r) for r in records])
    return RAGPipeline(embedder, store, LLMClient(SETTINGS, client=fake)), store, fake


def test_dataset_is_valid_and_annotated():
    records = load_records()
    assert len(records) >= 40
    assert all(r["description"] and r["source"].startswith("http") for r in records)


def test_all_records_are_uploaded():
    _, store, _ = build_pipeline()
    assert store.count() == len(load_records())


def test_vector_search_finds_relevant_place():
    pipeline, _, _ = build_pipeline()
    docs = pipeline.retrieve("Khoja Ahmed Yasawi mausoleum Turkistan", top_k=3)
    assert docs[0]["doc_id"] == "turkistan-yasawi"


def test_metadata_filter_is_applied():
    pipeline, _, _ = build_pipeline()
    docs = pipeline.retrieve("lake", top_k=8, filters={"season": "winter"})
    assert docs and all("winter" in d["best_seasons"] for d in docs)


def test_question_and_documents_go_to_llm_in_one_prompt():
    pipeline, _, fake = build_pipeline()
    result = pipeline.answer("Charyn Canyon Valley of Castles", top_k=2)
    user_prompt = fake.last_messages[-1]["content"]
    assert "Charyn Canyon" in user_prompt and "QUESTION: Charyn Canyon" in user_prompt
    assert result.answer == "fake answer [1]" and len(result.sources) == 2


def test_history_is_limited():
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    messages = build_messages("q", [], history)
    assert len(messages) == 1 + 4 + 1  # system + last 4 turns + question
