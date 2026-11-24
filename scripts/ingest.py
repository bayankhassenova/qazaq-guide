"""Fill the vector database with the dataset.

Steps: load JSON -> build text chunks -> create embeddings -> upload to Qdrant.

Usage:
    python -m scripts.ingest            # create/update the collection
    python -m scripts.ingest --recreate # delete and build the collection from scratch
"""
import argparse
import time

from src.config import get_settings
from src.dataset import load_records, record_to_chunk, record_to_payload
from src.embeddings_client import EmbeddingsClient
from src.vector_db import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Qazaq Guide dataset into Qdrant")
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    settings = get_settings()
    records = load_records()
    print(f"Loaded {len(records)} records from data/places.json")

    chunks = [record_to_chunk(r) for r in records]
    payloads = [record_to_payload(r) for r in records]
    ids = [r["id"] for r in records]

    print(f"Creating embeddings with '{settings.embedding_model}' ({settings.provider})...")
    started = time.time()
    embedder = EmbeddingsClient(settings)
    vectors = embedder.embed_batch(chunks)
    print(f"Created {len(vectors)} vectors of size {len(vectors[0])} in {time.time() - started:.1f}s")

    store = VectorStore(settings.qdrant_url, settings.collection_name)
    store.ensure_collection(vector_size=len(vectors[0]), recreate=args.recreate)
    store.upsert(ids, vectors, payloads)
    print(f"Done. Collection '{settings.collection_name}' now has {store.count()} points.")
    print(f"Open the Qdrant dashboard: {settings.qdrant_url}/dashboard")


if __name__ == "__main__":
    main()
