"""Step 4 check: semantic search in the database from the command line (no LLM).

Usage:  python -m scripts.search "where to see ancient history in the south"
"""
import sys

from src.config import get_settings
from src.embeddings_client import EmbeddingsClient
from src.vector_db import VectorStore

query = " ".join(sys.argv[1:]) or "beautiful lake in the mountains"
settings = get_settings()
vector = EmbeddingsClient(settings).embed(query)
results = VectorStore(settings.qdrant_url, settings.collection_name).search(vector, top_k=5)
print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['name']}  (score {r['score']:.3f}, {r['region']}, {r['category']})")
