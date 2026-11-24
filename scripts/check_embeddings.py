"""Step 3 check: create an embedding and show how similar texts get similar vectors.

Usage:  python -m scripts.check_embeddings
"""
import math

from src.embeddings_client import EmbeddingsClient


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


texts = [
    "A mountain lake with turquoise water near Almaty",
    "Alpine lake in the mountains, good for a day trip",
    "Ancient mausoleum and pilgrimage site in the south",
]
client = EmbeddingsClient()
vectors = client.embed_batch(texts)
print(f"Model: {client.model}, vector size: {len(vectors[0])}")
print(f"First 5 numbers of vector 1: {[round(x, 4) for x in vectors[0][:5]]}")
print(f"Similarity 1 vs 2 (both about lakes): {cosine(vectors[0], vectors[1]):.3f}")
print(f"Similarity 1 vs 3 (lake vs mausoleum): {cosine(vectors[0], vectors[2]):.3f}")
