"""Qazaq Guide - command line chat (alternative to the web UI).

Run:  python cli.py
"""
from src.config import get_settings
from src.embeddings_client import EmbeddingsClient
from src.llm_client import LLMClient
from src.rag import RAGPipeline
from src.vector_db import VectorStore


def main() -> None:
    s = get_settings()
    pipeline = RAGPipeline(EmbeddingsClient(s), VectorStore(s.qdrant_url, s.collection_name), LLMClient(s))
    print("Qazaq Guide CLI. Ask a question about travelling in Kazakhstan (empty line to exit).")
    history = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        result = pipeline.answer(question, top_k=s.top_k, history=history)
        print(f"\nGuide: {result.answer}\n")
        for i, d in enumerate(result.sources, 1):
            print(f"  [{i}] {d['name']} (similarity {d['score']:.3f})")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]


if __name__ == "__main__":
    main()
