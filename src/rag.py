"""The RAG pipeline: question -> embedding -> vector search -> LLM answer."""
from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings_client import EmbeddingsClient
from .llm_client import LLMClient
from .vector_db import VectorStore

SYSTEM_PROMPT = """You are Qazaq Guide, a friendly travel assistant for Kazakhstan.

Rules:
1. Answer ONLY using the facts in the CONTEXT section (documents from the knowledge base).
2. After each fact, cite the document number in square brackets, e.g. [1] or [2].
3. If the context does not contain the answer, say honestly that the knowledge base has
   no information about it. Do not invent places, prices, dates or distances.
4. Detect the language of the QUESTION and write the whole answer in exactly that language
   (English question -> English answer, Russian question -> Russian answer).
   Do not start with a greeting.
5. Be practical: when relevant mention the best season, how to get there and useful tips.
6. Keep the answer short and well structured (a short intro, then a bullet list)."""

NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in the knowledge base. "
    "Try rephrasing the question or removing the filters."
)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    prompt_messages: list[dict] = field(default_factory=list)


def build_context(documents: list[dict]) -> str:
    """Join the found documents into one numbered block for the prompt."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(f"[{i}] (source: {doc['source']})\n{doc['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, documents: list[dict], history: list[dict] | None = None) -> list[dict]:
    """Put the user's question and the found documents into ONE context window."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # A few previous turns help with follow-up questions ("and how do I get there?").
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": (f"CONTEXT:\n{build_context(documents)}\n\nQUESTION: {question}\n\n"
                    "Write the answer in the same language as the QUESTION above."),
    })
    return messages


class RAGPipeline:
    def __init__(self, embedder: EmbeddingsClient, store: VectorStore, llm: LLMClient):
        self.embedder = embedder
        self.store = store
        self.llm = llm

    def retrieve(self, question: str, top_k: int = 4, filters: dict | None = None,
                 score_threshold: float | None = None) -> list[dict]:
        # Step 1: convert the user input to an embedding (the search vector).
        query_vector = self.embedder.embed(question)
        # Step 2: find the most similar documents in the vector database.
        query_filter = self.store.build_filter(**(filters or {}))
        return self.store.search(query_vector, top_k=top_k, query_filter=query_filter,
                                 score_threshold=score_threshold)

    def answer(self, question: str, top_k: int = 4, filters: dict | None = None,
               score_threshold: float | None = None, history: list[dict] | None = None) -> RAGAnswer:
        documents = self.retrieve(question, top_k=top_k, filters=filters, score_threshold=score_threshold)
        if not documents:
            return RAGAnswer(question=question, answer=NO_CONTEXT_ANSWER)
        # Step 3: send the question + found documents to the LLM in one prompt.
        messages = build_messages(question, documents, history)
        answer = self.llm.chat(messages)
        return RAGAnswer(question=question, answer=answer, sources=documents, prompt_messages=messages)
