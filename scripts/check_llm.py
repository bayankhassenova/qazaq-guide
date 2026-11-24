"""Step 5 check: a simple request-response with the LLM (no RAG yet).

Usage:  python -m scripts.check_llm "What is the capital of Kazakhstan?"
"""
import sys

from src.llm_client import LLMClient

question = " ".join(sys.argv[1:]) or "Say hello and tell me in one sentence what you can do."
llm = LLMClient()
print(f"Model: {llm.model}")
print(f"You: {question}")
print(f"AI:  {llm.ask(question)}")
