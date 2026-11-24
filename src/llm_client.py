"""LLM client: simple request-response chat with the chosen provider."""
from __future__ import annotations

import time

import openai

from .config import Settings, get_settings, make_openai_client

# Errors that mean "the service is busy, try again or try another model".
# NotFoundError: the provider retired this model, so we skip to the next one.
RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError,
             openai.APIConnectionError, openai.NotFoundError)


class LLMClient:
    def __init__(self, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self.model = self.settings.chat_model
        self.last_model_used = self.model
        self._client = client or make_openai_client(self.settings)

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """Send a list of chat messages and return the assistant's answer text.

        Free AI tiers are sometimes overloaded (HTTP 503) or rate limited (429).
        Then we wait a little and try the next model from the fallback list.
        """
        models = [self.model, *getattr(self.settings, "fallback_models", ())]
        last_error = None
        for attempt, model in enumerate(models):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.settings.temperature if temperature is None else temperature,
                )
                self.last_model_used = model
                return response.choices[0].message.content or ""
            except RETRYABLE as exc:
                last_error = exc
                print(f"[LLM] {model} is busy or unavailable ({type(exc).__name__}), trying the next model...")
                time.sleep(1 + attempt)
        raise RuntimeError(f"All models are busy right now, please try again in a minute. Last error: {last_error}")

    def ask(self, question: str, system_prompt: str | None = None) -> str:
        """The simplest possible interaction: one question in, one answer out."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return self.chat(messages)
