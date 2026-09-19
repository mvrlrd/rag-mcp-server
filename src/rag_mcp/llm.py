"""Обёртка над Ollama: rewrite запроса, grade чанка, генерация ответа.

В графе Corrective RAG узлы получают объект LLM и вызывают эти методы.
В тестах вместо OllamaLLM подставляется mock с той же сигнатурой (см. Protocol).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import ollama

from .config import settings
from .retrieval import RetrievedChunk

REWRITE_PROMPT = (
    "Ты помогаешь искать документы в базе знаний. Переформулируй вопрос "
    "пользователя так, чтобы он лучше находил релевантные фрагменты: "
    "уточни термины, убери лишнее. Верни только переформулированный запрос "
    "без пояснений.\n\nВопрос: {query}"
)

GRADE_PROMPT = (
    "Оцени, помогает ли фрагмент ответить на вопрос. "
    "Ответь строго одним словом: да или нет.\n\n"
    "Вопрос: {query}\n\nФрагмент:\n{chunk}"
)

GENERATE_PROMPT = (
    "Ответь на вопрос, опираясь только на приведённый контекст. "
    "Если ответа в контексте нет — так и скажи. "
    "В конце укажи использованные источники.\n\n"
    "Вопрос: {query}\n\nКонтекст:\n{context}"
)


@runtime_checkable
class LLM(Protocol):
    def rewrite_query(self, query: str) -> str: ...
    def grade_chunk(self, query: str, chunk: str) -> bool: ...
    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str: ...


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, 1):
        blocks.append(f"[{i}] источник: {ch.source}\n{ch.document}")
    return "\n\n".join(blocks)


def _parse_yes_no(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith(("да", "yes"))


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None, client=None):
        self.model = model or settings.llm_model
        self.client = client or ollama.Client(host=host or settings.ollama_host)

    def _chat(self, prompt: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    def rewrite_query(self, query: str) -> str:
        out = self._chat(REWRITE_PROMPT.format(query=query)).strip()
        return out or query

    def grade_chunk(self, query: str, chunk: str) -> bool:
        return _parse_yes_no(self._chat(GRADE_PROMPT.format(query=query, chunk=chunk)))

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = _format_context(chunks)
        return self._chat(GENERATE_PROMPT.format(query=query, context=context)).strip()


def get_llm() -> OllamaLLM:
    return OllamaLLM()
