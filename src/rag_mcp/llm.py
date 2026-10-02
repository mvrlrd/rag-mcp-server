"""Обёртка над Ollama: rewrite запроса, grade чанка, генерация ответа.

В графе Corrective RAG узлы получают объект LLM и вызывают эти методы.
В тестах вместо OllamaLLM подставляется mock с той же сигнатурой (см. Protocol).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import ollama

from .config import settings
from .retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "Ты помогаешь искать документы в специализированной базе знаний. "
    "Переформулируй вопрос так, чтобы лучше находить релевантные фрагменты: "
    "уточни смысл, убери лишнее. "
    "ВАЖНО: сохраняй все имена, названия и термины из вопроса без изменений — "
    "они могут быть специфичными для данной предметной области. "
    "Верни только переформулированный запрос без пояснений.\n\nВопрос: {query}"
)

BROADEN_PROMPT = (
    "Ты помогаешь расширить поисковый запрос, когда первоначальный запрос "
    "не нашёл достаточно релевантных документов. Убери лишние уточнения и "
    "сделай формулировку проще и короче. Сохраняй ключевые имена, названия "
    "и термины из исходного запроса без изменений — не заменяй их синонимами "
    "или общеупотребительными понятиями, это может быть специфичная "
    "терминология базы знаний. Верни только новый запрос без пояснений.\n\n"
    "Исходный запрос: {query}"
)

GRADE_PROMPT = (
    "Ты определяешь, помогает ли фрагмент документа ответить на вопрос.\n\n"
    "Пример 1 (прямой ответ есть):\n"
    "Источник: sotrudniki.md\n"
    "Фрагмент:\nИванов работает в компании директором с 2010 года.\n\n"
    "Вопрос: Кто директор компании?\n\n"
    "Цитата: Иванов работает в компании директором с 2010 года.\n"
    "Оценка: 10\n\n"
    "Пример 2 (та же тема, но без прямого ответа):\n"
    "Источник: observatorii.md\n"
    "Фрагмент:\nНа обсерватории Нормаль работает смотритель Петров. У него есть "
    "собака, но её имя нигде не указано.\n\n"
    "Вопрос: Как зовут собаку обсерватории Нормаль?\n\n"
    "Цитата: НЕТ\n"
    "Оценка: 3\n\n"
    "Пример 3 (не по теме):\n"
    "Источник: byudzhet.md\n"
    "Фрагмент:\nБюджет отдела продаж на 2021 год составил 5 млн рублей.\n\n"
    "Вопрос: Как зовут кошку на обсерватории?\n\n"
    "Цитата: НЕТ\n"
    "Оценка: 0\n\n"
    "Теперь оцени реальный фрагмент.\n\n"
    "Источник: {source}\n"
    "Фрагмент:\n{chunk}\n\n"
    "Вопрос: {query}\n\n"
    "Выполни два шага:\n"
    "1) Найди в фрагменте дословную цитату, отвечающую на вопрос. "
    "Если её нет — напиши НЕТ.\n"
    "2) Оцени релевантность по шкале 0-10: 0 — не по теме; "
    "3-5 — та же тема, но без ответа; 8-10 — прямой ответ.\n\n"
    "Ответь СТРОГО в формате:\nЦитата: <цитата или НЕТ>\nОценка: <0-10>\n\n"
    "Твой ответ:"
)

GENERATE_PROMPT = (
    "Ответь на вопрос коротко и точно, опираясь ТОЛЬКО на контекст ниже. "
    "Не пересказывай контекст целиком. Не добавляй ничего от себя. "
    "Если ответа в контексте нет — скажи: «Информация отсутствует».\n\n"
    "Вопрос: {query}\n\nКонтекст:\n{context}"
)


@dataclass
class GradeResult:
    score: int
    quote: str | None


@runtime_checkable
class LLM(Protocol):
    def rewrite_query(self, query: str) -> str: ...
    def broaden_query(self, query: str) -> str: ...
    def grade_chunk(self, query: str, chunk: str, source: str = "") -> GradeResult: ...
    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str: ...


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, 1):
        blocks.append(f"[{i}] источник: {ch.source}\n{ch.document}")
    return "\n\n".join(blocks)


_SCORE_RE = re.compile(r"оценка\s*:?\s*(\d+)", re.IGNORECASE)
_QUOTE_RE = re.compile(r"цитата\s*:?\s*(.+)", re.IGNORECASE)


def _parse_grade(text: str) -> GradeResult:
    quote = None
    quote_match = _QUOTE_RE.search(text)
    if quote_match:
        raw_quote = quote_match.group(1).splitlines()[0].strip()
        if raw_quote and raw_quote.upper() != "НЕТ":
            quote = raw_quote

    score_match = _SCORE_RE.search(text)
    if not score_match:
        # Неразобранный ответ: либеральный fallback — оставляем чанк
        return GradeResult(score=10, quote=quote)
    score = max(0, min(10, int(score_match.group(1))))
    return GradeResult(score=score, quote=quote)


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None, client=None):
        self.model = model or settings.llm_model
        self.client = client or ollama.Client(host=host or settings.ollama_host)

    def _chat(self, prompt: str, extra_options: dict | None = None) -> str:
        options = {"temperature": 0.0}
        if extra_options:
            options.update(extra_options)
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return resp["message"]["content"]

    def rewrite_query(self, query: str) -> str:
        out = self._chat(REWRITE_PROMPT.format(query=query)).strip()
        return out or query

    def broaden_query(self, query: str) -> str:
        out = self._chat(BROADEN_PROMPT.format(query=query)).strip()
        return out or query

    def grade_chunk(self, query: str, chunk: str, source: str = "") -> GradeResult:
        prompt = GRADE_PROMPT.format(query=query, chunk=chunk, source=source)
        text = self._chat(prompt, extra_options={"num_predict": 80})
        result = _parse_grade(text)
        if result.quote and result.quote not in chunk:
            logger.warning(
                "grade_chunk: quote not in chunk (possible hallucination): %r",
                result.quote,
            )
        return result

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = _format_context(chunks)
        return self._chat(GENERATE_PROMPT.format(query=query, context=context)).strip()


def get_llm() -> OllamaLLM:
    return OllamaLLM()
