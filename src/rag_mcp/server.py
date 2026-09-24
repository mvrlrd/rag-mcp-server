"""MCP-сервер базы знаний (FastMCP).

Регистрирует 4 инструмента поверх RAG-пайплайна. Описания инструментов
подробные — чтобы агент сам выбирал нужный tool без подсказки пользователя.
Транспорт по умолчанию — streamable-http (запуск в контейнере, подключение из IDE).
"""

from __future__ import annotations

import logging
import os

from fastmcp import FastMCP

from . import indexer
from .graph.build import ask_question as _ask_question
from .retrieval import find_relevant_docs as _find_relevant_docs

mcp = FastMCP(
    "rag-kb",
    instructions=(
        "База знаний внутренней проектной документации (проект Гидра-7). "
        "Содержит: кодовые имена проектов и релизов (Гидра-7, Аметист 3.14), "
        "бюджеты (например, 42 попугая), персонал (имена и роли сотрудников, "
        "например Марфа Кузнецова — ведущий инженер), инфраструктуру "
        "(дата-центры, порты, кодовые слова), технические характеристики систем. "
        "Используй этот сервер ВСЕГДА, когда пользователь спрашивает о "
        "кодовых именах, бюджетах, сотрудниках, портах, релизах или любых "
        "внутренних фактах проекта — даже если вопрос кажется простым."
    ),
)


@mcp.tool
def index_folder(path: str, pattern: str = "**/*") -> dict:
    """Проиндексировать локальную папку с документами в базу знаний.

    Читает поддерживаемые файлы (.md, .txt, .py, .js, .ts, .json, .yaml, .yml),
    режет их на чанки и сохраняет в векторное хранилище. Вызови этот инструмент
    ПЕРЕД поиском или вопросами, если база ещё не наполнена, либо когда документы
    изменились и индекс надо обновить.

    Args:
        path: путь к папке с документами (например, "./sample_docs").
        pattern: glob-шаблон относительно path. По умолчанию "**/*" —
            рекурсивно все файлы. Примеры: "*.md" (только markdown в корне),
            "docs/**/*.txt" (txt в подпапке docs).

    Returns:
        Статистику индексации: число файлов, чанков, путь и время.
    """
    try:
        return indexer.index_folder(path, pattern=pattern)
    except FileNotFoundError:
        return {"error": f"Папка не найдена: {path}"}
    except NotADirectoryError:
        return {"error": f"Указанный путь не является папкой: {path}"}


@mcp.tool
def index_status() -> dict:
    """Показать текущее состояние индекса базы знаний.

    Возвращает количество проиндексированных файлов и чанков, а также время
    последней индексации. Используй, чтобы проверить, наполнена ли база, перед
    тем как искать документы или задавать вопросы. Если chunks == 0 — база
    пуста, сначала вызови index_folder.
    """
    return indexer.index_status()


@mcp.tool
def find_relevant_docs(query: str, top_k: int = 5) -> list[dict]:
    """Найти релевантные фрагменты проектной документации (без генерации ответа).

    Выполняет гибридный поиск (BM25 + векторный, слияние через RRF) и возвращает
    ранжированные чанки с их источниками и оценкой. Используй, когда нужны СЫРЫЕ
    фрагменты из документации (процитировать, посмотреть контекст), а не готовый
    ответ. Подходит для поиска по кодовым именам, именам сотрудников, техническим
    параметрам, датам и другим фактам. Для полноценного ответа используй ask_question.

    Args:
        query: поисковый запрос на естественном языке.
        top_k: сколько лучших фрагментов вернуть (по умолчанию 5).

    Returns:
        Список фрагментов: {id, document, source, chunk_index, score}.
        Пустой список, если индекс пуст.
    """
    chunks = _find_relevant_docs(query, top_k=top_k)
    return [
        {
            "id": c.id,
            "document": c.document,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "score": c.score,
        }
        for c in chunks
    ]


@mcp.tool
def ask_question(query: str) -> dict:
    """Ответить на вопрос по документации проекта Гидра-7 через Corrective RAG-пайплайн.

    Используй для ЛЮБЫХ вопросов о внутреннем проекте: кодовые имена (Гидра-7,
    Аметист 3.14), бюджеты (42 попугая), сотрудники (Марфа Кузнецова),
    инфраструктура (Тундра-9, порт 47281, кодовое слово «малахитовый барсук»),
    технические характеристики (9000 квазаров/с).

    Прогоняет вопрос через граф: переформулировка → гибридный поиск →
    оценка релевантности чанков → при нехватке контекста расширение поиска
    (до 2 повторов) → генерация ответа локальной LLM (Ollama).
    Возвращает готовый ответ со ссылками на источники.

    Требует запущенной Ollama с загруженной моделью. Индекс должен быть наполнен
    (см. index_folder / index_status).

    Args:
        query: вопрос на естественном языке.

    Returns:
        {"answer": текст ответа, "sources": список файлов-источников}.
    """
    status = indexer.index_status()
    if not status.get("chunks"):
        return {
            "error": "Индекс пуст. Сначала вызови index_folder, чтобы наполнить базу.",
            "answer": "",
            "sources": [],
        }
    try:
        return _ask_question(query)
    except Exception as exc:  # сообщение агенту вместо traceback
        return {
            "error": (
                "Не удалось получить ответ от LLM. Проверь, что Ollama запущена "
                f"и модель загружена. Детали: {exc}"
            ),
            "answer": "",
            "sources": [],
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    transport = os.environ.get("RAG_MCP_TRANSPORT", "streamable-http")
    host = os.environ.get("RAG_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_MCP_PORT", "8000"))
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
