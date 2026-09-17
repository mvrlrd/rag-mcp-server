"""Общие фикстуры тестов."""

import uuid

import chromadb
import pytest

from rag_mcp.indexer import HashingEmbeddingFunction


@pytest.fixture
def collection():
    """Ephemeral Chroma-коллекция с offline-эмбеддингами (без сети)."""
    client = chromadb.EphemeralClient()
    return client.create_collection(
        f"test_kb_{uuid.uuid4().hex}",
        embedding_function=HashingEmbeddingFunction(),
    )


@pytest.fixture
def sample_docs(tmp_path):
    """Небольшой набор файлов для индексации."""
    (tmp_path / "note.md").write_text(
        "# Проект Гидра\n\n"
        "Кодовое имя проекта — Гидра-7. Бюджет составляет 42 попугая.\n"
        "Ответственный инженер: Марфа Кузнецова.\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "def launch_hydra():\n"
        "    # запуск проекта Гидра-7\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "ignore.bin").write_text("binary-ish", encoding="utf-8")
    return tmp_path
