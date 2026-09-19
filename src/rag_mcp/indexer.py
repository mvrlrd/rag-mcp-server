"""Индексация локальных документов в ChromaDB.

Сканирует папку, читает поддерживаемые файлы, режет на чанки, пишет в Chroma
(persist на диск). Также хранит статистику последней индексации.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .config import settings

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}

_LANGUAGE_BY_EXT = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
}

_client: chromadb.ClientAPI | None = None


class HashingEmbeddingFunction(EmbeddingFunction):
    """Детерминированные offline-эмбеддинги (bag-of-words hashing).

    Используются в тестах, чтобы не тянуть модель из сети. В продакшене
    по умолчанию работает встроенная модель ChromaDB.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def name() -> str:
        return "hashing"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbeddingFunction:
        return HashingEmbeddingFunction(dim=config.get("dim", 256))


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client


def reset_client() -> None:
    """Сбросить закешированный клиент (нужно тестам при смене chroma_dir)."""
    global _client
    _client = None


def default_embedding_function():
    """Эмбеддинги: Ollama, если задан RAG_EMBED_MODEL, иначе встроенные Chroma."""
    if settings.embed_model:
        from chromadb.utils import embedding_functions

        return embedding_functions.OllamaEmbeddingFunction(
            url=f"{settings.ollama_host}/api/embeddings",
            model_name=settings.embed_model,
        )
    return None


def get_collection():
    ef = default_embedding_function()
    if ef is None:
        return get_client().get_or_create_collection(settings.collection_name)
    return get_client().get_or_create_collection(
        settings.collection_name, embedding_function=ef
    )


def _split_file(path: Path, text: str) -> list[str]:
    ext = path.suffix.lower()
    if ext in _LANGUAGE_BY_EXT:
        splitter = RecursiveCharacterTextSplitter.from_language(
            _LANGUAGE_BY_EXT[ext],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return [c for c in splitter.split_text(text) if c.strip()]


def _status_path() -> Path:
    return Path(settings.chroma_dir) / "index_status.json"


def _save_status(stats: dict) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats), encoding="utf-8")


def index_folder(path: str, pattern: str = "**/*", collection=None) -> dict:
    """Проиндексировать папку с документами. Возвращает статистику индексации.

    ``pattern`` — glob относительно ``path`` (по умолчанию рекурсивно все файлы).
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Не папка: {path}")

    col = collection or get_collection()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    files = 0

    for fp in sorted(root.glob(pattern)):
        if not fp.is_file() or fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip():
            continue
        files += 1
        for i, chunk in enumerate(_split_file(fp, text)):
            ids.append(f"{fp}::{i}")
            docs.append(chunk)
            metas.append({"source": str(fp), "chunk_index": i})

    if docs:
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    stats = {
        "files": files,
        "chunks": len(docs),
        "indexed_at": time.time(),
        "path": str(root),
    }
    _save_status(stats)
    return stats


def index_status(collection=None) -> dict:
    """Текущее состояние индекса: файлы, чанки, время последней индексации."""
    col = collection or get_collection()
    count = col.count()
    path = _status_path()
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "files": saved.get("files", 0),
        "chunks": count,
        "indexed_at": saved.get("indexed_at"),
        "last_path": saved.get("path"),
    }
