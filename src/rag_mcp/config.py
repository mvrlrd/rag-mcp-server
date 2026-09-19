"""Настройки приложения. Всё через env-переменные (без хардкода)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore"
    )

    # --- Ollama ---
    # По умолчанию host.docker.internal — чтобы контейнер ходил в Ollama на хосте.
    ollama_host: str = "http://host.docker.internal:11434"
    llm_model: str = "qwen2.5:3b"
    # Опциональные эмбеддинги через Ollama. Пусто — встроенные ChromaDB.
    embed_model: str = ""

    # --- Хранилище / индексация ---
    chroma_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    docs_dir: str = "./sample_docs"

    # --- Чанкинг ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Поиск ---
    top_k: int = 5
    # Кол-во кандидатов от каждого ретривера до RRF-слияния.
    candidate_k: int = 10
    rrf_k: int = 60

    # --- Граф Corrective RAG ---
    max_loops: int = 2


settings = Settings()
