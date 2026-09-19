"""E2E-тесты 4 MCP-инструментов через in-memory FastMCP client (без сети).

Chroma пишется в tmp, эмбеддинги — offline hashing, LLM для ask_question —
mock. Проверяется полный цикл: пустой индекс → индексация → статистика →
поиск → вопрос-ответ.
"""

import asyncio

import pytest
from fastmcp import Client

from rag_mcp import indexer, server
from rag_mcp.graph import build as graph_build
from rag_mcp.indexer import HashingEmbeddingFunction
from rag_mcp.retrieval import RetrievedChunk


class MockLLM:
    def rewrite_query(self, query):
        return query

    def grade_chunk(self, query, chunk):
        return True

    def generate_answer(self, query, chunks: list[RetrievedChunk]):
        joined = " ".join(c.document for c in chunks)
        fact = "Гидра-7" if "Гидра-7" in joined else ""
        return f"Ответ по базе. {fact}".strip()


@pytest.fixture
def mcp_env(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "project.md").write_text(
        "# Проект Гидра\n\n"
        "Кодовое имя — Гидра-7. Бюджет проекта: 42 попугая.\n"
        "Ответственный инженер — Марфа Кузнецова.\n",
        encoding="utf-8",
    )
    (docs / "notes.txt").write_text(
        "Запуск Гидра-7 запланирован на четверг. Резервный канал связи открыт.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(indexer.settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(
        indexer, "default_embedding_function", lambda: HashingEmbeddingFunction()
    )
    monkeypatch.setattr(graph_build, "get_llm", lambda: MockLLM())
    indexer.reset_client()
    yield docs
    indexer.reset_client()


def _call(name, args=None):
    async def run():
        async with Client(server.mcp) as client:
            res = await client.call_tool(name, args or {})
            return res.data

    return asyncio.run(run())


def test_full_tool_cycle(mcp_env):
    docs = mcp_env

    status = _call("index_status")
    assert status["chunks"] == 0

    stats = _call("index_folder", {"path": str(docs)})
    assert stats["files"] == 2
    assert stats["chunks"] >= 2

    status = _call("index_status")
    assert status["chunks"] == stats["chunks"]
    assert status["files"] == 2

    docs_found = _call("find_relevant_docs", {"query": "бюджет Гидра попугай"})
    assert isinstance(docs_found, list)
    assert docs_found
    assert all({"document", "source", "score"} <= set(d) for d in docs_found)
    scores = [d["score"] for d in docs_found]
    assert scores == sorted(scores, reverse=True)
    assert any("Гидра-7" in d["document"] for d in docs_found)


def test_ask_question_with_mock_llm(mcp_env):
    docs = mcp_env
    _call("index_folder", {"path": str(docs)})

    out = _call("ask_question", {"query": "Какое кодовое имя проекта?"})
    assert "Гидра-7" in out["answer"]
    assert out["sources"]


def test_index_folder_missing_path_returns_error(mcp_env):
    out = _call("index_folder", {"path": "/no/such/dir/xyz"})
    assert "error" in out


def test_ask_question_on_empty_index_returns_error(mcp_env):
    out = _call("ask_question", {"query": "что угодно"})
    assert "error" in out
    assert out["answer"] == ""
