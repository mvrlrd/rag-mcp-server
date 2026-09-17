"""Состояние графа Corrective RAG."""

from __future__ import annotations

from typing import TypedDict

from ..retrieval import RetrievedChunk


class GraphState(TypedDict, total=False):
    query: str
    rewritten_query: str
    chunks: list[RetrievedChunk]
    graded: list[RetrievedChunk]
    loop_count: int
    answer: str
    sources: list[str]
