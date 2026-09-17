"""Узлы графа Corrective RAG.

Узлы создаются фабриками, замыкающими зависимости (LLM и функцию поиска),
чтобы в тестах их можно было подменить mock-объектами.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import settings
from ..llm import LLM
from ..retrieval import RetrievedChunk, find_relevant_docs
from .state import GraphState

RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def _default_retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
    return find_relevant_docs(query, top_k=top_k)


def make_rewrite(llm: LLM) -> Callable[[GraphState], dict]:
    def rewrite(state: GraphState) -> dict:
        return {"rewritten_query": llm.rewrite_query(state["query"])}

    return rewrite


def make_retrieve(retrieve_fn: RetrieveFn) -> Callable[[GraphState], dict]:
    def retrieve(state: GraphState) -> dict:
        # Каждый цикл broaden расширяет выдачу, чтобы захватить больше кандидатов.
        loop = state.get("loop_count", 0)
        top_k = settings.top_k * (loop + 1)
        chunks = retrieve_fn(state["rewritten_query"], top_k)
        return {"chunks": chunks}

    return retrieve


def make_grade(llm: LLM) -> Callable[[GraphState], dict]:
    def grade(state: GraphState) -> dict:
        query = state["query"]
        graded = [c for c in state["chunks"] if llm.grade_chunk(query, c.document)]
        return {"graded": graded}

    return grade


def broaden(state: GraphState) -> dict:
    return {"loop_count": state.get("loop_count", 0) + 1}


def make_generate(llm: LLM) -> Callable[[GraphState], dict]:
    def generate(state: GraphState) -> dict:
        chunks = state["graded"] or state["chunks"]
        answer = llm.generate_answer(state["query"], chunks)
        sources = list(dict.fromkeys(c.source for c in chunks))
        return {"answer": answer, "sources": sources}

    return generate


def decide_after_grade(state: GraphState) -> str:
    if state["graded"]:
        return "generate"
    if state.get("loop_count", 0) >= settings.max_loops:
        return "generate"
    return "broaden"
