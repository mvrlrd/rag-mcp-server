"""Узлы графа Corrective RAG.

Узлы создаются фабриками, замыкающими зависимости (LLM и функцию поиска),
чтобы в тестах их можно было подменить mock-объектами.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..config import settings
from ..llm import LLM
from ..retrieval import RetrievedChunk, find_relevant_docs
from .state import GraphState

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


def _default_retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
    return find_relevant_docs(query, top_k=top_k)


def make_rewrite(llm: LLM) -> Callable[[GraphState], dict]:
    def rewrite(state: GraphState) -> dict:
        original = state["query"]
        rewritten = llm.rewrite_query(original)
        logger.info("rewrite: %r → %r", original, rewritten)
        return {"rewritten_query": rewritten}

    return rewrite


def make_retrieve(retrieve_fn: RetrieveFn) -> Callable[[GraphState], dict]:
    def retrieve(state: GraphState) -> dict:
        # Каждый цикл broaden расширяет выдачу, чтобы захватить больше кандидатов.
        loop = state.get("loop_count", 0)
        top_k = settings.top_k * (loop + 1)
        query = state["rewritten_query"]
        chunks = retrieve_fn(query, top_k)
        logger.info("retrieve: query=%r top_k=%d found=%d", query, top_k, len(chunks))
        return {"chunks": chunks}

    return retrieve


def make_grade(llm: LLM) -> Callable[[GraphState], dict]:
    def grade(state: GraphState) -> dict:
        query = state["query"]
        graded = [c for c in state["chunks"] if llm.grade_chunk(query, c.document)]
        logger.info("grade: %d/%d chunks passed", len(graded), len(state["chunks"]))
        return {"graded": graded}

    return grade


def make_broaden(llm: LLM) -> Callable[[GraphState], dict]:
    def broaden(state: GraphState) -> dict:
        loop = state.get("loop_count", 0) + 1
        old_query = state["rewritten_query"]
        new_query = llm.broaden_query(old_query)
        if new_query.lower() == old_query.lower():
            logger.warning("broaden: loop=%d LLM returned same query, keeping as-is", loop)
            new_query = old_query
        else:
            logger.info("broaden: loop=%d, %r → %r", loop, old_query, new_query)
        return {"loop_count": loop, "rewritten_query": new_query}

    return broaden


def make_generate(llm: LLM) -> Callable[[GraphState], dict]:
    def generate(state: GraphState) -> dict:
        chunks = state["graded"] or state["chunks"]
        sources = list(dict.fromkeys(c.source for c in chunks))
        logger.info("generate: %d chunks, sources=%s", len(chunks), sources)
        answer = llm.generate_answer(state["query"], chunks)
        return {"answer": answer, "sources": sources}

    return generate


def decide_after_grade(state: GraphState) -> str:
    if state["graded"]:
        return "generate"
    if state.get("loop_count", 0) >= settings.max_loops:
        return "generate"
    return "broaden"
