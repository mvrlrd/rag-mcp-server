"""Сборка графа Corrective RAG и точка входа ask_question."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..llm import LLM, get_llm
from . import nodes
from .nodes import RetrieveFn
from .state import GraphState


def build_graph(llm: LLM, retrieve_fn: RetrieveFn | None = None):
    retrieve_fn = retrieve_fn or nodes._default_retrieve

    g = StateGraph(GraphState)
    g.add_node("rewrite", nodes.make_rewrite(llm))
    g.add_node("retrieve", nodes.make_retrieve(retrieve_fn))
    g.add_node("grade", nodes.make_grade(llm))
    g.add_node("broaden", nodes.make_broaden(llm))
    g.add_node("generate", nodes.make_generate(llm))

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        nodes.decide_after_grade,
        {"generate": "generate", "broaden": "broaden"},
    )
    g.add_edge("broaden", "retrieve")
    g.add_edge("generate", END)

    return g.compile()


def ask_question(query: str, llm: LLM | None = None, retrieve_fn=None) -> dict:
    """Прогнать вопрос через граф, вернуть ответ и источники."""
    llm = llm or get_llm()
    graph = build_graph(llm, retrieve_fn)
    final = graph.invoke({"query": query, "loop_count": 0})
    return {"answer": final.get("answer", ""), "sources": final.get("sources", [])}
