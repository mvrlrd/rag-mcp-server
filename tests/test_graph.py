"""Тесты графа Corrective RAG с mock LLM (без сети и без Chroma)."""

from rag_mcp.config import settings
from rag_mcp.graph.build import ask_question, build_graph
from rag_mcp.retrieval import RetrievedChunk


class MockLLM:
    def __init__(self, grade_result):
        self.grade_result = grade_result
        self.grade_calls = 0

    def rewrite_query(self, query):
        return f"{query} [rw]"

    def grade_chunk(self, query, chunk):
        self.grade_calls += 1
        return self.grade_result

    def generate_answer(self, query, chunks):
        sources = dict.fromkeys(c.source for c in chunks)
        return "answer:" + ",".join(sources)


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def __call__(self, query, top_k):
        self.calls.append({"query": query, "top_k": top_k})
        return [
            RetrievedChunk(
                id="1", document="d1", source="a.md", chunk_index=0, score=1.0
            ),
            RetrievedChunk(
                id="2", document="d2", source="a.md", chunk_index=1, score=0.9
            ),
        ]


def test_enough_chunks_goes_straight_to_generate():
    retriever = RecordingRetriever()
    llm = MockLLM(grade_result=True)

    out = ask_question("вопрос", llm=llm, retrieve_fn=retriever)

    assert out["answer"] == "answer:a.md"
    assert out["sources"] == ["a.md"]  # deduped
    assert len(retriever.calls) == 1  # no broaden retries


def test_rewrite_query_is_used_for_retrieval():
    retriever = RecordingRetriever()
    ask_question("вопрос", llm=MockLLM(grade_result=True), retrieve_fn=retriever)
    assert retriever.calls[0]["query"] == "вопрос [rw]"


def test_retry_limit_then_generate_anyway():
    retriever = RecordingRetriever()
    llm = MockLLM(grade_result=False)  # никогда не достаточно

    out = ask_question("вопрос", llm=llm, retrieve_fn=retriever)

    # первичный retrieve + max_loops повторов после broaden
    assert len(retriever.calls) == settings.max_loops + 1
    # broaden расширяет top_k на каждой итерации
    assert retriever.calls[-1]["top_k"] > retriever.calls[0]["top_k"]
    # всё равно генерируем ответ (fallback на сырые чанки)
    assert out["answer"] == "answer:a.md"


def test_graph_compiles():
    assert build_graph(MockLLM(True), RecordingRetriever()) is not None
