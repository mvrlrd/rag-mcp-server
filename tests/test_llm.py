"""Тесты обёртки OllamaLLM с fake-клиентом (без сети)."""

from rag_mcp.llm import OllamaLLM
from rag_mcp.retrieval import RetrievedChunk


class FakeClient:
    def __init__(self, reply="да"):
        self.reply = reply
        self.calls = []

    def chat(self, model, messages, options=None):
        self.calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": self.reply}}


def _chunk(text, source):
    return RetrievedChunk(
        id="x", document=text, source=source, chunk_index=0, score=1.0
    )


def test_rewrite_returns_stripped_reply():
    llm = OllamaLLM(model="m", client=FakeClient(reply="  новый запрос  "))
    assert llm.rewrite_query("старый") == "новый запрос"


def test_rewrite_falls_back_to_original_on_empty():
    llm = OllamaLLM(model="m", client=FakeClient(reply="   "))
    assert llm.rewrite_query("исходный вопрос") == "исходный вопрос"


def test_grade_parses_yes():
    llm = OllamaLLM(model="m", client=FakeClient(reply="Да, помогает"))
    assert llm.grade_chunk("вопрос", "фрагмент") is True


def test_grade_parses_no():
    llm = OllamaLLM(model="m", client=FakeClient(reply="Нет"))
    assert llm.grade_chunk("вопрос", "фрагмент") is False


def test_generate_includes_query_and_sources_in_prompt():
    fake = FakeClient(reply="Бюджет 42 попугая. Источники: note.md")
    llm = OllamaLLM(model="qwen", client=fake)
    chunks = [_chunk("Бюджет 42 попугая", "note.md")]

    answer = llm.generate_answer("какой бюджет?", chunks)

    assert answer == "Бюджет 42 попугая. Источники: note.md"
    sent = fake.calls[0]["messages"][0]["content"]
    assert "какой бюджет?" in sent
    assert "note.md" in sent
    assert "Бюджет 42 попугая" in sent
    assert fake.calls[0]["model"] == "qwen"
