# Архитектура: RAG Knowledge Base MCP-сервер

Документ описывает архитектуру проекта. Первоначально написан до кода как проектный
ориентир, затем актуализирован под финальную реализацию.

## 1. Цель

MCP-сервер, который индексирует локальную папку с документами и отвечает на вопросы
по ним через **Corrective RAG**-пайплайн на локальной LLM (Ollama). Запускается одной
командой `docker compose up`, подключается к IDE-агенту (VSCode Copilot / Claude).

## 2. Стек

| Слой | Технология | Роль |
|------|-----------|------|
| MCP | `FastMCP` | Регистрация инструментов, транспорт `streamable-http` на :8000, путь `/mcp` |
| Оркестрация | `langgraph` | Граф Corrective RAG с условными переходами |
| RAG-утилиты | `langchain` | Loaders, text splitters |
| Векторное хранилище | `chromadb` | Плотный поиск, persist на диск |
| Sparse retrieval | `rank_bm25` | Лексический поиск (BM25) |
| Слияние | RRF | Reciprocal Rank Fusion плотного и разреженного |
| LLM | Ollama (`qwen2.5:3b`) | Rewrite / grade / generate |
| Конфиг | `pydantic-settings` | Настройки через env, без хардкода |
| Тесты | `pytest` | Unit + e2e |

## 3. Компоненты (`src/rag_mcp/`)

```
server.py     — FastMCP: 4 инструмента + подробные description
config.py     — Settings (модель, пути, OLLAMA_HOST, top_k, chunk params)
indexer.py    — скан файлов → загрузка → чанкинг → эмбеддинги → Chroma
retrieval.py  — BM25 + vector + RRF (гибридный поиск)
llm.py        — обёртка Ollama (generate, grade, rewrite) + mock для тестов
graph/
  state.py    — TypedDict состояния графа
  nodes.py    — rewrite / retrieve / grade / generate / broaden
  build.py    — сборка графа, условные рёбра, лимит циклов
```

## 4. MCP-инструменты

Агент выбирает инструмент сам по `description`, поэтому описания — содержательные.

1. **`index_folder(path, pattern="**/*")`** — проиндексировать папку с документами
   в базу знаний. `pattern` — glob относительно `path` (например `*.md`,
   `docs/**/*.txt`; по умолчанию рекурсивно все файлы).
2. **`index_status()`** — статистика индекса: кол-во файлов, чанков, время индексации.
3. **`find_relevant_docs(query, top_k)`** — гибридный поиск релевантных фрагментов
   **без** генерации ответа (сырые ранжированные чанки + источники).
4. **`ask_question(query)`** — полный Corrective RAG: возвращает синтезированный ответ
   с указанием источников.

## 5. Пайплайн Corrective RAG (LangGraph)

```
        ┌─────────┐
        │ rewrite │  переформулировать запрос под retrieval
        └────┬────┘
             ▼
        ┌──────────┐
        │ retrieve │  гибрид BM25 + vector → RRF
        └────┬─────┘
             ▼
        ┌───────┐
        │ grade │  LLM оценивает релевантность чанков (yes/no)
        └───┬───┘
            ▼
      «достаточно?»  ──нет, и loops < max──►  ┌─────────┐
            │                                 │ broaden │ расширить запрос
            │ да / лимит                      └────┬────┘
            ▼                                      │
        ┌──────────┐   ◄─────────────────────────┘ (обратно в retrieve)
        │ generate │  синтез ответа + источники
        └────┬─────┘
             ▼
            END
```

### Состояние графа (`state.py`)

```python
class GraphState(TypedDict):
    query: str              # исходный вопрос
    rewritten_query: str    # переформулированный запрос
    chunks: list            # найденные фрагменты (Document)
    graded: list            # прошедшие grade релевантные фрагменты
    loop_count: int         # счётчик циклов retrieve (лимит = max_loops)
    answer: str             # финальный ответ
    sources: list           # источники (пути файлов)
```

### Условные переходы

- После `grade`: если релевантных чанков достаточно **или** достигнут `max_loops` (=2) →
  `generate`; иначе → `broaden` → `retrieve` (retry).
- Это и есть «корректирующая» петля: при плохом retrieval запрос расширяется и поиск
  повторяется, но не более 2 раз (защита от бесконечного цикла).

## 6. Индексация

- Поддерживаемые типы: `.md/.txt/.py/.js/.ts/.json/.yaml`.
- Текст → `RecursiveCharacterTextSplitter`; код → сплиттер с учётом языка.
- Метаданные чанка: `source` (путь), `chunk_index`.
- ChromaDB с persist-директорией (`RAG_CHROMA_DIR`).
- Эмбеддинги: встроенные ChromaDB по умолчанию; опционально Ollama-модель через
  `RAG_EMBED_MODEL`. В тестах — offline `HashingEmbeddingFunction`
  (детерминированные bag-of-words hashing-эмбеддинги, без обращения к сети).

## 7. Гибридный поиск + RRF

Два независимых ретривера возвращают по `candidate_k` кандидатов:
- **Vector** (Chroma) — семантическая близость.
- **BM25** (`rank_bm25`) — лексическое совпадение.

Слияние **Reciprocal Rank Fusion**:

```
score(d) = Σ_retrievers 1 / (rrf_k + rank_r(d))
```

где `rank_r(d)` — позиция документа в списке ретривера `r`, `rrf_k` = 60.
Итог — top_k документов по суммарному RRF-скору. RRF детерминирован и не требует LLM,
поэтому легко тестируется.

## 8. Развёртывание

Финальный сценарий — **Ollama внутри compose** (одна команда `docker compose up`):
- **ollama** — сервис LLM с healthcheck (`ollama list`).
- **ollama-pull** — одноразовый сервис: тянет `qwen2.5:3b`; `restart: "no"`. MCP-сервер
  `depends_on` его через `condition: service_completed_successfully` — стартует только
  после успешного пула модели.
- **mcp-server** — `depends_on` ollama по `condition: service_healthy`; ходит в LLM по
  `RAG_OLLAMA_HOST=http://ollama:11434`. Chroma persist — в volume (`RAG_CHROMA_DIR=/data/chroma`).
- Транспорт `streamable-http`, порт 8000, путь `/mcp` (`mcp-config.example.json`).

**Опциональный host-Ollama override.** Если Ollama уже установлена на хосте и модель
скачана, можно не поднимать сервисы `ollama`/`ollama-pull`, а направить сервер на
host-Ollama: `RAG_OLLAMA_HOST=http://host.docker.internal:11434 docker compose up mcp-server`
(это же значение — дефолт в `config.py`).

## 9. Тестирование

Всего **25 тестов** (`PYTHONPATH=src pytest -q`):

- `test_indexer.py` — загрузка, чанкинг, запись/чтение Chroma; индекс `sample_docs`
  содержит зашитые проверочные факты.
- `test_retrieval.py` — корректность RRF-слияния (детерминированно, без LLM).
- `test_graph.py` — ветки графа с **mock LLM**: достаточно/недостаточно чанков, retry-лимит.
- `test_llm.py` — парсинг grade (да/нет), форматирование контекста, fallback rewrite.
- `test_tools_e2e.py` — e2e 4 MCP-инструментов через in-memory `fastmcp.Client`,
  обёрнутый в `asyncio.run` (без `pytest-asyncio`); offline hashing-эмбеддинги и mock LLM.
