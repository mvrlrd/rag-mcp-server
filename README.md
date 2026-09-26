# RAG Knowledge Base — MCP-сервер (Corrective RAG)

MCP-сервер, который индексирует локальную папку с документами и отвечает на вопросы
по ним через **Corrective RAG**-пайплайн на **LangGraph** и локальной LLM (**Ollama**).
Гибридный поиск (BM25 + векторный, слияние через RRF), корректирующая петля
`rewrite → retrieve → grade → (broaden →) generate`. Подключается к IDE-агенту
(VSCode Copilot / Claude) как MCP-инструмент.

## Требования

- **Docker** + **Docker Compose**.
- GPU не нужен: модель по умолчанию `qwen2.5:3b` — CPU-friendly.
- При первом старте автоматически скачивается модель (~1.9 ГБ).

## Запуск одной командой

```bash
docker compose up
```

Что поднимается:

1. **ollama** — сервис LLM (healthcheck по `ollama list`).
2. **ollama-pull** — одноразовый сервис: тянет `qwen2.5:3b`; MCP-сервер стартует
   только после успешного пула (`service_completed_successfully`).
3. **mcp-server** — MCP на `http://localhost:8000/mcp` (транспорт `streamable-http`).

Chroma-индекс сохраняется в volume `chroma_data`, модель — в `ollama_models`.

### Смена модели

Модель по умолчанию — `qwen2.5:3b`. Переопределить через env `RAG_LLM_MODEL`
(и подтянуть её в сервисе `ollama-pull`):

```bash
RAG_LLM_MODEL=llama3.2:3b docker compose up
```

### Ollama уже установлена на хосте (override)

Если Ollama уже крутится на хосте и модель скачана — можно не поднимать сервисы
`ollama`/`ollama-pull`, а направить сервер на host-Ollama:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 RAG_OLLAMA_HOST=http://host.docker.internal:11434 docker compose up --no-deps mcp-server
```

## Подключение MCP к IDE-агенту

Пример конфига — `mcp-config.example.json` (HTTP-транспорт, путь `/mcp`):

```json
{
  "servers": {
    "rag-kb": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Скопируй его в конфиг MCP своего клиента (VSCode Copilot / Claude) — агент увидит
4 инструмента и сам выберет нужный по описанию.

## Инструменты

| Инструмент | Назначение |
|-----------|-----------|
| `index_status()` | Статистика индекса: файлы, чанки, время индексации. |
| `index_folder(path, pattern="**/*")` | Индексация папки; `pattern` — glob (`*.md`, `docs/**/*.txt`). |
| `find_relevant_docs(query, top_k=5)` | Гибридный поиск сырых чанков (без генерации). |
| `ask_question(query)` | Полный Corrective RAG: ответ + источники. |

Типичный сценарий:

```
index_status()                       # chunks == 0 → база пуста
index_folder("./sample_docs")        # индексируем демо-документы
index_status()                       # проверяем: файлы/чанки > 0
find_relevant_docs("история института", top_k=3)
ask_question("Кто основал институт?")
```

## Разработка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src pytest -q       # 25 тестов
ruff check src tests           # линт
```

Конфигурация — через env-переменные с префиксом `RAG_` (см. `src/rag_mcp/config.py`):
`RAG_OLLAMA_HOST`, `RAG_LLM_MODEL`, `RAG_CHROMA_DIR`, `RAG_TOP_K`, `RAG_MAX_LOOPS` и др.

Подробнее об архитектуре — `ARCHITECTURE.md`, о процессе разработки — `REPORT.md`.
