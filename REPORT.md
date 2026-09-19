# Отчёт о работе над дипломным проектом

Журнал ведётся по ходу разработки: инструменты, промпты, проблемы и решения.

## Выбор темы

Выбран вариант **RAG MCP-сервер** (вместо full-stack) — цель развить новые навыки:
MCP, LangGraph, Corrective RAG, локальные LLM (Ollama). Приоритет — надёжно закрыть
критерии сдачи, глубина вторична.

## Стек и обоснование

- **FastMCP** — минимум кода для MCP-сервера с инструментами.
- **LangGraph** — граф с условными переходами идеально ложится на Corrective RAG.
- **ChromaDB** — in-process векторное хранилище с persist, без внешнего сервиса.
- **rank_bm25** — простой lexical retrieval для гибрида.
- **Ollama + qwen2.5:3b** — CPU-friendly локальная модель, работает без GPU.
- **pydantic-settings** — конфиг через env, без хардкода.

## Инструменты разработки

- Claude Code (генерация кода, структура проекта).
- Git-flow: `main` / `develop` / `feature/*`, атомарные коммиты, мерджи через PR
  (merge-commit = аналог `--no-ff`), CI на GitHub Actions (ruff + pytest).

## Журнал этапов

### Этап 0 — Подготовка и документация
- Создан git-репозиторий, `.gitignore`, скелет.
- Написан ARCHITECTURE.md **до** кода (архитектура графа, компоненты, RRF).
- Заведён REPORT.md.
- `config.py` со всеми настройками (модель, пути, chunk params, top_k, max_loops).

### Этап 00-ci — CI (перенесён раньше)
- `.github/workflows/ci.yml`: `ruff check src tests` + `PYTHONPATH=src pytest -q`.
- С этого этапа — PR-flow: push feature-ветки → `gh pr create --base develop` →
  зелёный CI → `gh pr merge --merge --delete-branch`.

### Этап 1 — Индексация (`indexer.py`)
- Loaders для `.md/.txt/.py/.js/.ts/.json/.yaml/.yml`, чанкинг
  `RecursiveCharacterTextSplitter` (для кода — `from_language`), метаданные
  `source`/`chunk_index`, persist в ChromaDB. `index_status` со статистикой.

### Этап 2 — Гибридный поиск (`retrieval.py`)
- Vector (Chroma) + BM25 (`rank_bm25`), слияние через RRF (`rrf_k=60`).
- `find_relevant_docs(query, top_k)` — детерминированно, без LLM.

### Этап 3 — LLM-обёртка (`llm.py`)
- `OllamaLLM`: rewrite / grade (yes/no) / generate. `LLM` Protocol для подмены mock
  в тестах. Промпты `REWRITE_PROMPT` / `GRADE_PROMPT` / `GENERATE_PROMPT`.

### Этап 4 — LangGraph Corrective RAG (`graph/`)
- Узлы rewrite → retrieve → grade → (broaden →) generate, условное ребро
  `decide_after_grade`, лимит `max_loops=2`. Узлы — фабрики с замыканием зависимостей.

### Этап 5 — MCP-сервер (`server.py`) · PR #4
- FastMCP `rag-kb`, 4 инструмента с подробными `description`, graceful-ошибки
  (папка не найдена, пустой индекс, Ollama недоступна). glob в `index_folder`.
- `test_tools_e2e.py`: in-memory `fastmcp.Client`, `asyncio.run`, offline hashing, mock LLM.

### Этап 6 — Демо-документы + проверочные факты · PR #5
- `sample_docs/` ~612 КБ (md/txt/py/json/yaml), 8 зашитых проверочных фактов.
- Тест: индекс `sample_docs` содержит зашитый факт.

### Этап 7 — Инфраструктура · PR #6
- `Dockerfile`, `.dockerignore`, `docker-compose.yml` (ollama healthcheck + one-shot
  `ollama-pull` + mcp-server), `mcp-config.example.json` (http `/mcp` :8000).
- Проверено вживую: 27 файлов / 565 чанков; `ask_question` вернул зашитые факты
  («Марфа Кузнецова», «42 попугая») с источниками.

### Этап 8 — Финализация документации
- README (запуск одной командой, требования, примеры, проверочные факты, подключение MCP).
- ARCHITECTURE актуализирован под финальную реализацию (streamable-http, Ollama в compose, glob).
- REPORT дозаполнен (журнал этапов, проблемы/решения, примеры промптов, выводы).
- Финальный релиз `develop → main`.

## Проблемы и решения

- **ruff `RUF100`**: `# noqa: BLE001` не сработал — правило `BLE001` не входит в `select`
  в `ruff.toml`, поэтому noqa считался «лишним». Решение: убрали noqa (широкий `except`
  в `server.py` оправдан — отдаём агенту сообщение вместо traceback).
- **Нет `pytest-asyncio`**: e2e MCP-инструменты гоняем через `asyncio.run` вокруг
  async-хелпера с in-memory `fastmcp.Client(server.mcp)` — не тянем лишнюю зависимость.
- **`env_prefix="RAG_"`**: в compose переменные именуются `RAG_OLLAMA_HOST` /
  `RAG_LLM_MODEL` / `RAG_MCP_*`, а не голый `OLLAMA_HOST` — иначе pydantic-settings их
  не подхватит.
- **Путь MCP `/mcp`**: FastMCP `streamable_http_path` по умолчанию `/mcp` — согласовали
  URL в `mcp-config.example.json` и порт 8000.
- **Один mutable `settings`-синглтон**: в тестах `monkeypatch.setattr(indexer.settings, ...)`
  + `indexer.reset_client()`, чтобы chroma-dir и эмбеддинги переключались чисто между тестами.
- **One-shot pull модели**: `ollama-pull` с `restart: "no"`; mcp-server стартует только
  после `condition: service_completed_successfully` — не бьётся в LLM до скачивания модели.
- **`sample_docs ≥500 КБ`**: сгенерировали связную (не lorem) документацию из пулов фраз
  + вшили проверочные факты; поймали и убрали артефакт-обрезку слов в шаблоне метрики.

## Примеры промптов

### Удачный: `ask_question` с двумя фактами

Вопрос: **«Кто ведущий инженер и какой годовой бюджет?»**

Пайплайн: `rewrite` уточняет запрос → `retrieve` (BM25+vector→RRF) достаёт чанки из
`01_team_and_roles.md` и `00_project_overview.md` → `grade` (`GRADE_PROMPT`, ответ
строго «да/нет») отбирает релевантные → `generate` (`GENERATE_PROMPT`: «опирайся только
на контекст… в конце укажи источники») синтезирует ответ.

Результат (реальный e2e на Ollama `qwen2.5:3b`): ответ содержит оба факта —
*ведущий инженер **Марфа Кузнецова***, *бюджет **42 попугая*** — и перечисляет
файлы-источники. `GENERATE_PROMPT` с явным «только из контекста» удерживает модель
от галлюцинаций: факты специфичны и в интернете отсутствуют, значит ответ — из нашей базы.

### Корректирующая петля: grade → broaden

Если по нерелевантному/узкому запросу `grade` не отобрал ни одного чанка
(`state["graded"]` пуст) и `loop_count < max_loops`, `decide_after_grade` возвращает
`broaden`: `broaden` инкрементит `loop_count`, а `retrieve` расширяет выдачу
(`top_k * (loop + 1)`) и поиск повторяется. Так «корректирующий» RAG сам добирает
контекст при слабом первом retrieval, но не более 2 повторов (защита от бесконечного
цикла). Если и после лимита релевантных чанков нет — `generate` работает по сырым
чанкам, а модель по промпту честно сообщает, что ответа в контексте нет.

## Выводы

- **Corrective RAG** даёт устойчивость к слабому retrieval: петля grade→broaden
  добирает контекст без участия пользователя, а лимит циклов защищает от зацикливания.
- **Гибрид BM25 + vector → RRF** покрывает и лексические (точные имена/числа —
  проверочные факты), и семантические запросы; RRF детерминирован и легко тестируется
  без LLM.
- **Ради надёжной сдачи упростили**: offline hashing-эмбеддинги в тестах (нет сети),
  `asyncio.run` вместо `pytest-asyncio`, mock LLM в графовых тестах, CPU-friendly
  `qwen2.5:3b`. Приоритет — воспроизводимый `docker compose up` и зелёный CI.
