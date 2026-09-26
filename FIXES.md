# Исправления по замечаниям преподавателя

| # | Проблема | Подтверждено | Файл(ы) | Исправление | Статус |
|---|----------|:---:|---------|-------------|:------:|
| 1 | Описания MCP-инструментов слишком общие — агент не знает когда их использовать | ✅ | `server.py` | Добавлен `instructions=` в `FastMCP` с описанием предметной области; улучшены docstring `find_relevant_docs` и `ask_question` | ✅ Done |
| 2 | `broaden` недописан — повторный поиск идёт с тем же запросом | ✅ | `llm.py`, `graph/nodes.py`, `graph/build.py`, `tests/test_graph.py` | Добавлены `BROADEN_PROMPT` и `OllamaLLM.broaden_query`; `broaden` заменён на `make_broaden(llm)` — вызывает LLM для расширения запроса; добавлен тест `test_broaden_changes_query` | ✅ Done |
| 3 | `make_grade` может отвергать нормальные ответы (английский "No", неразобранный ответ) | ⚠️ частично | `llm.py` | `_parse_yes_no`: явная обработка "no"/"нет" → False; либеральный fallback — неразобранный ответ → True; улучшен `GRADE_PROMPT` с few-shot примерами | ✅ Done |
| 4 | Нет логов в узлах графа | ✅ | `graph/nodes.py` | Добавлен `logging` во все узлы: rewrite, retrieve, grade, broaden, generate | ✅ Done |
| 5 | `RAG_OLLAMA_HOST` захардкожен в `docker-compose.yml` — нельзя переопределить через env | ✅ | `docker-compose.yml` | `http://ollama:11434` → `${RAG_OLLAMA_HOST:-http://ollama:11434}` | ✅ Done |

## Верификация

```
PYTHONPATH=src pytest -q   # 26 passed (было 25, добавлен test_broaden_changes_query)
ruff check src tests       # All checks passed!
```

## Замена тестового корпуса

| Параметр | Было | Стало |
|---|---|---|
| Корпус | старый набор (26 файлов, плоская структура) | `flatland_cosmology` (48 файлов, 8 поддиректорий) |
| Тематика | фиктивный проект «Гидра-7» | ИПК им. А. Н. Квадратова (плоская космология) |
| Форматы | md, py, json, yaml | md, txt, csv, json, yaml |
| Структура | `api/`, `code/`, `config/`, `modules/`, `runbooks/` | `01_obzor/`, `02_teoriya/`, `03_istoriya/`, `04_persony/`, `05_nablyudeniya/`, `06_protokoly/`, `07_dannye/`, `08_specifikacii/` |
| Тест | проверял «Гидра-7», «Марфа Кузнецова», «малахитовый барсук» | проверяет «Квадратов», «Абсцисса», «Хордова» |

Коммит: `54f7733` (`chore(corpus): replace sample_docs with flatland_cosmology dataset`)

ANSWER_KEY.md (25 тестовых вопросов с ответами) находится в `sample_docs/ANSWER_KEY.md`.
