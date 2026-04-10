# План: переписать experiment 13 без LangGraph

## Контекст

Experiment 13 использует `LangGraph create_react_agent` с gemma4:26b через Ollama.
Проблема: gemma4 через Ollama не поддерживает надёжно ни tool calling, ни JSON
format — все три подхода (response_format, with_structured_output, ollama format=schema)
возвращают пустой content или ломаются.

Experiment 12 работает стабильно — он использует `OllamaClient.generate()` напрямую
и парсит JSON из текста. Это единственный рабочий паттерн с этой моделью.

## Подход: ручной agent loop без LangGraph

Убрать LangGraph целиком. Реализовать простой цикл:
1. Отправить промпт через `OllamaClient.generate()` — с инструкцией отвечать JSON
2. Если модель пишет `SEARCH: <запрос>` — выполнить поиск через `ddgs.DDGS().text()`
3. Добавить результат в контекст, повторить (макс. 2 итерации поиска)
4. Парсить JSON из финального ответа (как в exp 12: `json.loads` от `[` до `]`)

### Почему этот подход правильный:
- **Exp 12 уже так работает** — `OllamaClient.generate()` + JSON парсинг, стабильно на 26K пар
- **Минимальные изменения** — заменяем только `create_agent` / `label_one_pair`, остальная инфраструктура (blocking, resume, sharding, traces) остаётся
- **Web search сохраняется** — просто через `ddgs` напрямую вместо LangChain tool
- **Нет зависимости от tool calling** — модель отвечает текстом, мы парсим JSON

### Что меняется в файле:
1. **Удалить**: `from langchain_*`, `from langgraph.*`, класс `_ChatOllamaFC`
2. **Добавить**: `from ddgs import DDGS`, импорт `OllamaClient`
3. **Переписать `create_agent()`** → `create_client()` — возвращает `OllamaClient`
4. **Переписать `label_one_pair()`** — ручной цикл: generate → [search → generate] → parse JSON
5. **Обновить промпт** — убрать "вызови submit_answer", добавить "ответь JSON" + "напиши SEARCH: запрос если нужен поиск"
6. **Убрать `_extract_structured`** — JSON парсится из ответа модели напрямую (как `parse_llm_response` в exp 12)
7. **Обновить `label_candidates()`** — убрать `model`/`host` параметры, передавать `OllamaClient`

### Что НЕ меняется:
- Blocking, FAISS, build_candidates
- Resume-логика, sharding, merge_shards
- Trace writer, logging
- main() структура (аргументы, flow)
- MatchResult schema (для типизации результата)

## Файлы
- `experiments/13_label_with_agent.py` — основные изменения
- `src/table_unifier/ollama_client.py` — только чтение (переиспользуем)

## Верификация
```bash
# Удалить старые результаты
rm -f data/labeled/labeled_pairs_agent*.parquet data/labeled/agent_traces*.jsonl data/labeled/agent_labeling*.log
# Запустить
uv run python experiments/13_label_with_agent.py --skip-blocking --host http://nvidia-server:11434
# Проверить что первые 5-10 пар размечены без ошибок
```
