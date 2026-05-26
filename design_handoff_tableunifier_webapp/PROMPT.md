# Промт для Claude Code

Скопируй в Claude Code CLI и запусти из корня репозитория `TableUnifier`.

---

Я хочу превратить мой ML-проект **TableUnifier** в работающее веб-приложение для **инференса** Entity Resolution. Все необходимые материалы лежат в папке `design_handoff_tableunifier_webapp/`:

- `README.md` — полная спецификация API + дизайн-токены + описание экранов + чеклист реализации
- `design/` — HTML/React/CSS прототип фронтенда (12 файлов, dark theme, all screens, mock-данные)
- `PROMPT.md` — этот файл

**Что нужно сделать:**

1. Прочитай `design_handoff_tableunifier_webapp/README.md` целиком.
2. Прочитай `CLAUDE.md` и `src/table_unifier/` чтобы понять существующий код (особенно `dataset/embedding_generation.py`, `dataset/graph_builder.py`, `models/entity_resolution.py`, `training/er_trainer.py` — функции `get_row_embeddings`, `find_duplicates`).
3. Создай пакет `src/table_unifier/server/` со всеми эндпоинтами и WebSocket-стримом из README.
4. Скопируй файлы из `design_handoff_tableunifier_webapp/design/` в `src/table_unifier/server/static/`, отдавай через `StaticFiles`.
5. Замени `data.js` на `api.js` — реальные fetch-вызовы + WebSocket-подписку. Обнови все 5 экранов чтобы использовать API вместо setTimeout-моков.
6. Добавь зависимости в `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `python-multipart`, `openpyxl`, `umap-learn`.
7. Добавь команду запуска в `CLAUDE.md`:
   ```bash
   uv run python -m table_unifier.server.main --host 0.0.0.0 --port 8000
   ```
8. Протестируй end-to-end на двух .xlsx файлах из `data/raw_ru/auto_ru/` (если они есть) или на любых тестовых таблицах из `tests/conftest.py`.

**Важно:**
- Это приложение **для инференса**, не для обучения. Чекпоинт уже есть в `output/bge-m3/v17_views_gat_model.pt`. Не дублируй обучающий код.
- Не ломай существующие эксперименты и тесты — сервер должен быть отдельным модулем.
- Если Ollama недоступна (типичный кейс при локальной отладке), бэкенд должен возвращать понятную ошибку через WebSocket, а не падать молча.
- Соблюди визуальный дизайн пиксель-в-пиксель — все токены в `styles.css` уже определены.

Начни с чтения README и плана работ.
