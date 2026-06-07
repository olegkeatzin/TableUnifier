# Handoff: TableUnifier — Web UI + Inference Server

## Overview

Этот пакет содержит дизайн **веб-приложения** для существующего ML-проекта **TableUnifier** (`github.com/olegkeatzin/TableUnifier`). Приложение позволяет пользователю:

1. Загрузить несколько Excel/CSV-таблиц
2. Запустить **инференс** обученной GNN/GAT-модели (Entity Resolution) — НЕ обучение
3. Визуально посмотреть гетерогенный граф (row + token nodes)
4. Посмотреть пошаговую работу алгоритма (message passing, сходимость эмбеддингов, кластеризация)
5. Вручную проверить и подтвердить/отклонить пары-кандидаты
6. Получить унифицированную таблицу с экспортом в .xlsx/.csv/.parquet

## About the Design Files

Файлы в этом бандле — **дизайн-референсы, сделанные в HTML + React (через CDN, без сборки)**. Это прототип, показывающий нужный внешний вид и поведение.

**Важно — фронт уже написан под реальный API.** Все 5 экранов уже вызывают `window.API` (файл `api.js`): `uploadFiles`,
`buildGraph`, `getGraph`, `runInference`, `subscribeRun` (WebSocket), `getClusters`,
`postDecisions`, `exportUrl`. Никакого `data.js`/setTimeout-мока в боевом фронте нет.
Единственный мок — `_preview/mock-api.js` — нужен **только** чтобы открыть прототип без
бэкенда (`_preview/_preview.html`); в production его не подключать.

**Задача**: реализовать бэкенд под эти вызовы (REST + WebSocket, см. спек ниже), который вызывает функции из `src/table_unifier/`, и раздать фронт как статику. Сам React-фронтенд можно либо оставить в HTML-виде (тогда нужен только Python-бэкенд + статика), либо переписать на Vite + React + TypeScript — на выбор.

## Fidelity

**High-fidelity** — финальные цвета, типографика, отступы, анимации. Дизайн нужно воссоздать пиксель-в-пиксель.

## Архитектура: что нужно построить

```
┌──────────────────────────────────┐    ┌───────────────────────────────┐
│  Frontend (этот прототип)        │    │  Backend — НОВЫЙ КОД          │
│  React + CSS (dark theme)        │◄──►│  FastAPI + Uvicorn            │
│  • 5 экранов pipeline            │ws  │  • REST endpoints             │
│  • SVG-визуализации              │    │  • WebSocket для прогресса    │
└──────────────────────────────────┘    │  • Вызовы src/table_unifier/  │
                                        └───────────┬───────────────────┘
                                                    │
                                        ┌───────────▼───────────────────┐
                                        │  Существующий код TableUnifier │
                                        │  src/table_unifier/            │
                                        │  • dataset/embedding_generation│
                                        │  • dataset/graph_builder       │
                                        │  • models/entity_resolution    │
                                        │  • evaluation/clustering (CC)  │
                                        │  • output/bge-m3/v17_views_gat │
                                        │    _model.pt (обученные веса)  │
                                        └────────────────────────────────┘
                                                    │
                                        ┌───────────▼───────────────────┐
                                        │  External: Ollama @ nvidia-srv │
                                        │  qwen3-embedding:8b · column   │
                                        └────────────────────────────────┘
```

## Backend API Spec

### Стек

- **Python 3.12+**, `uv` для зависимостей (репо уже использует uv)
- **FastAPI** + **uvicorn**
- **websockets** (или встроенные FastAPI WebSockets) для стрима прогресса
- **pandas** + **openpyxl** для парсинга .xlsx
- **pyarrow** для .parquet
- Существующий `torch`, `torch_geometric`, `deap`, `transformers` из `pyproject.toml` (HDBSCAN убрали — эксперимент не зашёл)

### Структура нового кода

Создать новый пакет `src/table_unifier/server/`:

```
src/table_unifier/server/
├── __init__.py
├── main.py              # FastAPI app + uvicorn entrypoint
├── routes/
│   ├── __init__.py
│   ├── sources.py       # POST /sources/upload, GET /sources, DELETE /sources/{id}
│   ├── graph.py         # POST /graph/build, GET /graph/{run_id}
│   ├── infer.py         # POST /infer/run (запуск инференса), WS /infer/{run_id}/stream
│   ├── clusters.py      # GET /clusters/{run_id}, POST /clusters/{run_id}/decisions
│   └── export.py        # GET /export/{run_id}.{format}
├── services/
│   ├── inference.py     # Orchestrates: graph_build → forward pass → similarity → CC
│   ├── progress.py      # WebSocket broadcaster
│   └── storage.py       # In-memory store of run_id → state; sessions persist to .runs/
└── models.py            # Pydantic models for request/response
```

CLI запуск:
```bash
uv run python -m table_unifier.server.main --host 0.0.0.0 --port 8000
```

### Endpoints

#### 1. `POST /sources/upload`
multipart/form-data. Принимает 1+ файлов `.xlsx` / `.csv` / `.parquet`.

Response:
```json
{
  "session_id": "sess_a1b2c3",
  "sources": [
    {
      "id": "src_001",
      "name": "auto_ru_2023_q4.xlsx",
      "rows": 14,
      "cols": ["sell_id", "mark", "model", "year", "mileage", "color", "bodyType", "engine", "transmission", "price"],
      "size_bytes": 18420,
      "sample": [[...первые 50 строк...]],
      "dtypes": {"sell_id": "string", "year": "int64", ...}
    }
  ]
}
```

После загрузки: пробегает по таблицам и считает schema-divergence для warning-callout (см. экран 1):
- какие колонки переименованы (через qwen3-embedding косинус на column-эмбеддингах)
- какие колонки имеют форматные расхождения (text vs hex, регистр, числа vs строки)

#### 2. `POST /graph/build`
Запускает построение HeteroData-графа.

Request:
```json
{
  "session_id": "sess_a1b2c3",
  "source_ids": ["src_001", "src_002"],
  "model_tag": "bge-m3",
  "idf_min_df": 2,
  "target_col_dim": 1024
}
```

Response:
```json
{ "run_id": "run_xyz", "status": "started" }
```

Запуск асинхронный. Прогресс — через WebSocket `/runs/{run_id}/stream` (см. ниже).

Под капотом — последовательность из `experiments/02_build_graphs.py` или `experiments/14_build_unified_graph_mrl.py`:
1. `TokenEmbedder(model_name="BAAI/bge-m3").embed_rows(df_a, df_b)` → row embeddings (1024-d, npy)
2. `OllamaClient().embed_columns(df_a.columns, df_b.columns)` → column embeddings (4096-d)
3. `build_hetero_data(...)` → `HeteroData` объект с фильтрацией по IDF
4. Сохранить в `data/graphs/<model_tag>/runs/<run_id>/`

#### 3. `POST /infer/run`
Запускает инференс (forward pass + similarity + connected components).

Request:
```json
{
  "run_id": "run_xyz",
  "checkpoint": "output/bge-m3/v17_views_gat_model.pt",
  "similarity_threshold": 0.831,
  "use_ga_tuning": true
}
```

Response: `{ "status": "started" }`. Прогресс — по тому же WebSocket-каналу `run_id`.

Под капотом:
1. Загрузить `EntityResolutionGAT` + state_dict
2. `model.eval()`, forward pass → row_embeddings
3. Pairwise cosine similarity (только cross-table пары: A × B)
4. Threshold + connected components: ребро есть, если sim ≥ thr; кластеры = компоненты связности (union-find)
5. (опционально) GA-оптимизация порога по validation pairs — `evaluation/ga_cc.py`
6. Сохранить результаты в `output/<model_tag>/runs/<run_id>/results.json`

#### 4. WebSocket `/runs/{run_id}/stream`
Стрим JSON-событий по мере работы pipeline.

Формат событий:
```json
{ "type": "phase",    "phase": "embed",   "label": "TokenEmbedder · bge-m3" }
{ "type": "progress", "phase": "embed",   "progress": 0.45 }
{ "type": "log",      "level": "info",    "msg": "computed row embeddings 12/14 · 1024-d" }
{ "type": "log",      "level": "ok",      "msg": "graph ready · 27 row · 41 token · 108 edges" }
{ "type": "phase",    "phase": "tokenize", "label": "Ollama qwen3-embedding · column" }
{ "type": "phase",    "phase": "l1",      "label": "GATv2Conv[0] · token→row" }
{ "type": "phase",    "phase": "l2",      "label": "GATv2Conv[1] · row→token" }
{ "type": "phase",    "phase": "sim",     "label": "cosine similarity matrix" }
{ "type": "phase",    "phase": "cluster", "label": "connected components + GA" }
{ "type": "metric",   "key": "F1",        "value": 0.913 }
{ "type": "done",     "result_url": "/runs/run_xyz/results" }
{ "type": "error",    "msg": "Ollama unreachable: connection refused" }
```

Фазы должны соответствовать дизайну (см. screen-training.jsx — теперь это инференс):
- `load` — загрузка чекпоинта
- `embed` — row embeddings (только при `/graph/build`)
- `tokenize` — column embeddings (только при `/graph/build`)
- `build` — HeteroData (только при `/graph/build`)
- `l1`, `l2` — forward pass по слоям (при `/infer/run`)
- `sim` — pairwise similarity
- `cluster` — threshold + connected components

#### 5. `GET /runs/{run_id}/graph`
Возвращает JSON графа для визуализации.

```json
{
  "rows": [
    { "id": "A0", "source": "A", "label": "BMW X5 2018", "cols": {...}, "x_init": 0.32, "y_init": 0.48 },
    ...
  ],
  "tokens": [
    { "id": "t_0", "text": "bmw", "df": 4, "x_init": 0.5, "y_init": 0.5 },
    ...
  ],
  "edges": [
    { "row": "A0", "token": "t_0", "col": "mark", "weight": 1.0 },
    ...
  ]
}
```

Координаты `x_init`, `y_init` — нормализованные [0,1], можно вычислять force-directed серверно или просто рандомно (фронт сделает свой layout).

#### 6. `GET /runs/{run_id}/embeddings`
2D-проекция row embeddings для embedding-space view.

```json
{
  "method": "umap",  // или "tsne"
  "points": [
    { "row_id": "A0", "x": 0.42, "y": 0.18, "cluster": "C-001" },
    ...
  ]
}
```

Серверу нужно вызвать `umap-learn` или `sklearn.manifold.TSNE` на финальных row embeddings (1024 → 2-d).

#### 7. `GET /runs/{run_id}/clusters`
Результаты после инференса.

```json
{
  "candidates": [
    {
      "id": "pair_001",
      "a": "A0",
      "b": "B0",
      "similarity": 0.94,
      "verdict": "auto",  // "auto" | "review" | "reject"
      "cluster_id": "C-001",
      "field_divergence": ["color", "bodyType"]
    },
    ...
  ],
  "clusters": [
    {
      "id": "C-001",
      "members": [{"source": "A", "row": 0}, {"source": "B", "row": 0}],
      "canonical": {"brand": "BMW", "model": "X5", "year": 2018, ...},
      "similarity": 0.94,
      "needs_review": false
    },
    ...
  ],
  "metrics": {
    "n_pairs_found": 9,
    "n_clusters": 18,
    "n_input_rows": 27,
    "f1": 0.913,
    "precision": 0.940,
    "recall": 0.890,
    "roc_auc": 0.957,
    "latency_ms": 312
  }
}
```

#### 8. `POST /runs/{run_id}/clusters/decisions`
Принимает решения пользователя из экрана ревью.

```json
{
  "decisions": [
    { "pair_id": "pair_009", "verdict": "approve" },
    { "pair_id": "pair_010", "verdict": "reject" }
  ]
}
```

Применяет к финальной таблице (отклонённые пары → раздельные кластеры; одобренные → слиты).

#### 9. `GET /runs/{run_id}/unified.{format}`
Экспорт финальной унифицированной таблицы. `format` ∈ {`xlsx`, `csv`, `parquet`, `json`}.

Логика merge для каждого кластера:
- `brand`, `model` — берётся mode (наиболее частое значение)
- `color` — предпочесть text-format (А) если в кластере есть и text и hex
- `bodyType` — нормализация к Title Case
- `mileage` — average по членам
- `price` — min (или average; settable)
- `engine` — нормализация к литрам с одним десятичным

Колонки выхода: те, что в исходных таблицах + `cluster_id`, `source_ids`, `n_members`, `confidence`.

#### 10. Optional: `POST /infer/single_pair`
Для интерактивного "что если" в экране ревью (когда юзер крутит slider threshold).

Принимает два row_id и возвращает {similarity, verdict_at_threshold}.

## State Management

### На сервере
- `runs: Dict[run_id, RunState]` — состояние каждого запуска
- `sessions: Dict[session_id, SessionState]` — загруженные источники
- `websocket_subscribers: Dict[run_id, List[WebSocket]]` — подписчики на стрим
- Артефакты на диске: `data/graphs/{model_tag}/runs/{run_id}/`, `output/{model_tag}/runs/{run_id}/`

### На клиенте (фронтенд)
Уже описано в `app.jsx`. Главные поля state:
- `step: number` — текущий экран (0..4)
- `completed: Set<number>` — пройденные шаги
- `sessionId: string` — после upload
- `runId: string` — после graph/build
- `reviewDecisions: Record<pairKey, 'approve'|'reject'>`

Нужно добавить вызовы fetch к API и WS-подписки. Сейчас всё мокается через `data.js` и setTimeout-анимации.

## Как фронт уже ходит в API (реализуй бэкенд под это)

Весь клиентский код уже написан — ничего переписывать на фронте не нужно. Достаточно поднять бэкенд с этими endpoint'ами. Поведение экранов:

1. **`api.js`** — единый клиент: `uploadFiles()`, `buildGraph()`, `getGraph()`, `getEmbeddings()`, `runInference()`, `subscribeRun(runId, onEvent, { kind })`, `getClusters()`, `postDecisions()`, `singlePair()`, `exportUrl()`. Состояние живёт в `window.__STATE__` (sessionStorage) и `window.__DATA__`.
2. **`screen-upload.jsx`**: на `onChange`/`onDrop` шлёт `POST /api/sources/upload`, сохраняет `session_id` и `sources`. Список файлов восстанавливается из `__DATA__.sources` (переживает релоад/возврат).
3. **`screen-graph.jsx`**: если `runId` ещё нет — `POST /api/graph/build`; затем WS `subscribeRun(runId, fn, { kind: 'build' })`. Ждёт фазы `embed`/`tokenize`/`build` и финальное событие **`graph_done`** (`{n_rows, n_tokens, n_edges}`), после чего тянет `GET /api/runs/{run_id}/graph`. **Если граф уже в памяти — сборка не перезапускается** (экран сразу в состоянии done).
4. **`screen-training.jsx`** (инференс): если инференс ещё не отработан — `POST /api/infer/run` + WS `subscribeRun(runId, fn, { kind: 'infer' })`. На `type=phase` с `l1`/`l2`/`sim`/`cluster` обновляет `pulseLayer`/`graphProgress`; на `type=done` тянет `GET /api/runs/{run_id}/clusters`. **Повторный вход на экран не перезапускает инференс** — явный перезапуск только кнопкой «↻ перезапустить».
5. **`screen-review.jsx`**: при mount — `GET /api/runs/{run_id}/clusters` (если кандидатов ещё нет). Решения ✓/✗ копятся локально, на переходе дальше — `POST /api/runs/{run_id}/clusters/decisions`.
6. **`screen-result.jsx`**: кнопки экспорта → `GET /api/runs/{run_id}/unified.{format}` (с `Content-Disposition: attachment`).

### Важно про WebSocket `?kind=`
Оба экрана (граф и инференс) подписываются на один канал `run_id`, но с разным query-параметром: `…/stream?kind=build` и `…/stream?kind=infer`. Бэкенд может использовать это, чтобы решить, какую фазу пайплайна реплеить. Событие сборки графа — `graph_done` (не `done`); событие конца инференса — `done`.

## Design Tokens

Все определены в `styles.css` (`:root`-переменные). Главное:

### Цвета (oklch)
- `--bg: oklch(0.155 0.005 250)` — основной фон
- `--bg-elev: oklch(0.185 0.006 250)` — поднятые поверхности
- `--surface: oklch(0.215 0.008 250)` — кнопки, поля
- `--border: oklch(0.295 0.012 250)` — стандартные границы
- `--text: oklch(0.96 0.004 250)` — основной текст
- `--text-3: oklch(0.58 0.010 250)` — приглушённый
- **Акценты** (общая хрома 0.15, общая lightness ~0.72):
  - `--row: oklch(0.72 0.15 248)` — синий, row nodes / Table A
  - `--token: oklch(0.72 0.15 295)` — фиолетовый, token nodes / Table B
  - `--cluster: oklch(0.78 0.15 152)` — зелёный, подтверждённый кластер
  - `--warn: oklch(0.80 0.15 75)` — амбер, требует ревью
  - `--bad: oklch(0.70 0.16 25)` — красный, отклонено

### Типографика
- **Inter** 400/500/600/700 — UI
- **JetBrains Mono** 400/500/600 — все числовые данные, ID, код

### Радиусы
- `--r-sm: 6px` · `--r-md: 10px` · `--r-lg: 14px` · `--r-xl: 20px`

### Анимации
- Pulse along edges: `requestAnimationFrame`, скорость ~0.75/сек
- Spinner: 0.8s linear infinite
- Transition: 0.12–0.15s ease для hover/state

## Screens / Views

### Экран 1 — Источники (`screen-upload.jsx`)

**Purpose**: загрузить .xlsx-файлы, показать превью.

**Layout**: 
- Grid `1fr 320px` сверху (dropzone + список файлов)
- Под ним — warning callout о schema-divergence (если ≥2 таблиц)
- Под ним — Grid `1fr 1fr` (два превью таблиц)

**Components**:
- **Dropzone**: `padding: 40px`, `border: 1.5px dashed var(--border-strong)`, состояние `dragging` → `border-color: var(--row)`, `background: var(--row-soft)`. Принимает `.xlsx,.csv,.parquet`.
- **File list**: `panel` с шапкой "Загружено". Каждая строка: цветной квадратик `A`/`B`, имя файла (моно), размер + статус.
- **Schema divergence callout**: жёлтая (амбер) иконка `!`, текст с конкретными расхождениями, чипы справа (`qwen3-emb · 4096d`, `auto-match`).
- **Table preview**: `.table-card` — заголовок с цветной точкой источника, мета (`{rows} rows · {cols} cols`), tbody — скролл с моно-шрифтом, цветной swatch для hex-значений.

### Экран 2 — Гетерогенный граф (`screen-graph.jsx`)

**Purpose**: построить и показать HeteroData-граф.

**Layout**: Grid `1fr 320px` — слева canvas, справа панель.

**Components**:
- **Canvas**: SVG full-bleed, фон radial-gradient к центру. Узлы — квадраты row + круги token. Рёбра — линии opacity 0.10–0.25. Hover row → подсветка связей + панель с деталями.
- **Phase HUD** (overlay top-left): чек-лист из 3 фаз (`TokenEmbedder`, `Ollama qwen3-embedding`, `HeteroData build`) с спиннером/чек-марком + прогресс-бар.
- **Selected node panel** (overlay top-right): детали выбранной строки + связанные токены как чипы.
- **Stats panel** (right): metrics (row=27, token=41, edges=108, col_dim=4096) + IDF slider + build log.

### Экран 3 — Инференс GNN (`screen-training.jsx`)

**Purpose**: применить обученную модель, показать message passing.

**Layout**: Grid вертикально — top-row `1fr` (граф+эмбеддинги+панель), bottom `auto` (histogram + metrics).

**Components**:
- **Tabs**: `Граф` / `Граф + Эмбеддинги` (split) / `Эмбеддинги`.
- **Граф**: `HeteroGraph` с `pulseLayer` (0=token→row, 1=row→token) и `pulsePhase` (0..1) для движения dots вдоль рёбер. По мере `graphProgress` (0→1) row-узлы сдвигаются от случайных к кластерным позициям.
- **Embedding space**: SVG 2D-скаттер с сеткой. Точки начинают рандомно, плавно сходятся в кластеры. Trail-линии (полупрозрачные) от начальной позиции к текущей.
- **Right panel**: 5 фаз pipeline (с чек-марками), архитектура (model, layers, dims), mini layer diagram (token ↔ row с подсвечивающейся стрелкой по `pulseLayer`), inference log.
- **Bottom**: гистограмма similarity (20 бинов от 0 до 1, threshold-line на 0.65), 4 финальные метрики.

### Экран 4 — Ревью пар (`screen-review.jsx`)

**Purpose**: ручная проверка кандидатов.

**Layout**: Grid `300px 1fr` — слева фильтры, справа список пар.

**Components**:
- **Filter list**: 4 кнопки с counts (Все / Авто / Требует ревью / Отклонено).
- **Similarity threshold slider**: 0.5–0.99, отметка `GA: 0.831`.
- **GA-tuned threshold box**: `clustering: connected components`, `metric: cosine`, `GA pop/gen`, `fitness: F1@thr`.
- **Pair card**: 
  - Заголовок: `A/1 ↔ B/1` + verdict chip + cluster_id chip + similarity + buttons `✓ объединить` / `✗ разделить`.
  - Body: Grid `1fr 32px 1fr` — два side-by-side списка полей с подсветкой расхождений (фон `var(--warn-soft)`), посередине вертикальная надпись `SIMILARITY 0.94`.

### Экран 5 — Результат (`screen-result.jsx`)

**Purpose**: финальная таблица + экспорт.

**Layout**: Grid `1fr 280px` — слева табл/sankey, справа сводка.

**Components**:
- **Tabs**: `Таблица` / `Sankey`.
- **UnifiedTable**: моноширинная таблица. Cluster-row кликабельна, разворачивает member-rows (`↳ A/1`). Цветной swatch для color.
- **SankeyView**: 3 колонки прямоугольников (sources / clusters / unified) с bezier-flows между ними, толщина ∝ количеству записей.
- **Right summary**: 4 metric-карточки (in/out/merged/singletons), список конфликтов полей со стратегией merge, финальные метрики, success callout.
- **Export buttons** в header: `.xlsx`, `.parquet`, `.csv`.

## Interactions & Behavior

### Глобальная навигация
- Sidebar слева с 5 пронумерованными шагами. Клик переходит на шаг (только если он уже пройден или текущий).
- Бэк-кнопка в footer `← Назад`. Primary-кнопка `Продолжить →`.

### Анимации
- **Pulse сообщений вдоль рёбер**: каждое ребро рисует маленький dot, движущийся от token к row (layer 1) или обратно (layer 2). Скорость ~0.75/сек, направление зависит от `pulseLayer`. Stagger по индексу ребра.
- **Сходимость графа**: интерполяция `lerp` между двумя precomputed layouts (initial random + final clustered) с коэффициентом `progress`.
- **Cluster halos**: при `progress > 0.6` появляются полупрозрачные зелёные круги вокруг центроидов кластеров.
- **Loss curve**: SVG path обновляется каждый "epoch" → в реальной версии — каждое WS-событие с метриками.

### Файловая загрузка
- Drag&drop → `e.preventDefault()` + `setDragging`.
- Клик по dropzone → инпут.
- Поддерживаемые форматы: `.xlsx`, `.csv`, `.parquet`.

### Ошибки
- WS disconnect → переподключаться 3 раза с backoff, после — показать toast "Соединение с сервером потеряно".
- Upload error → красный chip над dropzone с сообщением.
- Ollama unreachable → сообщение в build log + блокировка кнопки "Продолжить".

## Файлы дизайна

Все файлы фронта лежат в корне пакета (раньше была папка `design/` — удалена, это был устаревший дубль):

- `index.html` — боевой shell (грузит `api.js` + экраны; порядок загрузки важен!)
- `api.js` — **реальный API-клиент** (REST + WebSocket). Это то, под что пишется бэкенд.
- `styles.css` — все design tokens + классы (тёмная + светлая темы)
- `ui.jsx` — Sidebar, Topbar, StatusBar, Tabs, ScreenFooter, ColorSwatch, тема/масштаб
- `graph-viz.jsx` — `HeteroGraph` компонент + layout helpers
- `embedding-viz.jsx` — `EmbeddingSpace`, `LossCurve`
- `screen-upload.jsx` — экран 1
- `screen-graph.jsx` — экран 2
- `screen-training.jsx` — экран 3 (несмотря на имя — это **инференс**, не обучение)
- `screen-review.jsx` — экран 4
- `screen-result.jsx` — экран 5
- `app.jsx` — top-level App + роутинг между шагами
- `_preview/` — **только для превью без бэкенда**: `_preview.html` + `mock-api.js` (имитирует `window.API`). В production не подключать.

## Чеклист реализации

- [ ] Создать `src/table_unifier/server/` с FastAPI + uvicorn
- [ ] `POST /sources/upload` — парсинг xlsx/csv/parquet + schema-divergence detection
- [ ] `POST /graph/build` + WS-стрим прогресса (использует `dataset.embedding_generation`, `dataset.graph_builder`)
- [ ] `POST /infer/run` + WS-стрим (загружает чекпоинт `output/bge-m3/v17_views_gat_model.pt`, forward pass)
- [ ] Threshold + connected components (union-find) + опционально GA-tuning через `evaluation/ga_cc.py`
- [ ] `GET /runs/{run_id}/graph` — JSON графа для viz
- [ ] `GET /runs/{run_id}/embeddings` — UMAP 1024→2 проекция
- [ ] `GET /runs/{run_id}/clusters` — результаты + метрики
- [ ] `POST /clusters/decisions` — пользовательские approve/reject
- [ ] `GET /unified.{format}` — экспорт с конфликт-резолюшеном
- [x] ~~Заменить `data.js` на `api.js` во фронте~~ — уже сделано (`api.js`)
- [x] ~~Перевести все 5 экранов на реальные fetch + WS~~ — уже сделано (экраны вызывают `window.API`)
- [ ] **Все endpoint'ы под префиксом `/api`** (клиент ходит в `/api/sources/upload`, `/api/graph/build`, `/api/infer/run`, `/api/runs/{id}/…`, WS `/api/ws/runs/{id}/stream?kind=build|infer`)
- [ ] Сервер раздаёт фронт как статику (`StaticFiles` mount на `/`) — берём корневые файлы (без `_preview/`)
- [ ] Документировать запуск в `README.md` (uvicorn + Ollama хост)
- [ ] Опционально: Dockerfile с CUDA-base image

## Запуск

```bash
# install
uv sync

# Backend + frontend (на одном порту)
uv run python -m table_unifier.server.main --host 0.0.0.0 --port 8000

# Открыть http://localhost:8000/
```

Перед запуском убедиться:
- Ollama доступна по адресу из `OllamaConfig.host` (nvidia-server по дефолту)
- Чекпоинт лежит в `output/bge-m3/v17_views_gat_model.pt`
- CUDA доступна (или сервер падает в CPU fallback с warning)

## Дополнительные заметки

- **Не реализовывать обучение** — этот фронт только для инференса. Кнопка "перезапустить инференс" в экране 3 — да; кнопка "переобучить" — нет.
- **Loss — только NT-Xent (InfoNCE).** Из `models/losses.py` используется только `NTXentLoss`. Чекпоинт `output/bge-m3/v17_views_gat_model.pt` обучен через `train_entity_resolution_minibatch` с NT-Xent. **Не использовать** `train_entity_resolution_bce` и не предлагать BCE как опцию в UI — этот вариант признан неудачным.
- **Не использовать HDBSCAN.** Кластеризация — только threshold + connected components (см. `evaluation/clustering.py` + опционально `evaluation/ga_cc.py`). HDBSCAN-эксперимент (exp 18) признан неудачным и из приложения исключён.
- **Multi-tenant не нужен** — это инструмент для одного дипломника.
- **Аутентификация не нужна** для MVP.
- **Размер таблиц** — рассчитывать на 100–10 000 строк на каждый источник. Бóльшие — стримить + батчевать.
- **MLflow** не подключать — это для экспериментов, не для приложения.
