// API-клиент для TableUnifier backend.
//
// Заменяет ранее использовавшийся data.js (моки). Экспортирует:
//   window.API                — REST/WS вызовы
//   window.__STATE__          — session_id / run_id / status флаги (mutable)
//   window.__DATA__           — текущие данные (tableA, tableB, clusters,
//                                candidates, divergence). Заполняется по мере
//                                ответов API. Тот же формат, что был в моке,
//                                чтобы существующие экраны не переписывать.
//
// Все ошибки бэка показываются через console.error + событие `apierror` на
// window — экран может слушать и показать toast.

(function () {
  const BASE = ""; // same-origin

  // ---- состояние ----------------------------------------------------------
  // sessionStorage переживает F5 в той же вкладке; новый таб → чистая сессия.
  const STATE_KEY = "tableunifier:state:v1";
  const _defaults = {
    sessionId: null, runId: null,
    graphReady: false, inferDone: false, inferRunId: null,
  };
  let _saved = {};
  try { _saved = JSON.parse(sessionStorage.getItem(STATE_KEY) || "{}"); }
  catch (_e) { _saved = {}; }

  window.__STATE__ = new Proxy({ ..._defaults, ..._saved }, {
    set(t, k, v) {
      t[k] = v;
      try { sessionStorage.setItem(STATE_KEY, JSON.stringify(t)); }
      catch (_e) { /* quota / private mode — игнорируем */ }
      return true;
    },
  });
  // Утилита для полного сброса (вызывается из app.jsx reset()).
  window.__STATE_RESET__ = () => {
    for (const k of Object.keys(_defaults)) window.__STATE__[k] = _defaults[k];
    try { sessionStorage.removeItem(STATE_KEY); } catch (_e) { /* noop */ }
  };

  // плейсхолдер-таблицы пока ничего не загружено
  window.__DATA__ = {
    tableA: null,
    tableB: null,
    clusters: [],
    candidates: [],
    metrics: {},
    divergence: [],
    sources: [],
    histogram: new Array(20).fill(0),
  };

  function emitError(msg) {
    console.error("[api]", msg);
    window.dispatchEvent(new CustomEvent("apierror", { detail: msg }));
  }

  async function _json(url, opts = {}) {
    const r = await fetch(BASE + url, opts);
    if (!r.ok) {
      const text = await r.text();
      emitError(`${url} → ${r.status}: ${text}`);
      throw new Error(text || `HTTP ${r.status}`);
    }
    return r.json();
  }

  // ---- Sources ------------------------------------------------------------

  async function uploadFiles(files) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    if (window.__STATE__.sessionId) fd.append("session_id", window.__STATE__.sessionId);
    const res = await _json("/api/sources/upload", { method: "POST", body: fd });
    window.__STATE__.sessionId = res.session_id;
    window.__DATA__.sources = res.sources;
    window.__DATA__.divergence = res.divergence || [];
    // populate tableA / tableB из первых двух source'ов (формат как у моков)
    const [a, b] = res.sources;
    if (a) window.__DATA__.tableA = sourceToTable(a, "A");
    if (b) window.__DATA__.tableB = sourceToTable(b, "B");
    return res;
  }

  function sourceToTable(src, sideId) {
    return {
      id: sideId,
      source_id: src.id,
      name: src.name,
      rows: src.rows,
      cols: src.cols,
      data: (src.sample || []).map((row) => {
        // sample приходит как dict-of-cells через pandas? нет — мы делаем .values, то есть массив
        if (Array.isArray(row)) {
          // Пропустим колонку 'id' если она есть первой
          // (build_graph требует id, но в превью id может мешать)
          return row;
        }
        // если вдруг dict — собираем в массив по cols
        return src.cols.map((c) => row[c]);
      }),
    };
  }

  // ---- Graph build --------------------------------------------------------

  async function buildGraph({ sessionId, sourceIds, modelTag = "bge-m3",
                              idfMinDf = 2, maxTokenDf = 0.3,
                              targetColDim = 1024 } = {}) {
    const body = {
      session_id: sessionId || window.__STATE__.sessionId,
      source_ids: sourceIds || (window.__DATA__.sources || []).map((s) => s.id),
      model_tag: modelTag, idf_min_df: idfMinDf,
      max_token_df: maxTokenDf, target_col_dim: targetColDim,
    };
    const res = await _json("/api/graph/build", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    window.__STATE__.runId = res.run_id;
    return res;
  }

  async function getGraph(runId) {
    const res = await _json(`/api/runs/${runId}/graph`);
    window.__DATA__.graph = res;
    window.dispatchEvent(new CustomEvent("graph-updated"));
    return res;
  }

  async function getEmbeddings(runId) {
    return _json(`/api/runs/${runId}/embeddings`);
  }

  // ---- Inference ----------------------------------------------------------

  async function runInference({ runId, checkpoint, threshold = 0.831,
                                useGa = false } = {}) {
    const body = {
      run_id: runId || window.__STATE__.runId,
      checkpoint: checkpoint || "output/bge-m3/v14_mrl_gat_model.pt",
      similarity_threshold: threshold,
      use_ga_tuning: useGa,
    };
    return _json("/api/infer/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  // WebSocket подписка. Возвращает { close }. opts.kind ('build'|'infer')
  // помечает, какую стадию пайплайна слушаем — бэк реплеит её буфер.
  function subscribeRun(runId, onEvent, opts = {}) {
    let ws;
    let attempt = 0;
    let closed = false;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const kindQ = opts.kind ? `?kind=${encodeURIComponent(opts.kind)}` : "";
    const url = `${proto}//${location.host}/api/ws/runs/${runId}/stream${kindQ}`;

    function connect() {
      ws = new WebSocket(url);
      ws.onmessage = (e) => {
        try { onEvent(JSON.parse(e.data)); }
        catch (err) { console.error("ws parse", err); }
      };
      ws.onclose = () => {
        if (closed) return;
        if (attempt < 3) {
          attempt++;
          setTimeout(connect, 300 * attempt);
        } else {
          emitError("WebSocket disconnected after 3 retries");
        }
      };
      ws.onerror = () => { /* пускай onclose доделает retry */ };
    }
    connect();
    return { close: () => { closed = true; if (ws) ws.close(); } };
  }

  // ---- Clusters / decisions ----------------------------------------------

  async function getClusters(runId) {
    const res = await _json(`/api/runs/${runId}/clusters`);
    window.__DATA__.clusters = res.clusters || [];
    window.__DATA__.candidates = (res.candidates || []).map((c) => ({
      // удобный формат для PairCard: a=['A',i], b=['B',j]
      id: c.id,
      a: ["A", c.a_idx],
      b: ["B", c.b_idx],
      sim: c.similarity,
      verdict: c.verdict,
      cluster: c.cluster_id,
      divergence: c.field_divergence,
    }));
    window.__DATA__.metrics = res.metrics || {};
    window.__DATA__.histogram = res.histogram || new Array(20).fill(0);
    // Обновим таблицы из payload (полные, не только sample).
    if (res.table_a) {
      window.__DATA__.tableA = {
        id: "A", name: res.table_a.name, rows: res.table_a.rows,
        cols: res.table_a.cols, data: res.table_a.data,
      };
    }
    if (res.table_b) {
      window.__DATA__.tableB = {
        id: "B", name: res.table_b.name, rows: res.table_b.rows,
        cols: res.table_b.cols, data: res.table_b.data,
      };
    }
    // После инференса перезапросим граф: рёбра теперь несут attention-веса
    // GAT (толщина/прозрачность). До этого вес был не определён.
    let g = window.__DATA__.graph;
    if (g) {
      try { await getGraph(runId); } catch (e) { /* граф мог истечь — не критично */ }
      g = window.__DATA__.graph;
    }
    // Прокинем cluster_id в graph (для clustered-layout в HeteroGraph).
    if (g) {
      const byRow = {};
      for (const c of (res.candidates || [])) {
        if (!c.cluster_id) continue;
        byRow[`A${c.a_idx}`] = c.cluster_id;
        byRow[`B${c.b_idx}`] = c.cluster_id;
      }
      // Кластеры из clusters[] тоже добавим (на случай рядов без cand-пар).
      for (const cl of (res.clusters || [])) {
        for (const m of (cl.members || [])) {
          byRow[`${m.source}${m.row}`] = cl.id;
        }
      }
      g.clusterByRow = byRow;
      window.dispatchEvent(new CustomEvent("graph-updated"));
    }
    return res;
  }

  async function postDecisions(runId, decisions) {
    const list = Object.entries(decisions).map(([pair_id, verdict]) => ({
      pair_id, verdict,
    }));
    return _json(`/api/runs/${runId}/clusters/decisions`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions: list }),
    });
  }

  async function singlePair(runId, a, b, threshold) {
    return _json("/api/infer/single_pair", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, a, b, threshold }),
    });
  }

  // ---- Export -------------------------------------------------------------

  function exportUrl(runId, format) {
    return `/api/runs/${runId}/unified.${format}`;
  }

  window.API = {
    uploadFiles, buildGraph, getGraph, getEmbeddings,
    runInference, subscribeRun, getClusters, postDecisions,
    singlePair, exportUrl,
  };
})();
