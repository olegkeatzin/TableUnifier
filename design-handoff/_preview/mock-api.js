// PREVIEW-ONLY mock backend. NOT shipped. Lets the live frontend render
// without FastAPI by synthesizing __DATA__.graph (with edge weights),
// __STATE__, and a fake API whose subscribeRun drives the inference phases.

(function () {
  // ---- source rows (compact, from design/data.js) ----
  const A = [
    ['BMW','X5',2018,'белый','внедорожник'],
    ['Toyota','Camry',2020,'чёрный','седан'],
    ['Mercedes','E-Class',2019,'серебристый','седан'],
    ['Audi','A6',2021,'серый','седан'],
    ['Volkswagen','Tiguan',2017,'красный','внедорожник'],
    ['Kia','Rio',2019,'синий','седан'],
    ['Hyundai','Solaris',2018,'белый','седан'],
    ['Lada','Vesta',2020,'тёмно-синий','седан'],
    ['Skoda','Octavia',2019,'белый','лифтбек'],
    ['Renault','Logan',2017,'серебристый','седан'],
    ['BMW','X5',2020,'чёрный','внедорожник'],
    ['Toyota','RAV4',2019,'белый','внедорожник'],
    ['Mazda','CX-5',2018,'красный','внедорожник'],
    ['Mitsubishi','Outlander',2017,'серый','внедорожник'],
  ];
  const B = [
    ['BMW','X5',2018,'белый','внедорожник'],
    ['Toyota','Camry',2020,'чёрный','седан'],
    ['Mercedes','E200',2019,'серебристый','седан'],
    ['Audi','A6',2021,'серый','седан'],
    ['Hyundai','Solaris',2018,'белый','седан'],
    ['Skoda','Octavia',2019,'белый','лифтбек'],
    ['Toyota','RAV4',2019,'белый','внедорожник'],
    ['Mazda','CX-5',2018,'красный','внедорожник'],
    ['Ford','Focus',2016,'синий','седан'],
    ['Nissan','Qashqai',2019,'белый','внедорожник'],
    ['BMW','X3',2019,'чёрный','внедорожник'],
    ['BMW','X5',2022,'синий','внедорожник'],
    ['Lada','Vesta',2020,'синий','седан'],
  ];
  const COLS = ['mark', 'model', 'year', 'color', 'bodyType'];

  const rows = [];
  function addRows(arr, src) {
    arr.forEach((r, i) => {
      const cols = {};
      COLS.forEach((c, k) => (cols[c] = r[k]));
      rows.push({ id: `${src}${i}`, source: src, label: `${r[0]} ${r[1]} ${r[2]}`, cols });
    });
  }
  addRows(A, 'A'); addRows(B, 'B');

  // ---- tokenize -> tokens + edges ----
  const tokMap = {};   // text -> {id, df}
  const edgesRaw = []; // {row, token, col}
  rows.forEach((row) => {
    COLS.forEach((col) => {
      const val = String(row.cols[col]).toLowerCase();
      val.split(/[\s\-]+/).filter(Boolean).forEach((w) => {
        if (!tokMap[w]) tokMap[w] = { id: `t_${Object.keys(tokMap).length}`, text: w, df: 0 };
        tokMap[w].df += 1;
        edgesRaw.push({ row: row.id, token: tokMap[w].id, col });
      });
    });
  });
  const tokens = Object.values(tokMap);
  const N = rows.length;
  // idf weight per edge, normalized to [0.08, 1]
  const byId = {}; tokens.forEach((t) => (byId[t.id] = t));
  let maxIdf = 0;
  edgesRaw.forEach((e) => {
    const df = byId[e.token].df;
    e._idf = Math.log((N + 1) / (df + 0.5));
    if (e._idf > maxIdf) maxIdf = e._idf;
  });
  const edges = edgesRaw.map((e) => ({
    row: e.row, token: e.token, col: e.col,
    weight: +(0.08 + 0.92 * (e._idf / maxIdf)).toFixed(3),
  }));

  // ---- clusters ----
  const dup = [[0,0],[1,1],[2,2],[3,3],[6,4],[8,5],[11,6],[12,7],[7,12]];
  const clusterByRow = {};
  dup.forEach(([a, b], i) => {
    const cid = `C-${String(i + 1).padStart(3, '0')}`;
    clusterByRow[`A${a}`] = cid;
    clusterByRow[`B${b}`] = cid;
  });

  window.__DATA__ = {
    tableA: { id: 'A', name: 'auto_ru_2023_q4.xlsx', rows: A.length, cols: COLS, data: A },
    tableB: { id: 'B', name: 'auto_ru_2024_q1.xlsx', rows: B.length, cols: COLS, data: B },
    clusters: [], candidates: [], metrics: {}, divergence: [],
    sources: [
      { id: 'src_001', name: 'auto_ru_2023_q4.xlsx', rows: A.length, cols: COLS, size_bytes: 18420 },
      { id: 'src_002', name: 'auto_ru_2024_q1.xlsx', rows: B.length, cols: COLS, size_bytes: 16380 },
    ],
    histogram: [0,0,0,0,0,1,2,1,0,0,1,2,1,0,3,5,7,9,6,2],
    graph: { rows, tokens, edges, stats: { n_rows: N, n_tokens: tokens.length, n_edges: edges.length, col_dim: 4096 }, clusterByRow: null },
  };

  // start without clusters; reveal them on "done"
  window.__STATE__ = { sessionId: 'sess_mock', runId: 'run_mock', inferRunId: null, inferDone: false, graphReady: true };
  window.__STATE_RESET__ = () => {};

  // ---- fake API ----
  function emit(onEvent, ev, ms) { setTimeout(() => onEvent(ev), ms); }
  window.API = {
    uploadFiles: async () => ({}),
    buildGraph: async () => ({ run_id: 'run_mock' }),
    getGraph: async () => { window.dispatchEvent(new CustomEvent('graph-updated')); return window.__DATA__.graph; },
    getEmbeddings: async () => ({ points: [] }),
    runInference: async () => ({ status: 'started' }),
    getClusters: async () => {
      window.__DATA__.graph.clusterByRow = clusterByRow;
      // Candidate pairs for the review screen — a deliberately varied mix so the
      // interface can be evaluated: clean auto-matches, manual-review cases with
      // real field divergence, and false-positive rejects (high token overlap,
      // different listing).
      window.__DATA__.candidates = [
        // — clean duplicates: identical across both dumps → auto-resolved —
        { id: 'p01', a: ['A', 0],  b: ['B', 0],  sim: 0.997, verdict: 'auto',   cluster: 'C-001', divergence: [] },
        { id: 'p02', a: ['A', 1],  b: ['B', 1],  sim: 0.994, verdict: 'auto',   cluster: 'C-002', divergence: [] },
        { id: 'p03', a: ['A', 3],  b: ['B', 3],  sim: 0.991, verdict: 'auto',   cluster: 'C-003', divergence: [] },
        { id: 'p04', a: ['A', 6],  b: ['B', 4],  sim: 0.989, verdict: 'auto',   cluster: 'C-004', divergence: [] },
        { id: 'p05', a: ['A', 8],  b: ['B', 5],  sim: 0.992, verdict: 'auto',   cluster: 'C-005', divergence: [] },
        { id: 'p06', a: ['A', 11], b: ['B', 6],  sim: 0.995, verdict: 'auto',   cluster: 'C-006', divergence: [] },
        { id: 'p07', a: ['A', 12], b: ['B', 7],  sim: 0.988, verdict: 'auto',   cluster: 'C-007', divergence: [] },
        // — true duplicates with formatting divergence → manual review —
        { id: 'p08', a: ['A', 2],  b: ['B', 2],  sim: 0.864, verdict: 'review', cluster: 'C-008', divergence: ['model'] },  // E-Class ↔ E200
        { id: 'p09', a: ['A', 7],  b: ['B', 12], sim: 0.882, verdict: 'review', cluster: 'C-009', divergence: ['color'] },  // тёмно-синий ↔ синий
        // — ambiguous: same model, different listing → review (likely split) —
        { id: 'p10', a: ['A', 0],  b: ['B', 11], sim: 0.812, verdict: 'review', cluster: null,    divergence: ['year', 'color'] }, // X5 2018 белый ↔ X5 2022 синий
        // — false positive: similar tokens, different model → reject —
        { id: 'p11', a: ['A', 10], b: ['B', 10], sim: 0.778, verdict: 'reject', cluster: null,    divergence: ['model', 'year'] }, // X5 ↔ X3
      ];
      window.dispatchEvent(new CustomEvent('graph-updated'));
      // Final clusters: 9 merged duplicate pairs + singletons for every
      // unmatched row → 18 clusters total (matches metrics.n_clusters). The
      // result screen reads __DATA__.clusters; without this it renders empty.
      const simByPair = {};
      window.__DATA__.candidates.forEach((c) => { simByPair[`A${c.a[1]}-B${c.b[1]}`] = c.sim; });
      const mergedA = new Set(), mergedB = new Set();
      const clusters = dup.map(([a, b], i) => {
        mergedA.add(a); mergedB.add(b);
        const sim = simByPair[`A${a}-B${b}`] ?? 0.99;
        return { id: `C-${String(i + 1).padStart(3, '0')}`,
                 members: [{ source: 'A', row: a }, { source: 'B', row: b }],
                 similarity: sim, needs_review: sim < 0.9 };
      });
      let n = dup.length;
      A.forEach((_, i) => { if (!mergedA.has(i)) { n++; clusters.push({ id: `C-${String(n).padStart(3, '0')}`, members: [{ source: 'A', row: i }], similarity: 1, needs_review: false }); } });
      B.forEach((_, i) => { if (!mergedB.has(i)) { n++; clusters.push({ id: `C-${String(n).padStart(3, '0')}`, members: [{ source: 'B', row: i }], similarity: 1, needs_review: false }); } });
      window.__DATA__.clusters = clusters;
      const metrics = { n_pairs_found: 11, n_clusters: 18, n_input_rows: 27, latency_ms: 312, threshold: 0.831, f1: 0.913,
        t_col_descriptions_ms: 1840, t_col_embeddings_ms: 760, t_row_embeddings_ms: 2130, t_gat_ms: 312, t_total_ms: 5042,
        graph_mem_mb: 18.4, graph_bytes: 19293798 };
      window.__DATA__.metrics = metrics;
      return { metrics };
    },
    subscribeRun: (runId, onEvent, opts = {}) => {
      const kind = opts.kind || 'infer';
      let t = 200;
      const step = (ev, dt) => { emit(onEvent, ev, t); t += dt; };

      if (kind === 'build') {
        // Graph-build stream — phases screen-graph.jsx expects, ending in graph_done.
        const st = (window.__DATA__.graph && window.__DATA__.graph.stats) || { n_rows: 27, n_tokens: 41, n_edges: 108 };
        step({ type: 'phase', phase: 'embed', label: 'TokenEmbedder · bge-m3' }, 250);
        for (let p = 0.25; p <= 1.0001; p += 0.25) step({ type: 'progress', phase: 'embed', progress: Math.min(1, p) }, 160);
        step({ type: 'log', level: 'info', msg: `computed row embeddings ${st.n_rows}/${st.n_rows} · 1024-d` }, 150);
        step({ type: 'phase', phase: 'tokenize', label: 'Ollama qwen3-embedding · столбцы' }, 250);
        for (let p = 0.34; p <= 1.0001; p += 0.33) step({ type: 'progress', phase: 'tokenize', progress: Math.min(1, p) }, 170);
        step({ type: 'phase', phase: 'build', label: 'HeteroData (строки + токены + рёбра)' }, 250);
        for (let p = 0.5; p <= 1.0001; p += 0.5) step({ type: 'progress', phase: 'build', progress: Math.min(1, p) }, 180);
        step({ type: 'graph_done', n_rows: st.n_rows, n_tokens: st.n_tokens, n_edges: st.n_edges }, 200);
        return { close: () => {} };
      }

      // Inference stream.
      step({ type: 'phase', phase: 'load', label: 'load checkpoint · v17_views_gat' }, 700);
      step({ type: 'log', level: 'info', msg: 'state_dict loaded · 2×GATv2Conv' }, 300);
      step({ type: 'phase', phase: 'l1', label: 'GATv2Conv[0] · token→row' }, 200);
      for (let p = 0.2; p <= 1; p += 0.4) step({ type: 'progress', phase: 'l1', progress: p }, 250);
      step({ type: 'phase', phase: 'l2', label: 'GATv2Conv[1] · row→token' }, 200);
      for (let p = 0.2; p <= 1; p += 0.4) step({ type: 'progress', phase: 'l2', progress: p }, 250);
      step({ type: 'phase', phase: 'sim', label: 'cosine similarity matrix' }, 600);
      step({ type: 'phase', phase: 'cluster', label: 'connected components + GA' }, 600);
      step({ type: 'metric', key: 'threshold', value: 0.831 }, 100);
      step({ type: 'log', level: 'ok', msg: '9 pairs · 18 clusters · F1=0.913' }, 200);
      step({ type: 'done', result_url: '/runs/run_mock/results' }, 200);
      return { close: () => {} };
    },
    postDecisions: async () => ({}),
    singlePair: async () => ({ similarity: 0.8 }),
    exportUrl: (rid, fmt) => `#${fmt}`,
  };

  // land on the inference screen
  try {
    sessionStorage.setItem('tableunifier:app:v1', JSON.stringify({ step: 2, completed: [0, 1], reviewDecisions: {} }));
  } catch (_e) {}
})();
