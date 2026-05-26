// Гетерографическая визуализация — fully data-driven.
//
// Источник данных: window.__DATA__.graph = {
//   rows:    [{ id, source: 'A'|'B', label, cols: {col_name: value, ...} }],
//   tokens:  [{ id, text, df }],
//   edges:   [{ row, token, col, weight }],
//   stats:   { n_rows, n_tokens, n_edges, col_dim },
//   clusterByRow: { rowId -> clusterId }  // populated после инференса
// }
//
// Заполняется в screen-graph.jsx через window.API.getGraph(runId).

// ---- утилиты ------------------------------------------------------------
function makeSeededRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
}

function lerp(a, b, t) { return a + (b - a) * t; }


// Лейаут: A слева, B справа, токены — в центре сеткой,
// упорядоченные по среднему вертикальному положению связанных строк
// (даёт читаемый bipartite-like рисунок).
function buildInitialLayout(W, H, graph) {
  const rng = makeSeededRng(42);
  const rows = {}, tokens = {};
  const aRows = graph.rows.filter((r) => r.source === 'A');
  const bRows = graph.rows.filter((r) => r.source === 'B');

  aRows.forEach((r, i) => {
    const t = aRows.length > 1 ? i / (aRows.length - 1) : 0.5;
    rows[r.id] = {
      x: W * 0.12 + (rng() - 0.5) * 12,
      y: H * 0.08 + t * H * 0.84 + (rng() - 0.5) * 8,
    };
  });
  bRows.forEach((r, i) => {
    const t = bRows.length > 1 ? i / (bRows.length - 1) : 0.5;
    rows[r.id] = {
      x: W * 0.88 + (rng() - 0.5) * 12,
      y: H * 0.08 + t * H * 0.84 + (rng() - 0.5) * 8,
    };
  });

  // Для токена считаем средний y соседей -> упорядочиваем по нему.
  const tokenNeighborY = {};
  graph.edges.forEach((e) => {
    const r = rows[e.row];
    if (!r) return;
    if (!tokenNeighborY[e.token]) tokenNeighborY[e.token] = [];
    tokenNeighborY[e.token].push(r.y);
  });

  const tokenList = graph.tokens.slice();
  // токены без соседей — в конец
  tokenList.sort((a, b) => {
    const ya = tokenNeighborY[a.id];
    const yb = tokenNeighborY[b.id];
    const ma = ya ? ya.reduce((s, v) => s + v, 0) / ya.length : H * 1e6;
    const mb = yb ? yb.reduce((s, v) => s + v, 0) / yb.length : H * 1e6;
    return ma - mb;
  });

  const N = tokenList.length;
  const cols = Math.min(6, Math.max(3, Math.ceil(Math.sqrt(N))));
  const totalRows = Math.ceil(N / cols);
  tokenList.forEach((t, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    tokens[t.id] = {
      x: W * 0.32 + (cols > 1 ? (col / (cols - 1)) : 0.5) * W * 0.36 + (rng() - 0.5) * 12,
      y: H * 0.10 + (totalRows > 1 ? (row / (totalRows - 1)) : 0.5) * H * 0.80 + (rng() - 0.5) * 10,
    };
  });
  return { rows, tokens };
}


// Лейаут «после инференса»: группируем строки по cluster_id.
// Для рядов без кластера — оставляем рядом с initial-позицией.
function buildClusteredLayout(W, H, graph, L0) {
  const rng = makeSeededRng(17);
  const rows = {}, tokens = {};

  const clusterByRow = graph.clusterByRow || {};
  const clusterMembers = {};
  for (const r of graph.rows) {
    const cid = clusterByRow[r.id];
    if (!cid) continue;
    (clusterMembers[cid] = clusterMembers[cid] || []).push(r.id);
  }

  const clusterIds = Object.keys(clusterMembers);
  const cols = Math.max(4, Math.ceil(Math.sqrt(clusterIds.length)));
  const rowsCount = Math.max(1, Math.ceil(clusterIds.length / cols));
  const clusterCenter = {};
  clusterIds.forEach((cid, i) => {
    const cx = W * (0.12 + (cols > 1 ? (i % cols) / (cols - 1) : 0.5) * 0.76);
    const cy = H * (0.18 + (rowsCount > 1 ? Math.floor(i / cols) / (rowsCount - 1) : 0.5) * 0.64);
    clusterCenter[cid] = { x: cx, y: cy };
  });

  // Кластерные строки — по окружности вокруг центра.
  for (const cid of clusterIds) {
    const members = clusterMembers[cid];
    const { x: cx, y: cy } = clusterCenter[cid];
    members.forEach((rid, j) => {
      const angle = (j / members.length) * Math.PI * 2 + rng() * 0.4;
      const r = members.length > 1 ? 18 : 0;
      rows[rid] = { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
    });
  }

  // Внеклассерные строки — оставляем на исходной позиции (чуть подтянуты к центру).
  for (const r of graph.rows) {
    if (rows[r.id]) continue;
    const init = L0.rows[r.id] || { x: W / 2, y: H / 2 };
    rows[r.id] = { x: lerp(init.x, W * 0.5, 0.12), y: init.y };
  }

  // Токены — почти на месте, мягко тянем к центру.
  for (const t of graph.tokens) {
    const init = L0.tokens[t.id] || { x: W / 2, y: H / 2 };
    tokens[t.id] = { x: lerp(init.x, W * 0.5, 0.12), y: lerp(init.y, H * 0.5, 0.12) };
  }
  return { rows, tokens };
}

function interpLayout(L0, L1, t) {
  const rows = {}, tokens = {};
  for (const k in L0.rows) {
    const a = L0.rows[k], b = L1.rows[k] || a;
    rows[k] = { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
  }
  for (const k in L0.tokens) {
    const a = L0.tokens[k], b = L1.tokens[k] || a;
    tokens[k] = { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
  }
  return { rows, tokens };
}


// ---- основной компонент -------------------------------------------------
function HeteroGraph({
  progress = 0,
  pulseLayer = null,
  pulsePhase = 0,
  showTokens = true,
  showEdges = true,
  selected = null,
  onSelectRow = null,
  highlightClusters = false,
}) {
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [hoverTip, setHoverTip] = useState(null);
  // tick — заставляет компонент перерендериться, когда __DATA__.graph
  // меняется снаружи (его кладёт screen-graph.jsx после API.getGraph).
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setSize({ w: r.width, h: r.height });
    return () => ro.disconnect();
  }, []);

  // Подписка на «новые данные графа» — кастомное событие из api.js.
  useEffect(() => {
    const h = () => setTick((t) => t + 1);
    window.addEventListener('graph-updated', h);
    return () => window.removeEventListener('graph-updated', h);
  }, []);

  const graph = window.__DATA__.graph;
  const haveGraph = graph && graph.rows && graph.rows.length > 0;

  const { layout, edgesByRow, edgesByToken } = useMemo(() => {
    if (!haveGraph) return { layout: { rows: {}, tokens: {} }, edgesByRow: {}, edgesByToken: {} };
    const W = size.w, H = size.h;
    if (W < 50 || H < 50) return { layout: { rows: {}, tokens: {} }, edgesByRow: {}, edgesByToken: {} };
    const L0 = buildInitialLayout(W, H, graph);
    let layout = L0;
    if (progress > 0.4 && graph.clusterByRow) {
      const L1 = buildClusteredLayout(W, H, graph, L0);
      const t = Math.max(0, Math.min(1, (progress - 0.4) / 0.6));
      layout = interpLayout(L0, L1, t);
    }
    const ebr = {}, ebt = {};
    for (const e of graph.edges) {
      (ebr[e.row] = ebr[e.row] || []).push(e);
      (ebt[e.token] = ebt[e.token] || []).push(e);
    }
    return { layout, edgesByRow: ebr, edgesByToken: ebt };
  }, [size.w, size.h, haveGraph, tick, progress]);

  if (!haveGraph) {
    return (
      <div ref={wrapRef} className="canvas-wrap" style={{
        display: 'grid', placeItems: 'center', color: 'var(--text-4)',
        fontFamily: 'var(--font-mono)', fontSize: 11.5,
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: 4 }}>граф ещё не построен</div>
          <div style={{ fontSize: 10.5, opacity: 0.7 }}>ждём событий по WebSocket…</div>
        </div>
      </div>
    );
  }

  if (!layout.rows || Object.keys(layout.rows).length === 0) {
    return <div ref={wrapRef} className="canvas-wrap"></div>;
  }

  // соседи выбранной строки
  const selectedNeighbors = new Set();
  if (selected && edgesByRow[selected]) {
    edgesByRow[selected].forEach((e) => selectedNeighbors.add(e.token));
  }

  // кластерные «coworkers» выбранной строки
  const selectedClusterMates = new Set();
  if (selected && graph.clusterByRow) {
    const cid = graph.clusterByRow[selected];
    if (cid) {
      for (const r of graph.rows) {
        if (graph.clusterByRow[r.id] === cid) selectedClusterMates.add(r.id);
      }
    }
  }

  const aCount = graph.rows.filter((r) => r.source === 'A').length;
  const bCount = graph.rows.filter((r) => r.source === 'B').length;

  // ограничим рендер: если слишком много рёбер — рисуем подвыборку, но pulse
  // продолжаем по всем (визуально не страшно).
  const maxEdgesRendered = 1500;
  const renderedEdges = graph.edges.length > maxEdgesRendered
    ? graph.edges.filter((_, i) => i % Math.ceil(graph.edges.length / maxEdgesRendered) === 0)
    : graph.edges;

  return (
    <div ref={wrapRef} className="canvas-wrap">
      <svg viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <radialGradient id="cluster-halo" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--cluster)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--cluster)" stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* edges */}
        {showEdges && showTokens && renderedEdges.map((e, i) => {
          const r = layout.rows[e.row];
          const t = layout.tokens[e.token];
          if (!r || !t) return null;
          const isSel = selected && (e.row === selected || selectedNeighbors.has(e.token));
          const op = isSel ? 0.75 : 0.22;
          return (
            <line key={i}
              x1={r.x} y1={r.y} x2={t.x} y2={t.y}
              stroke={isSel ? 'var(--row)' : 'var(--border-strong)'}
              strokeWidth={isSel ? 1.3 : 0.7}
              opacity={op} />
          );
        })}

        {/* pulse — message passing */}
        {pulseLayer !== null && showEdges && showTokens && renderedEdges.map((e, i) => {
          const dir = pulseLayer % 2 === 0 ? 1 : -1;
          const r = layout.rows[e.row];
          const t = layout.tokens[e.token];
          if (!r || !t) return null;
          const start = dir === 1 ? t : r;
          const end = dir === 1 ? r : t;
          const localPhase = (pulsePhase + (i * 0.013)) % 1;
          if (localPhase < 0.02 || localPhase > 0.98) return null;
          const x = lerp(start.x, end.x, localPhase);
          const y = lerp(start.y, end.y, localPhase);
          const color = pulseLayer === 0 ? 'var(--token)' : 'var(--row)';
          return (
            <circle key={`p-${i}`} cx={x} cy={y} r={1.6}
              fill={color}
              opacity={0.85 * (1 - Math.abs(localPhase - 0.5) * 1.5)} />
          );
        })}

        {/* token nodes */}
        {showTokens && graph.tokens.map((tok) => {
          const p = layout.tokens[tok.id];
          if (!p) return null;
          const isSel = selectedNeighbors.has(tok.id);
          return (
            <g key={`t-${tok.id}`}
              style={{ cursor: 'pointer' }}
              onMouseEnter={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip({ kind: 'token', id: tok.id, x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseMove={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip((h) => h ? { ...h, x: e.clientX - rect.left, y: e.clientY - rect.top } : h);
              }}
              onMouseLeave={() => setHoverTip(null)}>
              <circle cx={p.x} cy={p.y} r={isSel ? 4.5 : 3}
                fill={isSel ? 'var(--token)' : 'var(--surface-2)'}
                stroke={isSel ? 'var(--token)' : 'var(--border-strong)'}
                strokeWidth={isSel ? 1.5 : 1} />
              <circle cx={p.x} cy={p.y} r={8} fill="transparent" />
              {isSel && (
                <text x={p.x} y={p.y - 7} textAnchor="middle" fontSize="9"
                  fontFamily="var(--font-mono)" fill="var(--token)">
                  {tok.text}
                </text>
              )}
            </g>
          );
        })}

        {/* row nodes */}
        {graph.rows.map((r) => {
          const p = layout.rows[r.id];
          if (!p) return null;
          const isA = r.source === 'A';
          const isSel = selected === r.id;
          const isClusterMate = selectedClusterMates.has(r.id) && r.id !== selected;
          const fill = isSel ? 'var(--warn)' : (isA ? 'var(--row)' : 'var(--token)');
          const sz = isSel ? 16 : 10;
          return (
            <g key={`r-${r.id}`}
              transform={`translate(${p.x - sz/2}, ${p.y - sz/2})`}
              style={{ cursor: onSelectRow ? 'pointer' : 'default' }}
              onClick={() => onSelectRow && onSelectRow(r.id)}
              onMouseEnter={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip({ kind: 'row', id: r.id, x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseMove={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip((h) => h ? { ...h, x: e.clientX - rect.left, y: e.clientY - rect.top } : h);
              }}
              onMouseLeave={() => setHoverTip(null)}>
              {isSel && (
                <circle cx={sz/2} cy={sz/2} r={sz * 1.4}
                  fill="none" stroke="var(--warn)" strokeWidth="1.5" opacity="0.5">
                  <animate attributeName="r" from={sz * 0.8} to={sz * 1.6} dur="1.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from="0.7" to="0" dur="1.4s" repeatCount="indefinite" />
                </circle>
              )}
              <rect width={sz} height={sz} rx={2}
                fill={fill}
                stroke={isSel ? 'var(--text)' : (isClusterMate ? 'var(--cluster)' : 'transparent')}
                strokeWidth={isSel ? 2 : (isClusterMate ? 1.5 : 0)}
                filter={isSel ? 'url(#glow)' : undefined} />
            </g>
          );
        })}
      </svg>

      {hoverTip && hoverTip.id !== selected && (
        <HoverTooltip tip={hoverTip} graph={graph} canvasW={size.w} canvasH={size.h}
                      edgesByRow={edgesByRow} edgesByToken={edgesByToken} />
      )}

      <div className="legend">
        <div className="row-l"><span className="swatch sq" style={{ background: 'var(--row)' }}></span> row · table A ({aCount})</div>
        <div className="row-l"><span className="swatch sq" style={{ background: 'var(--token)' }}></span> row · table B ({bCount})</div>
        <div className="row-l"><span className="swatch" style={{ background: 'var(--surface-2)', border: '1px solid var(--border-strong)' }}></span> token ({graph.tokens.length})</div>
        {highlightClusters && (
          <div className="row-l"><span className="swatch" style={{ background: 'var(--cluster)', opacity: 0.5 }}></span> кластер</div>
        )}
      </div>
    </div>
  );
}


// ---- hover tooltip ------------------------------------------------------
function HoverTooltip({ tip, graph, canvasW, canvasH, edgesByRow, edgesByToken }) {
  const TIP_W = 240;
  const flipX = tip.x + TIP_W + 18 > canvasW;
  const left = flipX ? tip.x - TIP_W - 12 : tip.x + 12;
  const top  = Math.min(canvasH - 160, tip.y + 12);

  let content = null;
  if (tip.kind === 'row') {
    const row = graph.rows.find((r) => r.id === tip.id);
    if (!row) return null;
    const isA = row.source === 'A';
    const accent = isA ? 'var(--row)' : 'var(--token)';
    const colsArr = Object.entries(row.cols || {})
      .filter(([k, v]) => k !== 'id' && v != null && String(v) !== '')
      .slice(0, 8);
    content = (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: accent, display: 'inline-block' }}></span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: accent, fontWeight: 600 }}>{row.id}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)', marginLeft: 'auto' }}>
            {row.label || ''}
          </span>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)',
                      display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 10px' }}>
          {colsArr.map(([k, v]) => (
            <div key={k} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {k}: <span style={{ color: 'var(--text)' }}>{String(v)}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 10, color: 'var(--text-4)', fontFamily: 'var(--font-mono)' }}>
          degree = {(edgesByRow[row.id] || []).length} token edges
        </div>
      </>
    );
  } else if (tip.kind === 'token') {
    const tok = graph.tokens.find((t) => t.id === tip.id);
    if (!tok) return null;
    const incident = edgesByToken[tip.id] || [];
    const visible = incident.slice(0, 8);
    content = (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--token)', display: 'inline-block' }}></span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--token)', fontWeight: 600 }}>token</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)', marginLeft: 'auto' }}>df={tok.df}</span>
        </div>
        <div style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', marginBottom: 8, overflowWrap: 'anywhere' }}>
          {tok.text}
        </div>
        <div style={{ border: '1px solid var(--border)', borderRadius: 5, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>
            <thead>
              <tr style={{ background: 'var(--surface)' }}>
                <th style={{ padding: '3px 6px', textAlign: 'left', color: 'var(--text-4)', fontSize: 9.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.04 }}>row</th>
                <th style={{ padding: '3px 6px', textAlign: 'left', color: 'var(--text-4)', fontSize: 9.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.04 }}>col</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((e, i) => {
                const r = graph.rows.find((x) => x.id === e.row);
                const isA = r && r.source === 'A';
                return (
                  <tr key={i} style={{ borderTop: i > 0 ? '1px solid var(--border)' : 'none' }}>
                    <td style={{ padding: '3px 6px', color: isA ? 'var(--row)' : 'var(--token)', fontWeight: 600 }}>{e.row}</td>
                    <td style={{ padding: '3px 6px', color: 'var(--text-3)' }}>{e.col}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {incident.length > visible.length && (
          <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 4, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
            +{incident.length - visible.length} ещё
          </div>
        )}
      </>
    );
  }

  return (
    <div style={{
      position: 'absolute',
      left, top,
      width: TIP_W,
      background: 'color-mix(in oklch, var(--bg-elev) 95%, transparent)',
      border: '1px solid var(--border-strong)',
      borderRadius: 8,
      padding: '10px 12px',
      backdropFilter: 'blur(8px)',
      boxShadow: 'var(--shadow-md)',
      pointerEvents: 'none',
      zIndex: 10,
      maxHeight: tip.kind === 'token' ? 320 : 180,
    }}>
      {content}
    </div>
  );
}

Object.assign(window, { HeteroGraph });
