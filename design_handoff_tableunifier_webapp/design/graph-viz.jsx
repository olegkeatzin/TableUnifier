// Heterogeneous graph visualization: row nodes (squares) + token nodes (circles).
// Supports animated "phases" — building, training (message passing), clustered.

// ---- Token vocabulary (shared across both tables) ----
const TOKENS = [
  // brands
  'bmw', 'toyota', 'mercedes', 'audi', 'volkswagen', 'kia',
  'hyundai', 'lada', 'skoda', 'renault', 'mazda', 'mitsubishi',
  'ford', 'nissan',
  // models
  'x5', 'camry', 'e-class', 'a6', 'tiguan', 'rio', 'solaris',
  'vesta', 'octavia', 'logan', 'rav4', 'cx-5', 'outlander',
  'focus', 'qashqai', 'x3',
  // bodyType / color / common
  'седан', 'внедорожник', 'лифтбек', 'белый', 'чёрный',
  'серебр.', 'красный', 'синий', 'серый',
];

// Per-row edges to tokens (indices into TOKENS).
// First N for table A (14 rows), then N for table B (13 rows).
const ROW_TOKENS = {
  // table A (14 cars)
  'A0':  [0, 14, 33, 35],            // BMW X5 / внедорожник / белый
  'A1':  [1, 15, 32, 36],            // Toyota Camry / седан / чёрный
  'A2':  [2, 16, 32, 37],            // Mercedes E-Class / седан / серебр.
  'A3':  [3, 17, 32, 40],            // Audi A6 / седан / серый
  'A4':  [4, 18, 33, 38],            // VW Tiguan / внедорожник / красный
  'A5':  [5, 19, 32, 39],            // Kia Rio / седан / синий
  'A6':  [6, 20, 32, 35],            // Hyundai Solaris / седан / белый
  'A7':  [7, 21, 32, 39],            // Lada Vesta / седан / синий
  'A8':  [8, 22, 34, 35],            // Skoda Octavia / лифтбек / белый
  'A9':  [9, 23, 32, 37],            // Renault Logan / седан / серебр.
  'A10': [0, 14, 33, 36],            // BMW X5 / внедорожник / чёрный
  'A11': [1, 24, 33, 35],            // Toyota RAV4 / внедорожник / белый
  'A12': [10, 25, 33, 38],           // Mazda CX-5 / внедорожник / красный
  'A13': [11, 26, 33, 40],           // Mitsubishi Outlander / внедорожник / серый
  // table B (13 cars)
  'B0':  [0, 14, 33, 35],            // BMW X5 / внедорожник / белый
  'B1':  [1, 15, 32, 36],            // Toyota Camry / седан / чёрный
  'B2':  [2, 16, 32, 37],            // Mercedes E / седан / серебр.
  'B3':  [3, 17, 32, 40],            // Audi A6 / седан / серый
  'B4':  [6, 20, 32, 35],            // Hyundai Solaris / седан / белый
  'B5':  [8, 22, 34, 35],            // Skoda Octavia / лифтбек / белый
  'B6':  [1, 24, 33, 35],            // Toyota RAV4 / внедорожник / белый
  'B7':  [10, 25, 33, 38],           // Mazda CX-5 / внедорожник / красный
  'B8':  [12, 27, 32, 39],           // Ford Focus / седан / синий
  'B9':  [13, 28, 33, 35],           // Nissan Qashqai / внедорожник / белый
  'B10': [0, 29, 33, 36],            // BMW X3 / внедорожник / чёрный
  'B11': [0, 14, 33, 39],            // BMW X5 2022 / внедорожник / синий (false-match-prone)
  'B12': [7, 21, 32, 39],            // Lada Vesta / седан / синий
};

// True duplicate clusters — for the "clustered" layout target
const TRUE_CLUSTERS = [
  ['A0', 'B0'], ['A1', 'B1'], ['A2', 'B2'], ['A3', 'B3'],
  ['A6', 'B4'], ['A8', 'B5'], ['A11', 'B6'], ['A12', 'B7'],
  ['A7', 'B12'],
  // singletons
  ['A4'], ['A5'], ['A9'], ['A10'], ['A13'],
  ['B8'], ['B9'], ['B10'], ['B11'],
];

// -------------------------------------------------------------
// Position layout helpers
// -------------------------------------------------------------
function makeSeededRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
}

// Initial: bipartite-ish stable layout — A rows on left column, B rows on right column,
// tokens spread in vertical center band. This layout is STATIC and shows graph topology.
function buildInitialLayout(W, H) {
  const rng = makeSeededRng(42);
  const rows = {}, tokens = {};
  const rowIds = Object.keys(ROW_TOKENS);
  const aIds = rowIds.filter((id) => id[0] === 'A');
  const bIds = rowIds.filter((id) => id[0] === 'B');

  // A rows down the left side, evenly spaced vertically
  aIds.forEach((id, i) => {
    const t = aIds.length > 1 ? i / (aIds.length - 1) : 0.5;
    rows[id] = {
      x: W * 0.12 + (rng() - 0.5) * 12,
      y: H * 0.08 + t * H * 0.84 + (rng() - 0.5) * 8,
    };
  });
  // B rows down the right side
  bIds.forEach((id, i) => {
    const t = bIds.length > 1 ? i / (bIds.length - 1) : 0.5;
    rows[id] = {
      x: W * 0.88 + (rng() - 0.5) * 12,
      y: H * 0.08 + t * H * 0.84 + (rng() - 0.5) * 8,
    };
  });

  // Tokens — clustered in the center column, in a couple loose rows
  TOKENS.forEach((_, i) => {
    const col = i % 5;
    const row = Math.floor(i / 5);
    const totalCols = 5;
    const totalRows = Math.ceil(TOKENS.length / totalCols);
    tokens[i] = {
      x: W * 0.32 + (col / (totalCols - 1)) * W * 0.36 + (rng() - 0.5) * 14,
      y: H * 0.10 + (row / Math.max(1, totalRows - 1)) * H * 0.80 + (rng() - 0.5) * 14,
    };
  });
  return { rows, tokens };
}

// Target: rows clustered by true cluster, tokens float free in mid
function buildClusteredLayout(W, H) {
  const rng = makeSeededRng(17);
  const rows = {}, tokens = {};
  // Arrange clusters in a loose grid
  const clusterCenters = [];
  const cols = 6, rowsCount = Math.ceil(TRUE_CLUSTERS.length / cols);
  TRUE_CLUSTERS.forEach((cluster, i) => {
    const cx = W * (0.10 + ((i % cols) / (cols - 1)) * 0.80);
    const cy = H * (0.18 + Math.floor(i / cols) / Math.max(1, rowsCount - 1) * 0.64);
    clusterCenters.push({ x: cx, y: cy });
    cluster.forEach((rowId, j) => {
      const angle = (j / cluster.length) * Math.PI * 2 + i * 0.7;
      const r = cluster.length > 1 ? 18 : 0;
      rows[rowId] = { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
    });
  });
  // tokens stay near their initial positions (slight settle toward center band)
  // — this keeps edges visually attached to real nodes instead of flying to a far ring
  const L0 = buildInitialLayout(W, H);
  TOKENS.forEach((_, i) => {
    const init = L0.tokens[i];
    // pull a little toward center, but keep dispersed
    tokens[i] = {
      x: lerp(init.x, W * 0.5, 0.15),
      y: lerp(init.y, H * 0.5, 0.15),
    };
  });
  return { rows, tokens };
}

// linear interp
function lerp(a, b, t) { return a + (b - a) * t; }
function interpLayout(L0, L1, t) {
  const rows = {}, tokens = {};
  for (const k in L0.rows) {
    rows[k] = { x: lerp(L0.rows[k].x, L1.rows[k].x, t), y: lerp(L0.rows[k].y, L1.rows[k].y, t) };
  }
  for (const k in L0.tokens) {
    tokens[k] = { x: lerp(L0.tokens[k].x, L1.tokens[k].x, t), y: lerp(L0.tokens[k].y, L1.tokens[k].y, t) };
  }
  return { rows, tokens };
}

// -------------------------------------------------------------
// Graph component
// -------------------------------------------------------------
// progress: ignored for layout (graph is structural and STATIC). Kept for API compatibility
//   so the calling screen can still drive halo + edge intensity if needed.
// pulseLayer: which token→row layer is currently message-passing (or null)
// hovered: rowId
// showTokens: bool
// showEdges: bool
// -------------------------------------------------------------
function HeteroGraph({
  progress = 0,
  pulseLayer = null,
  pulsePhase = 0, // 0..1, advanced by training animation
  showTokens = true,
  showEdges = true,
  selected = null,
  onSelectRow = null,
  highlightClusters = false,
}) {
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [hoverTip, setHoverTip] = useState(null); // { kind: 'row'|'token', id, x, y }

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

  const { layout, edges } = useMemo(() => {
    const W = size.w, H = size.h;
    if (W < 50 || H < 50) return { layout: { rows: {}, tokens: {} }, edges: [] };
    // STATIC layout — graph topology doesn't change during training
    const layout = buildInitialLayout(W, H);
    const edges = [];
    Object.entries(ROW_TOKENS).forEach(([rowId, tokIdxs]) => {
      tokIdxs.forEach((ti) => {
        edges.push({ rowId, tok: ti });
      });
    });
    return { layout, edges };
  }, [size.w, size.h]);

  if (!layout.rows || Object.keys(layout.rows).length === 0) {
    return <div ref={wrapRef} className="canvas-wrap"></div>;
  }

  // helper: which cluster does a row belong to
  const rowCluster = {};
  TRUE_CLUSTERS.forEach((c, i) => c.forEach((r) => { rowCluster[r] = i; }));

  // selected row's neighbors
  const selectedNeighbors = new Set();
  if (selected && ROW_TOKENS[selected]) {
    ROW_TOKENS[selected].forEach((t) => selectedNeighbors.add(t));
  }
  // selected row's cluster mates
  const selectedClusterMates = new Set();
  if (selected) {
    const ci = rowCluster[selected];
    if (ci != null) TRUE_CLUSTERS[ci].forEach((r) => selectedClusterMates.add(r));
  }

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

        {/* cluster halos disabled — structural graph is static, cluster visualization belongs to embedding space */}

        {/* edges */}
        {showEdges && edges.map((e, i) => {
          const r = layout.rows[e.rowId];
          const t = layout.tokens[e.tok];
          if (!r || !t) return null;
          const isSel = selected && (e.rowId === selected || selectedNeighbors.has(e.tok));
          if (!showTokens) return null;
          // base visibility — stable, graph is static
          const baseOp = 0.22;
          const op = isSel ? 0.75 : baseOp;
          return (
            <line key={i}
              x1={r.x} y1={r.y} x2={t.x} y2={t.y}
              stroke={isSel ? 'var(--row)' : 'var(--border-strong)'}
              strokeWidth={isSel ? 1.3 : 0.7}
              opacity={op} />
          );
        })}

        {/* pulse edges — message passing animation */}
        {pulseLayer !== null && showEdges && showTokens && edges.map((e, i) => {
          // direction depends on layer: 0 = token→row, 1 = row→token, 2 = token→row
          const dir = pulseLayer % 2 === 0 ? 1 : -1;
          const r = layout.rows[e.rowId];
          const t = layout.tokens[e.tok];
          if (!r || !t) return null;
          // animate a dot from t to r (or reverse)
          const start = dir === 1 ? t : r;
          const end = dir === 1 ? r : t;
          const px = lerp(start.x, end.x, pulsePhase);
          const py = lerp(start.y, end.y, pulsePhase);
          // stagger pulses by edge index
          const localPhase = (pulsePhase + (i * 0.013)) % 1;
          if (localPhase < 0.02 || localPhase > 0.98) return null;
          const x = lerp(start.x, end.x, localPhase);
          const y = lerp(start.y, end.y, localPhase);
          const color = pulseLayer === 0 ? 'var(--token)' : 'var(--row)';
          return (
            <circle key={`p-${i}`}
              cx={x} cy={y} r={1.6}
              fill={color}
              opacity={0.85 * (1 - Math.abs(localPhase - 0.5) * 1.5)} />
          );
        })}

        {/* token nodes */}
        {showTokens && Object.entries(layout.tokens).map(([ti, p]) => {
          const isSel = selectedNeighbors.has(parseInt(ti, 10));
          return (
            <g key={`t-${ti}`}
              style={{ cursor: 'pointer' }}
              onMouseEnter={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip({ kind: 'token', id: parseInt(ti, 10), x: e.clientX - rect.left, y: e.clientY - rect.top });
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
              {/* invisible larger hit area */}
              <circle cx={p.x} cy={p.y} r={8} fill="transparent" />
              {isSel && (
                <text x={p.x} y={p.y - 7}
                  textAnchor="middle" fontSize="9"
                  fontFamily="var(--font-mono)"
                  fill="var(--token)">
                  {TOKENS[ti]}
                </text>
              )}
            </g>
          );
        })}

        {/* row nodes */}
        {Object.entries(layout.rows).map(([rid, p]) => {
          const isA = rid[0] === 'A';
          const isSel = selected === rid;
          const isClusterMate = selectedClusterMates.has(rid) && rid !== selected;
          const fill = isSel ? 'var(--warn)' : (isA ? 'var(--row)' : 'var(--token)');
          const size = isSel ? 16 : 10;
          return (
            <g key={`r-${rid}`}
              transform={`translate(${p.x - size/2}, ${p.y - size/2})`}
              style={{ cursor: onSelectRow ? 'pointer' : 'default' }}
              onClick={() => onSelectRow && onSelectRow(rid)}
              onMouseEnter={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip({ kind: 'row', id: rid, x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseMove={(e) => {
                const rect = wrapRef.current.getBoundingClientRect();
                setHoverTip((h) => h ? { ...h, x: e.clientX - rect.left, y: e.clientY - rect.top } : h);
              }}
              onMouseLeave={() => setHoverTip(null)}>
              {isSel && (
                <circle cx={size/2} cy={size/2} r={size * 1.4}
                  fill="none"
                  stroke="var(--warn)"
                  strokeWidth="1.5"
                  opacity="0.5">
                  <animate attributeName="r" from={size * 0.8} to={size * 1.6} dur="1.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from="0.7" to="0" dur="1.4s" repeatCount="indefinite" />
                </circle>
              )}
              <rect
                width={size} height={size} rx={2}
                fill={fill}
                stroke={isSel ? 'var(--text)' : (isClusterMate ? 'var(--cluster)' : 'transparent')}
                strokeWidth={isSel ? 2 : (isClusterMate ? 1.5 : 0)}
                filter={isSel ? 'url(#glow)' : undefined} />
            </g>
          );
        })}
      </svg>

      {/* Hover tooltip */}
      {hoverTip && hoverTip.id !== selected && (
        <HoverTooltip tip={hoverTip} canvasW={size.w} canvasH={size.h} />
      )}

      <div className="legend">
        <div className="row-l"><span className="swatch sq" style={{ background: 'var(--row)' }}></span> row · table A (14)</div>
        <div className="row-l"><span className="swatch sq" style={{ background: 'var(--token)' }}></span> row · table B (13)</div>
        <div className="row-l"><span className="swatch" style={{ background: 'var(--surface-2)', border: '1px solid var(--border-strong)' }}></span> token ({TOKENS.length})</div>
        {highlightClusters && (
          <div className="row-l"><span className="swatch" style={{ background: 'var(--cluster)', opacity: 0.5 }}></span> кластер</div>
        )}
      </div>
    </div>
  );
}

Object.assign(window, { HeteroGraph, TOKENS, ROW_TOKENS, TRUE_CLUSTERS });

// ---- Hover tooltip ----
function HoverTooltip({ tip, canvasW, canvasH }) {
  const TIP_W = 220;
  // place to the right + below mouse, but flip if too close to edge
  const flipX = tip.x + TIP_W + 18 > canvasW;
  const left = flipX ? tip.x - TIP_W - 12 : tip.x + 12;
  const top  = Math.min(canvasH - 130, tip.y + 12);

  const D = window.__DATA__;
  let content = null;
  if (tip.kind === 'row') {
    const isA = tip.id[0] === 'A';
    const tbl = isA ? D.tableA : D.tableB;
    const idx = parseInt(tip.id.slice(1), 10);
    const row = tbl.data[idx];
    if (!row) return null;
    const accent = isA ? 'var(--row)' : 'var(--token)';
    content = (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: accent, display: 'inline-block' }}></span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: accent, fontWeight: 600 }}>{tip.id}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)', marginLeft: 'auto' }}>
            {tbl.name.replace('.xlsx', '')}
          </span>
        </div>
        <div style={{ fontSize: 12.5, fontWeight: 500, marginBottom: 6 }}>
          {row[1]} {row[2]} · {row[3]}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 10px' }}>
          <div>цвет: <span style={{ color: 'var(--text)' }}>{String(row[5])}</span></div>
          <div>кузов: <span style={{ color: 'var(--text)' }}>{String(row[6]).toLowerCase()}</span></div>
          <div>пробег: <span style={{ color: 'var(--text)' }}>{row[4].toLocaleString('ru-RU')}</span></div>
          <div>двиг.: <span style={{ color: 'var(--text)' }}>{isA ? row[7] + 'L' : (row[7] / 1000).toFixed(1) + 'L'}</span></div>
        </div>
      </>
    );
  } else if (tip.kind === 'token') {
    const text = TOKENS[tip.id];
    // derive column from token index range
    const tokCategory =
      tip.id <= 13 ? { aCol: 'mark',    bCol: 'brand',      colIdx: 1 } :
      tip.id <= 29 ? { aCol: 'model',   bCol: 'model_name', colIdx: 2 } :
      tip.id <= 32 ? { aCol: 'bodyType',bCol: 'body_type',  colIdx: 6 } :
                     { aCol: 'color',   bCol: 'color_hex',  colIdx: 5 };
    // count occurrences (df) and collect (row, col, value)
    const entries = [];
    Object.entries(ROW_TOKENS).forEach(([rid, tokIdxs]) => {
      if (tokIdxs.includes(tip.id)) {
        const isA = rid[0] === 'A';
        const tbl = isA ? D.tableA : D.tableB;
        const idx = parseInt(rid.slice(1), 10);
        const row = tbl.data[idx];
        if (row) {
          entries.push({
            rid,
            col: isA ? tokCategory.aCol : tokCategory.bCol,
            value: String(row[tokCategory.colIdx]),
            isA,
          });
        }
      }
    });
    const df = entries.length;
    const visible = entries.slice(0, 6);
    content = (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--token)', display: 'inline-block' }}></span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--token)', fontWeight: 600 }}>token</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)', marginLeft: 'auto' }}>df={df}</span>
        </div>
        <div style={{ fontSize: 14, fontWeight: 600, fontFamily: 'var(--font-mono)', marginBottom: 8 }}>
          {text}
        </div>
        <div style={{ border: '1px solid var(--border)', borderRadius: 5, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>
            <thead>
              <tr style={{ background: 'var(--surface)' }}>
                <th style={{ padding: '3px 6px', textAlign: 'left', color: 'var(--text-4)', fontSize: 9.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.04 }}>row</th>
                <th style={{ padding: '3px 6px', textAlign: 'left', color: 'var(--text-4)', fontSize: 9.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.04 }}>col</th>
                <th style={{ padding: '3px 6px', textAlign: 'left', color: 'var(--text-4)', fontSize: 9.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.04 }}>value</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((e, i) => (
                <tr key={i} style={{ borderTop: i > 0 ? '1px solid var(--border)' : 'none' }}>
                  <td style={{ padding: '3px 6px', color: e.isA ? 'var(--row)' : 'var(--token)', fontWeight: 600 }}>{e.rid}</td>
                  <td style={{ padding: '3px 6px', color: 'var(--text-3)' }}>{e.col}</td>
                  <td style={{ padding: '3px 6px', color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 90 }}>{e.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {df > 6 && (
          <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 4, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
            +{df - 6} ещё
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
      maxHeight: tip.kind === 'token' ? 280 : 140,
    }}>
      {content}
    </div>
  );
}
