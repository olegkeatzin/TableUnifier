// 2D embedding space — shows row embeddings as dots that move/cluster
// during inference. Uses real row data from window.__DATA__.graph.

function makeSeededRng2(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
}

// Hash a string to a uint32 seed.
function strHash(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  return h;
}

function buildEmbInitial(rows) {
  const pos = {};
  rows.forEach((r) => {
    const rng = makeSeededRng2(strHash(r.id) ^ 7);
    pos[r.id] = { x: rng() * 0.88 + 0.06, y: rng() * 0.88 + 0.06 };
  });
  return pos;
}

function buildEmbFinal(rows, clusterByRow) {
  const rng = makeSeededRng2(99);
  const pos = {};

  // Group rows by cluster id.
  const clusterMap = {};
  rows.forEach((r) => {
    const cid = (clusterByRow && clusterByRow[r.id]) || `_solo_${r.id}`;
    (clusterMap[cid] = clusterMap[cid] || []).push(r.id);
  });
  const clusterIds = Object.keys(clusterMap);
  const N = clusterIds.length || 1;
  clusterIds.forEach((cid, i) => {
    const angle = (i / N) * Math.PI * 2;
    const cx = 0.5 + Math.cos(angle) * 0.32;
    const cy = 0.5 + Math.sin(angle) * 0.32;
    clusterMap[cid].forEach((rid, j) => {
      const a2 = (j / Math.max(1, clusterMap[cid].length)) * Math.PI * 2 + i * 0.31;
      const spread = clusterMap[cid].length > 1 ? 0.035 : 0;
      pos[rid] = {
        x: cx + Math.cos(a2) * spread + (rng() - 0.5) * 0.01,
        y: cy + Math.sin(a2) * spread + (rng() - 0.5) * 0.01,
      };
    });
  });
  return pos;
}

function lerp2(a, b, t) { return a + (b - a) * t; }

// progress 0..1 — interpolates from random scatter to clustered layout.
// Uses real rows from window.__DATA__.graph; shows placeholder if not ready.
function EmbeddingSpace({ progress = 0, hovered = null, onHover = null, dims = '1024' }) {
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 400, h: 400 });
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

  // Re-render when graph data arrives (dispatched by api.js getGraph).
  useEffect(() => {
    const h = () => setTick((t) => t + 1);
    window.addEventListener('graph-updated', h);
    return () => window.removeEventListener('graph-updated', h);
  }, []);

  const graph = window.__DATA__.graph;
  const rows = (graph && graph.rows) || [];
  const clusterByRow = (graph && graph.clusterByRow) || {};

  const { p0, p1 } = useMemo(() => {
    if (rows.length === 0) return { p0: {}, p1: {} };
    return {
      p0: buildEmbInitial(rows),
      p1: buildEmbFinal(rows, clusterByRow),
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows.length, tick]);

  if (rows.length === 0) {
    return (
      <div ref={wrapRef} className="canvas-wrap" style={{
        background: 'var(--bg-elev)', borderRadius: 'var(--r-lg)',
        border: '1px solid var(--border)',
        display: 'grid', placeItems: 'center',
        color: 'var(--text-4)', fontFamily: 'var(--font-mono)', fontSize: 11.5,
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: 4 }}>эмбеддинги ещё не готовы</div>
          <div style={{ fontSize: 10.5, opacity: 0.7 }}>появятся после инференса…</div>
        </div>
      </div>
    );
  }

  // ease the progress with a slight overshoot
  const t = Math.max(0, Math.min(1, progress));

  // map normalized 0..1 → screen px (with margin)
  const M = 28;
  const W = size.w, H = size.h;
  const sx = (x) => M + x * (W - 2 * M);
  const sy = (y) => M + y * (H - 2 * M);

  // build current pos for each row
  const pos = {};
  Object.keys(p0).forEach((id) => {
    pos[id] = {
      x: lerp2(p0[id].x, p1[id].x, t),
      y: lerp2(p0[id].y, p1[id].y, t),
    };
  });

  // build "trails" — show the path from start to current
  return (
    <div ref={wrapRef} className="canvas-wrap" style={{ background: 'var(--bg-elev)', borderRadius: 'var(--r-lg)', border: '1px solid var(--border)' }}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <pattern id="emb-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.5" />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#emb-grid)" />

        {/* axes labels */}
        <text x={M} y={H - 10} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)">
          dim_1 ({dims}-d → UMAP-2)
        </text>
        <text x={M + 4} y={M + 4} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)" transform={`rotate(-90, ${M + 4}, ${M + 4})`}>
          dim_2
        </text>

        {/* trails */}
        {t > 0.05 && Object.entries(pos).map(([rid, p]) => (
          <line key={`tr-${rid}`}
            x1={sx(p0[rid].x)} y1={sy(p0[rid].y)}
            x2={sx(p.x)} y2={sy(p.y)}
            stroke={rid[0] === 'A' ? 'var(--row)' : 'var(--token)'}
            strokeWidth="0.5" opacity={0.25 * (1 - t * 0.7)} />
        ))}

        {/* points */}
        {Object.entries(pos).map(([rid, p]) => {
          const isA = rid[0] === 'A';
          const isHov = hovered === rid;
          return (
            <g key={rid}
              transform={`translate(${sx(p.x)}, ${sy(p.y)})`}
              onMouseEnter={() => onHover && onHover(rid)}
              onMouseLeave={() => onHover && onHover(null)}
              style={{ cursor: onHover ? 'pointer' : 'default' }}>
              <circle r={isHov ? 6 : 4}
                fill={isA ? 'var(--row)' : 'var(--token)'}
                stroke={isHov ? 'var(--text)' : 'transparent'}
                strokeWidth={1.5} />
              {isHov && (
                <text x={8} y={4} fontSize="10" fontFamily="var(--font-mono)" fill="var(--text)">
                  {rid}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ---- Loss curve component ----
function LossCurve({ data, height = 80 }) {
  // data: [{ epoch, loss, val_loss }]
  const wrapRef = useRef(null);
  const [w, setW] = useState(400);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.getBoundingClientRect().width));
    ro.observe(el);
    setW(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);

  if (data.length === 0) {
    return <div ref={wrapRef} style={{ height }}></div>;
  }
  const maxL = Math.max(...data.map((d) => Math.max(d.loss, d.val_loss || 0))) * 1.05;
  const minL = 0;
  const W = w, H = height, M = 22;
  const sx = (i) => M + (i / Math.max(1, data.length - 1)) * (W - M - 8);
  const sy = (v) => H - 14 - ((v - minL) / (maxL - minL || 1)) * (H - M - 8);
  const path = (key) => data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${sx(i)} ${sy(d[key])}`).join(' ');

  return (
    <div ref={wrapRef} style={{ width: '100%' }}>
      <svg width={W} height={H}>
        {/* grid */}
        {[0.25, 0.5, 0.75].map((p) => (
          <line key={p} x1={M} y1={M + (H - M - 14) * p} x2={W - 8} y2={M + (H - M - 14) * p}
            stroke="var(--border)" strokeWidth="0.5" opacity="0.6" />
        ))}
        {/* train */}
        <path d={path('loss')} fill="none" stroke="var(--row)" strokeWidth="1.5" />
        {/* val */}
        <path d={path('val_loss')} fill="none" stroke="var(--cluster)" strokeWidth="1.5" strokeDasharray="3 3" />
        {/* labels */}
        <text x={W - 8} y={M + 8} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--row)">train</text>
        <text x={W - 8} y={M + 20} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--cluster)">val</text>
        <text x={4} y={M} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)">{maxL.toFixed(2)}</text>
        <text x={4} y={H - 4} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)">0.00</text>
        <text x={M} y={H - 2} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)">epoch</text>
      </svg>
    </div>
  );
}

Object.assign(window, { EmbeddingSpace, LossCurve });
