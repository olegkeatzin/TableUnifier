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
function EmbeddingSpace({ progress = 0, hovered = null, onHover = null, dims = '1024',
                          candidates = [], threshold = null }) {
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 400, h: 400 });
  const [tick, setTick] = useState(0);
  const [hoverTip, setHoverTip] = useState(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setSize({ w: el.offsetWidth, h: el.offsetHeight });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  // Указатель → логическое пространство холста (как в HeteroGraph: offsetWidth —
  // layout px, getBoundingClientRect — zoom-scaled, их отношение убирает --ui-zoom).
  const localPt = (e) => {
    const el = wrapRef.current;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    const sx = rect.width ? el.offsetWidth / rect.width : 1;
    const sy = rect.height ? el.offsetHeight / rect.height : 1;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  };

  // Re-render when graph data arrives (dispatched by api.js getGraph).
  useEffect(() => {
    const h = () => setTick((t) => t + 1);
    window.addEventListener('graph-updated', h);
    return () => window.removeEventListener('graph-updated', h);
  }, []);

  const graph = window.__DATA__.graph;
  const rows = (graph && graph.rows) || [];
  const clusterByRow = (graph && graph.clusterByRow) || {};

  const { p0, p1, edgesByRow } = useMemo(() => {
    if (rows.length === 0) return { p0: {}, p1: {}, edgesByRow: {} };
    const ebr = {};
    for (const e of ((graph && graph.edges) || [])) {
      (ebr[e.row] = ebr[e.row] || []).push(e);
    }
    return {
      p0: buildEmbInitial(rows),
      p1: buildEmbFinal(rows, clusterByRow),
      edgesByRow: ebr,
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows.length, tick]);

  if (rows.length === 0) {
    return (
      <div ref={wrapRef} className="canvas-wrap" style={{
        width: '100%', height: '100%',
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
    <div ref={wrapRef} className="canvas-wrap" style={{ width: '100%', height: '100%', background: 'var(--bg-elev)', borderRadius: 'var(--r-lg)', border: '1px solid var(--border)' }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
        <defs>
          <pattern id="emb-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.5" />
          </pattern>
        </defs>
        <rect width={W} height={H} fill="url(#emb-grid)" />

        {/* оси проекции — подписаны по центру каждой стороны */}
        <text x={W / 2} y={H - 7} textAnchor="middle"
          fontSize="11" fontFamily="var(--font-mono)" fill="var(--text-4)" letterSpacing="0.04em">
          UMAP-1 →
        </text>
        <text x={13} y={H / 2} textAnchor="middle"
          fontSize="11" fontFamily="var(--font-mono)" fill="var(--text-4)" letterSpacing="0.04em"
          transform={`rotate(-90, 13, ${H / 2})`}>
          UMAP-2 →
        </text>

        {/* trails */}
        {t > 0.05 && Object.entries(pos).map(([rid, p]) => (
          <line key={`tr-${rid}`}
            x1={sx(p0[rid].x)} y1={sy(p0[rid].y)}
            x2={sx(p.x)} y2={sy(p.y)}
            stroke={rid[0] === 'A' ? 'var(--row)' : 'var(--row-b)'}
            strokeWidth="0.5" opacity={0.25 * (1 - t * 0.7)} />
        ))}

        {/* pair edges — рёбра между строками, объединёнными в пары моделью.
            Цвет относительно порога (как в окне «Проверка»): зелёный — авто-слияние,
            жёлтый — требует проверки, серый — ниже порога (de-emphasized). */}
        {(() => {
          if (!(threshold > 0) || !candidates || candidates.length === 0) return null;
          // classifyPair объявлена в screen-review.jsx (общий глобал между скриптами).
          const classify = (typeof classifyPair === 'function')
            ? classifyPair
            : ((s, thr) => (s >= thr ? 'auto' : 'reject'));
          const STYLE = {
            auto:   { stroke: 'var(--cluster)', opacity: 0.7,  width: 1.6 },
            review: { stroke: 'var(--warn)',    opacity: 0.8,  width: 1.6 },
            reject: { stroke: 'var(--text-4)',  opacity: 0.1,  width: 0.8 },
          };
          const fade = Math.max(0, Math.min(1, (t - 0.35) / 0.45)); // появляются по мере кластеризации
          if (fade <= 0) return null;
          // reject рисуем первыми, авто/проверку — поверх, чтобы яркие рёбра не перекрывались
          const order = { reject: 0, review: 1, auto: 2 };
          const edges = candidates
            .map((c) => {
              const ida = `${c.a[0]}${c.a[1]}`, idb = `${c.b[0]}${c.b[1]}`;
              const pa = pos[ida], pb = pos[idb];
              if (!pa || !pb) return null;
              const sim = c.sim ?? c.similarity ?? 0;
              const verdict = classify(sim, threshold);
              return { ida, idb, pa, pb, verdict };
            })
            .filter(Boolean)
            .sort((a, b) => order[a.verdict] - order[b.verdict]);
          return edges.map((e, i) => {
            const st = STYLE[e.verdict];
            const isHov = hovered === e.ida || hovered === e.idb;
            return (
              <line key={`pe-${i}`}
                x1={sx(e.pa.x)} y1={sy(e.pa.y)}
                x2={sx(e.pb.x)} y2={sy(e.pb.y)}
                stroke={st.stroke}
                strokeWidth={isHov ? st.width + 1 : st.width}
                opacity={(isHov ? Math.min(1, st.opacity + 0.25) : st.opacity) * fade}
                strokeLinecap="round" />
            );
          });
        })()}

        {/* points */}
        {Object.entries(pos).map(([rid, p]) => {
          const isA = rid[0] === 'A';
          const isHov = hovered === rid;
          return (
            <g key={rid}
              transform={`translate(${sx(p.x)}, ${sy(p.y)})`}
              onMouseEnter={(e) => {
                onHover && onHover(rid);
                const lp = localPt(e);
                setHoverTip({ kind: 'row', id: rid, x: lp.x, y: lp.y });
              }}
              onMouseMove={(e) => {
                const lp = localPt(e);
                setHoverTip((h) => (h ? { ...h, x: lp.x, y: lp.y } : h));
              }}
              onMouseLeave={() => { onHover && onHover(null); setHoverTip(null); }}
              style={{ cursor: onHover ? 'pointer' : 'default' }}>
              {/* увеличенная прозрачная зона захвата курсора */}
              <circle r={9} fill="transparent" />
              <circle r={isHov ? 6 : 4}
                fill={isA ? 'var(--row)' : 'var(--row-b)'}
                stroke={isHov ? 'var(--text)' : 'transparent'}
                strokeWidth={1.5} />
            </g>
          );
        })}

        {/* легенда цветов рёбер — появляется вместе с парами */}
        {(threshold > 0) && candidates && candidates.length > 0 && t > 0.4 && (
          <g transform={`translate(${W - 132}, 14)`} fontFamily="var(--font-mono)" fontSize="9.5">
            <rect x={-8} y={-9} width={132} height={34} rx={5}
              fill="var(--bg-elev)" stroke="var(--border)" strokeWidth="0.75" opacity="0.92" />
            <line x1={0} y1={1} x2={16} y2={1} stroke="var(--cluster)" strokeWidth="1.6" strokeLinecap="round" />
            <text x={21} y={4} fill="var(--text-3)">авто-слияние</text>
            <line x1={0} y1={15} x2={16} y2={15} stroke="var(--warn)" strokeWidth="1.6" strokeLinecap="round" />
            <text x={21} y={18} fill="var(--text-3)">требует проверки</text>
          </g>
        )}
      </svg>

      {/* описание строки при наведении — тот же тултип, что и в графе */}
      {hoverTip && typeof HoverTooltip === 'function' && (
        <HoverTooltip tip={hoverTip} graph={graph} canvasW={W} canvasH={H}
                      edgesByRow={edgesByRow} edgesByToken={{}} />
      )}
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
    // offsetWidth = layout px, immune to the --ui-zoom CSS zoom (getBoundingClientRect
    // would be zoom-scaled and overflow at high zoom).
    const measure = () => setW(el.offsetWidth);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
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
