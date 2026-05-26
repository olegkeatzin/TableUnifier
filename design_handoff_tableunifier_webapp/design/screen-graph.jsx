// Screen 2 — Build heterogeneous graph (row + token nodes)

function ScreenGraph({ onContinue, onBack }) {
  const [phase, setPhase] = useState('idle'); // idle | embed | tokenize | build | done
  const [progress, setProgress] = useState(0); // 0..1 within phase
  const [selectedRow, setSelectedRow] = useState(null);
  const [showTokens, setShowTokens] = useState(true);
  const [idfMin, setIdfMin] = useState(2);
  const [logs, setLogs] = useState([]);
  const [panelPos, setPanelPos] = useState({ x: null, y: 14 }); // null = default right anchor
  const [dragging, setDragging] = useState(null); // { offsetX, offsetY }
  const panelRef = useRef(null);

  const addLog = (level, msg) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs((L) => [...L, { t: ts, level, msg }].slice(-200));
  };

  // run the build animation
  useEffect(() => {
    if (phase !== 'idle') return;
    let cancelled = false;
    addLog('info', 'starting graph build · model=bge-m3 · target_col_dim=1024');

    const run = async () => {
      // Phase 1 — row embeddings
      setPhase('embed');
      addLog('info', 'TokenEmbedder.fit → batch=8 · pooling=cls');
      const N = 27;
      for (let i = 1; i <= N && !cancelled; i++) {
        setProgress(i / N);
        if (i === 12) addLog('info', `[A] computed row embeddings 12/14 · 312-dim`);
        if (i === 22) addLog('info', `[B] computed row embeddings 8/13 · 312-dim`);
        await sleep(40);
      }
      addLog('ok', `row embeddings ready · ${N} × 1024d`);

      // Phase 2 — column embeddings via Ollama
      if (cancelled) return;
      setPhase('tokenize');
      setProgress(0);
      addLog('info', 'requesting column embeddings → ollama @ nvidia-server');
      for (let i = 1; i <= 20 && !cancelled; i++) {
        setProgress(i / 20);
        if (i === 5) addLog('info', '[A] columns: mark, model, year, mileage, color...');
        if (i === 12) addLog('info', '[B] columns: brand, model_name, year, probeg_km, color_hex...');
        if (i === 18) addLog('info', 'qwen3-embedding:8b → 4096-dim · 20 columns');
        await sleep(60);
      }
      addLog('ok', 'IDF-filter applied · 41 unique tokens retained (min_df=2)');

      // Phase 3 — graph build
      if (cancelled) return;
      setPhase('build');
      setProgress(0);
      addLog('info', 'building HeteroData: row+token nodes, token→row edges');
      for (let i = 1; i <= 30 && !cancelled; i++) {
        setProgress(i / 30);
        await sleep(35);
      }
      addLog('ok', 'graph ready · 27 row · 41 token · 108 edges · column_emb on edges');
      addLog('ok', 'persisted → data/graphs/bge-m3/v17_views/');
      if (!cancelled) setPhase('done');
    };
    run();
    return () => { cancelled = true; };
  }, []);

  // map phase to graph progress (0 = unclustered, 1 = clustered)
  // here we want the *unclustered* layout — graph is built but not trained yet
  const graphProgress = phase === 'done' ? 0.15 : Math.min(0.15, progress * 0.15);

  const phaseLabels = {
    idle: 'инициализация',
    embed: `строковые эмбеддинги · ${Math.round(progress * 100)}%`,
    tokenize: `column embeddings (Ollama) · ${Math.round(progress * 100)}%`,
    build: `построение HeteroData · ${Math.round(progress * 100)}%`,
    done: 'готово',
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Гетерогенный граф</h1>
          <p>Строки таблиц → row nodes (квадраты). Токены из значений ячеек → token nodes (круги). Рёбра token→row несут column embeddings (4096-dim, qwen3) как атрибуты. Этот граф — вход для GNN.</p>
        </div>
        <div className="actions">
          <div className="tabs">
            <div className={`tab ${showTokens ? 'active' : ''}`} onClick={() => setShowTokens(true)}>full hetero</div>
            <div className={`tab ${!showTokens ? 'active' : ''}`} onClick={() => setShowTokens(false)}>row-only</div>
          </div>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '1fr 320px', minHeight: 0 }}>
        {/* graph canvas */}
        <div style={{ position: 'relative', display: 'flex', flexDirection: 'column' }}>
          <HeteroGraph
            progress={graphProgress}
            showTokens={showTokens}
            showEdges={true}
            selected={selectedRow}
            onSelectRow={setSelectedRow}
            highlightClusters={false}
          />

          {/* phase HUD */}
          <div className="overlay-card" style={{ top: 14, left: 14, width: 280 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', marginBottom: 6 }}>PIPELINE</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                ['embed', 'TokenEmbedder · bge-m3'],
                ['tokenize', 'Ollama qwen3-embedding · column'],
                ['build', 'HeteroData (row + token + edges)'],
              ].map(([k, label]) => {
                const isCur = phase === k;
                const isDone = (phase === 'done') ||
                  (phase === 'build' && (k === 'embed' || k === 'tokenize')) ||
                  (phase === 'tokenize' && k === 'embed');
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      width: 14, height: 14, borderRadius: 3,
                      background: isDone ? 'var(--cluster)' : (isCur ? 'var(--row)' : 'var(--surface-2)'),
                      display: 'grid', placeItems: 'center',
                      fontFamily: 'var(--font-mono)', fontSize: 9, color: 'oklch(0.16 0.005 250)',
                    }}>{isDone ? '✓' : (isCur ? <span className="spinner" style={{ width: 8, height: 8, borderWidth: 1.2 }}></span> : '')}</div>
                    <div style={{ flex: 1, fontSize: 11.5, color: isDone || isCur ? 'var(--text)' : 'var(--text-4)' }}>{label}</div>
                  </div>
                );
              })}
            </div>
            {phase !== 'done' && (
              <div style={{ marginTop: 10, height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
              </div>
            )}
          </div>

          {/* selected node panel — draggable */}
          {selectedRow && (
            <DraggableRowPanel
              rowId={selectedRow}
              onClose={() => setSelectedRow(null)}
            />
          )}
        </div>

        {/* right panel — stats + logs */}
        <div style={{
          borderLeft: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', minHeight: 0,
        }}>
          <div style={{ padding: '16px 16px 12px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>Узлы графа</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div className="metric"><div className="k">row</div><div className="v row">27</div><div className="delta">A: 14 · B: 13</div></div>
              <div className="metric"><div className="k">token</div><div className="v token">41</div><div className="delta">idf ≥ {idfMin}</div></div>
              <div className="metric"><div className="k">edges</div><div className="v">108</div><div className="delta">token → row</div></div>
              <div className="metric"><div className="k">col_dim</div><div className="v">4096</div><div className="delta">qwen3-emb</div></div>
            </div>
          </div>

          <div style={{ padding: '0 16px 12px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>IDF token filter</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <span>min_df</span>
              <input type="range" min={1} max={6} step={1} value={idfMin}
                onChange={(e) => setIdfMin(parseInt(e.target.value))} className="range" />
              <b>{idfMin}</b>
            </div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', marginTop: 4 }}>
              отсекает шумные одноразовые токены (опечатки, артикулы)
            </div>
          </div>

          <div style={{ flex: 1, borderTop: '1px solid var(--border)', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '8px 16px', fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, borderBottom: '1px solid var(--border)' }}>
              build log
            </div>
            <div className="logs" ref={(el) => el && (el.scrollTop = el.scrollHeight)}>
              {logs.map((l, i) => (
                <div key={i} className="l">
                  <span className="t">{l.t}</span>
                  <span className={`lvl-${l.level}`}>{l.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <ScreenFooter
        onBack={onBack}
        onNext={onContinue}
        nextLabel="Запустить GNN"
        nextDisabled={phase !== 'done'}>
        {phase === 'done'
          ? <>Граф построен. Дальше: 2-слойный GNN/GAT соберёт row embeddings через message passing по этим рёбрам.</>
          : <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp; {phaseLabels[phase]}</>}
      </ScreenFooter>
    </div>
  );
}

function RowDetail({ rowId }) {
  const D = window.__DATA__;
  const tbl = rowId[0] === 'A' ? D.tableA : D.tableB;
  const idx = parseInt(rowId.slice(1), 10);
  const row = tbl.data[idx];
  if (!row) return null;
  const tokens = ROW_TOKENS[rowId] || [];
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, color: 'var(--text)' }}>
        <span className="mono" style={{ color: 'var(--text-3)', fontSize: 11, marginRight: 6 }}>{tbl.name}</span>
        · #{idx + 1}
      </div>
      {/* mini horizontal table: header row + single data row */}
      <div style={{
        border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden', marginBottom: 10,
      }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="dt" style={{ minWidth: '100%' }}>
            <thead>
              <tr>
                {tbl.cols.map((c) => <th key={c}>{c}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr>
                {row.map((v, j) => {
                  const col = tbl.cols[j];
                  if (col === 'color_hex' || col === 'color') {
                    return <td key={j}><ColorSwatch value={String(v)} /></td>;
                  }
                  return <td key={j}>{String(v)}</td>;
                })}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-4)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.04 }}>
          Связанные токены · {tokens.length}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {tokens.map((ti) => (
            <span key={ti} className="chip token"><span className="dot"></span>{TOKENS[ti]}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- Draggable row detail panel ----
function DraggableRowPanel({ rowId, onClose }) {
  const [pos, setPos] = useState({ right: 14, top: 14, left: null });
  const [dragging, setDragging] = useState(null);
  const ref = useRef(null);

  const onMouseDown = (e) => {
    if (e.target.closest('.no-drag')) return;
    const rect = ref.current.getBoundingClientRect();
    // capture current absolute position and switch to left/top mode
    const parent = ref.current.offsetParent.getBoundingClientRect();
    setPos({ left: rect.left - parent.left, top: rect.top - parent.top, right: null });
    setDragging({
      offsetX: e.clientX - rect.left,
      offsetY: e.clientY - rect.top,
    });
    e.preventDefault();
  };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => {
      const parent = ref.current.offsetParent.getBoundingClientRect();
      const newLeft = e.clientX - parent.left - dragging.offsetX;
      const newTop  = e.clientY - parent.top  - dragging.offsetY;
      // clamp inside parent
      const W = parent.width, H = parent.height;
      const rect = ref.current.getBoundingClientRect();
      const clampedLeft = Math.max(0, Math.min(W - rect.width, newLeft));
      const clampedTop  = Math.max(0, Math.min(H - rect.height, newTop));
      setPos({ left: clampedLeft, top: clampedTop, right: null });
    };
    const onUp = () => setDragging(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging]);

  const styleObj = {
    width: 480,
    top: pos.top,
    cursor: dragging ? 'grabbing' : 'default',
    userSelect: dragging ? 'none' : undefined,
  };
  if (pos.right != null) styleObj.right = pos.right;
  if (pos.left  != null) styleObj.left  = pos.left;

  return (
    <div ref={ref} className="overlay-card" style={styleObj}>
      {/* drag handle row */}
      <div
        onMouseDown={onMouseDown}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: 10, padding: '2px 0',
          cursor: dragging ? 'grabbing' : 'grab',
        }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* drag grip */}
          <svg width="10" height="14" viewBox="0 0 10 14" style={{ opacity: 0.5 }}>
            <circle cx="2" cy="3"  r="1" fill="var(--text-3)" />
            <circle cx="8" cy="3"  r="1" fill="var(--text-3)" />
            <circle cx="2" cy="7"  r="1" fill="var(--text-3)" />
            <circle cx="8" cy="7"  r="1" fill="var(--text-3)" />
            <circle cx="2" cy="11" r="1" fill="var(--text-3)" />
            <circle cx="8" cy="11" r="1" fill="var(--text-3)" />
          </svg>
          <span style={{
            display: 'inline-block', width: 10, height: 10, borderRadius: 2,
            background: 'var(--warn)',
            boxShadow: '0 0 6px var(--warn)',
          }}></span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)' }}>
            row · <span style={{ color: 'var(--warn)', fontWeight: 600 }}>{rowId}</span>
          </span>
        </div>
        <button className="btn ghost icon no-drag" style={{ width: 22, height: 22 }} onClick={onClose}>×</button>
      </div>
      <RowDetail rowId={rowId} />
    </div>
  );
}

Object.assign(window, { ScreenGraph });
