// Screen 3 — Inference: run pre-trained model on uploaded data
//   Phases: load checkpoint → forward pass (L1: token→row) → forward pass (L2: row→token)
//          → similarity matrix → threshold + connected components → done.

function ScreenTraining({ onContinue, onBack }) {
  // phase: idle | load | l1 | l2 | sim | cluster | done
  const [phase, setPhase] = useState('idle');
  const [progress, setProgress] = useState(0);        // 0..1 within phase
  const [pulsePhase, setPulsePhase] = useState(0);    // animation phase along edges
  const [pulseLayer, setPulseLayer] = useState(null); // 0 | 1 | null
  const [graphProgress, setGraphProgress] = useState(0);
  const [running, setRunning] = useState(true);
  const [activeTab, setActiveTab] = useState('split');
  const [hovered, setHovered] = useState(null);
  const [logs, setLogs] = useState([]);

  const addLog = (level, msg) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs((L) => [...L, { t: ts, level, msg }].slice(-200));
  };

  // pulse animation loop
  useEffect(() => {
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000; last = now;
      setPulsePhase((p) => (p + dt * 0.75) % 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // inference sequence — runs once when "running" is true
  useEffect(() => {
    if (!running) return;
    let cancelled = false;

    const run = async () => {
      // PHASE 1 — load checkpoint
      setPhase('load');
      setProgress(0);
      addLog('info', 'loading checkpoint → output/bge-m3/v17_views_gat_model.pt');
      for (let i = 1; i <= 12 && !cancelled; i++) {
        setProgress(i / 12);
        if (i === 6) addLog('info', 'EntityResolutionGAT · 2 × GATv2Conv · heads=4');
        if (i === 10) addLog('info', 'loaded state_dict · 4.1M params · device=cuda:0');
        await sleep(70);
      }
      addLog('ok', 'model ready (best val_f1=0.913 @ epoch 64)');
      if (cancelled) return;

      // PHASE 2 — forward pass layer 1: token → row
      setPhase('l1'); setProgress(0); setPulseLayer(0);
      addLog('info', 'forward → GATv2Conv[0] · token → row aggregation');
      for (let i = 1; i <= 20 && !cancelled; i++) {
        setProgress(i / 20);
        setGraphProgress(0.05 + (i / 20) * 0.35);
        if (i === 10) addLog('info', 'edge_attr (col_emb 4096-d) · mean-aggregated · attn heads=4');
        await sleep(60);
      }
      addLog('ok', 'layer 1 complete · 27 row embeddings · 1024-d');
      if (cancelled) return;

      // PHASE 3 — forward pass layer 2: row → token
      setPhase('l2'); setProgress(0); setPulseLayer(1);
      addLog('info', 'forward → GATv2Conv[1] · row → token + residual');
      for (let i = 1; i <= 20 && !cancelled; i++) {
        setProgress(i / 20);
        setGraphProgress(0.40 + (i / 20) * 0.35);
        if (i === 12) addLog('info', 'GraphNorm + dropout 0.1');
        await sleep(60);
      }
      addLog('ok', 'layer 2 complete · row embeddings finalized');
      if (cancelled) return;

      // PHASE 4 — pairwise similarity
      setPhase('sim'); setProgress(0); setPulseLayer(null);
      addLog('info', 'computing pairwise cosine similarity (14 × 13)');
      for (let i = 1; i <= 16 && !cancelled; i++) {
        setProgress(i / 16);
        setGraphProgress(0.75 + (i / 16) * 0.10);
        await sleep(50);
      }
      addLog('ok', '182 pairs · 9 over threshold (sim ≥ 0.65)');
      if (cancelled) return;

      // PHASE 5 — clustering
      setPhase('cluster'); setProgress(0);
      addLog('info', 'connected components · GA-tuned threshold (gen=40, F1=0.913)');
      for (let i = 1; i <= 18 && !cancelled; i++) {
        setProgress(i / 18);
        setGraphProgress(0.85 + (i / 18) * 0.15);
        if (i === 10) addLog('info', '18 clusters formed · 9 multi-row, 9 singletons');
        await sleep(55);
      }
      addLog('ok', 'inference complete · 9 candidate pairs surfaced');
      addLog('ok', '3 require manual review (sim < 0.90)');

      if (cancelled) return;
      setPhase('done');
      setRunning(false);
      setGraphProgress(0.95);
    };

    run();
    return () => { cancelled = true; };
  }, []); // run once on mount

  const phaseInfo = {
    idle:    { label: 'инициализация',                  hint: '' },
    load:    { label: 'загрузка модели',                hint: 'state_dict + warmup' },
    l1:      { label: 'forward · layer 1 (token→row)',  hint: 'mean-aggregate сообщений' },
    l2:      { label: 'forward · layer 2 (row→token)',  hint: 'обновление row embeddings' },
    sim:     { label: 'cosine similarity matrix',       hint: '14 × 13 = 182 пар' },
    cluster: { label: 'connected components + GA',     hint: 'threshold по cosine sim · union-find' },
    done:    { label: 'инференс завершён',              hint: '9 кандидатов · 3 на ревью' },
  };

  // final metrics shown when done
  const f1 = phase === 'done' ? 0.913 : 0;

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Инференс модели</h1>
          <p><b style={{ color: 'var(--text-2)' }}>Слева</b> — структура графа: фиксированные row + token узлы, пульсация по рёбрам показывает message passing. <b style={{ color: 'var(--text-2)' }}>Справа</b> — embedding space (UMAP-проекция 1024→Н): row-векторы сходятся к кластерам по мере forward pass. Наведите на любой узел/точку — подсветятся кластер-mates и связанные токены.</p>
        </div>
        <div className="actions">
          <button className="btn ghost" onClick={() => {
            // restart inference
            setPhase('idle'); setProgress(0); setGraphProgress(0);
            setPulseLayer(null); setLogs([]); setRunning(true);
            setTimeout(() => setRunning(true), 50);
          }}>↻ перезапустить</button>
          <Tabs
            active={activeTab}
            setActive={setActiveTab}
            tabs={[
              { key: 'graph', label: 'Граф' },
              { key: 'split', label: 'Граф + Эмбеддинги' },
              { key: 'embed', label: 'Эмбеддинги' },
            ]}
          />
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateRows: '1fr auto', minHeight: 0 }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: activeTab === 'split' ? '1fr 1fr 280px' : '1fr 280px',
          minHeight: 0,
        }}>
          {/* main view */}
          {activeTab === 'graph' && (
            <div style={{ position: 'relative', display: 'flex', flexDirection: 'column' }}>
              <ViewLabel title="Структурный граф" subtitle="row + token nodes · message passing по рёбрам" />
              <HeteroGraph
                progress={graphProgress}
                pulseLayer={pulseLayer}
                pulsePhase={pulsePhase}
                showTokens={true}
                selected={hovered}
                onSelectRow={setHovered}
                highlightClusters={false}
              />
              <InferenceHud phase={phase} progress={progress} pulseLayer={pulseLayer} phaseInfo={phaseInfo} />
            </div>
          )}
          {activeTab === 'embed' && (
            <div style={{ padding: '14px 16px 16px', display: 'flex', flexDirection: 'column', gap: 10, position: 'relative' }}>
              <ViewLabel title="Embedding space" subtitle="UMAP · 1024 → 2 · результат forward pass" />
              <div style={{ flex: 1, minHeight: 0, marginTop: 24 }}>
                <EmbeddingSpace progress={graphProgress} hovered={hovered} onHover={setHovered} dims="1024" />
              </div>
            </div>
          )}
          {activeTab === 'split' && (
            <>
              <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--border)' }}>
                <ViewLabel title="Структурный граф" subtitle="row + token nodes · пульсации = message passing" />
                <HeteroGraph
                  progress={graphProgress}
                  pulseLayer={pulseLayer}
                  pulsePhase={pulsePhase}
                  showTokens={true}
                  selected={hovered}
                  onSelectRow={setHovered}
                  highlightClusters={false}
                />
                <InferenceHud phase={phase} progress={progress} pulseLayer={pulseLayer} phaseInfo={phaseInfo} />
              </div>
              <div style={{ padding: '14px 16px 16px', display: 'flex', flexDirection: 'column', gap: 10, position: 'relative' }}>
                <ViewLabel title="Embedding space" subtitle="UMAP · 1024 → 2 · результат forward pass" />
                <div style={{ flex: 1, minHeight: 0, marginTop: 24 }}>
                  <EmbeddingSpace progress={graphProgress} hovered={hovered} onHover={setHovered} dims="1024" />
                </div>
                <div style={{ fontSize: 10.5, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
                  каждая точка — строка таблицы (<span style={{ color: 'var(--row)' }}>■ A</span> · <span style={{ color: 'var(--token)' }}>■ B</span>). Дубликаты сходятся в одну точку.
                </div>
              </div>
            </>
          )}

          {/* right panel — phase progress + architecture */}
          <div style={{
            borderLeft: '1px solid var(--border)',
            display: 'flex', flexDirection: 'column', minHeight: 0,
          }}>
            <div style={{ padding: '16px 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>
                inference pipeline
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  ['load',    'load checkpoint'],
                  ['l1',      'GATv2Conv[0] · token→row'],
                  ['l2',      'GATv2Conv[1] · row→token'],
                  ['sim',     'cosine similarity'],
                  ['cluster', 'CC + GA-threshold'],
                ].map(([k, label]) => {
                  const order = ['load','l1','l2','sim','cluster','done'];
                  const curIdx = order.indexOf(phase);
                  const myIdx = order.indexOf(k);
                  const isCur = phase === k;
                  const isDone = myIdx < curIdx || phase === 'done';
                  return (
                    <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 14, height: 14, borderRadius: 3,
                        background: isDone ? 'var(--cluster)' : (isCur ? 'var(--row)' : 'var(--surface-2)'),
                        display: 'grid', placeItems: 'center',
                        fontFamily: 'var(--font-mono)', fontSize: 9, color: 'oklch(0.16 0.005 250)',
                      }}>{isDone ? '✓' : (isCur ? <span className="spinner" style={{ width: 8, height: 8, borderWidth: 1.2 }}></span> : '')}</div>
                      <div style={{ flex: 1, fontSize: 11.5, color: (isDone || isCur) ? 'var(--text)' : 'var(--text-4)' }}>{label}</div>
                    </div>
                  );
                })}
              </div>
              {phase !== 'done' && phase !== 'idle' && (
                <div style={{ marginTop: 10, height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
                </div>
              )}
            </div>

            <div style={{ padding: '0 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>
                Архитектура
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
                <div><span style={{ color: 'var(--text-4)' }}>model:</span> EntityResolutionGAT</div>
                <div><span style={{ color: 'var(--text-4)' }}>layers:</span> 2 × GATv2Conv (heads=4)</div>
                <div><span style={{ color: 'var(--text-4)' }}>row_dim:</span> 1024 (bge-m3)</div>
                <div><span style={{ color: 'var(--text-4)' }}>edge_attr:</span> 4096 (col_emb)</div>
                <div><span style={{ color: 'var(--text-4)' }}>params:</span> 4.1M · fp16</div>
              </div>
            </div>

            <div style={{ padding: '0 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>
                message passing
              </div>
              <LayerDiagram pulseLayer={pulseLayer} running={pulseLayer !== null} />
            </div>

            <div style={{ flex: 1, borderTop: '1px solid var(--border)', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '8px 16px', fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, borderBottom: '1px solid var(--border)' }}>
                inference log
              </div>
              <div className="logs" ref={(el) => el && (el.scrollTop = el.scrollHeight)}>
                {logs.map((l, i) => (
                  <div key={i} className="l"><span className="t">{l.t}</span><span className={`lvl-${l.level}`}>{l.msg}</span></div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* bottom — similarity histogram + threshold */}
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '12px 16px 14px',
          display: 'grid', gridTemplateColumns: '1fr 360px', gap: 20, alignItems: 'center',
        }}>
          <SimHistogram phase={phase} />
          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>
              {phase === 'done' ? 'результат инференса' : 'inference progress'}
            </div>
            {phase === 'done' ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div className="metric"><div className="k">пар найдено</div><div className="v cluster">9</div><div className="delta">sim ≥ 0.65</div></div>
                <div className="metric"><div className="k">кластеров</div><div className="v">18</div><div className="delta">CC · thr=0.83</div></div>
                <div className="metric"><div className="k">F1 (val)</div><div className="v">0.913</div><div className="delta">v17_views</div></div>
                <div className="metric"><div className="k">latency</div><div className="v">312<span style={{ fontSize: 12, color: 'var(--text-3)' }}>ms</span></div><div className="delta">end-to-end</div></div>
              </div>
            ) : (
              <>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 500, color: 'var(--text)' }}>
                  {phaseInfo[phase]?.label || '—'}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                  {phaseInfo[phase]?.hint}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <ScreenFooter
        onBack={onBack}
        onNext={onContinue}
        nextLabel="К ревью пар"
        nextDisabled={phase !== 'done'}>
        {phase === 'done'
          ? <>Инференс завершён · <b style={{ color: 'var(--cluster)' }}>9 пар-кандидатов</b> · <b style={{ color: 'var(--warn)' }}>3 на ручную проверку</b>.</>
          : <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp; {phaseInfo[phase]?.label || 'старт...'}</>}
      </ScreenFooter>
    </div>
  );
}

// ---- Mini layer diagram ----
function LayerDiagram({ pulseLayer, running }) {
  return (
    <svg width="100%" height="92" viewBox="0 0 260 92">
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--text-3)" />
        </marker>
      </defs>
      <g>
        <rect x="10" y="18" width="60" height="20" rx="4" fill="var(--token-soft)" stroke="var(--token)" />
        <text x="40" y="32" textAnchor="middle" fontSize="11" fontFamily="var(--font-mono)" fill="var(--token)">token</text>
        <rect x="10" y="54" width="60" height="20" rx="4" fill="var(--token-soft)" stroke="var(--token)" opacity="0.5" />
        <text x="40" y="68" textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--token)">h_t (1024)</text>
      </g>
      <g>
        <rect x="190" y="18" width="60" height="20" rx="4" fill="var(--row-soft)" stroke="var(--row)" />
        <text x="220" y="32" textAnchor="middle" fontSize="11" fontFamily="var(--font-mono)" fill="var(--row)">row</text>
        <rect x="190" y="54" width="60" height="20" rx="4" fill="var(--row-soft)" stroke="var(--row)" opacity="0.5" />
        <text x="220" y="68" textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--row)">h_r (1024)</text>
      </g>
      <g opacity={pulseLayer === 0 ? 1 : 0.3}>
        <path d="M 75 24 Q 130 8 185 24" fill="none" stroke="var(--token)" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="130" y="14" textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-3)">layer 1: token→row</text>
      </g>
      <g opacity={pulseLayer === 1 ? 1 : 0.3}>
        <path d="M 185 70 Q 130 88 75 70" fill="none" stroke="var(--row)" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="130" y="86" textAnchor="middle" fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-3)">layer 2: row→token</text>
      </g>
    </svg>
  );
}

// ---- Inference HUD ----
function InferenceHud({ phase, progress, pulseLayer, phaseInfo }) {
  if (phase === 'idle') return null;
  return (
    <div className="overlay-card" style={{ top: 56, left: 14, width: 240 }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', marginBottom: 4 }}>
        PHASE
      </div>
      <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 2 }}>
        {phaseInfo[phase]?.label}
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-4)', marginBottom: 8 }}>
        {phaseInfo[phase]?.hint}
      </div>
      {phase !== 'done' && (
        <div style={{ height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
        </div>
      )}
    </div>
  );
}

// ---- View label header on top of each panel ----
function ViewLabel({ title, subtitle }) {
  return (
    <div style={{
      position: 'absolute', top: 14, left: 16, zIndex: 2, pointerEvents: 'none',
    }}>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, fontFamily: 'var(--font-mono)' }}>
        {title}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
        {subtitle}
      </div>
    </div>
  );
}

// ---- Similarity histogram at bottom ----
function SimHistogram({ phase }) {
  // 20 bins from 0 to 1; pair counts populate as phase advances
  const bins = useMemo(() => {
    // realistic distribution: most pairs ~0.3-0.5 (random), some at 0.7-0.95 (matches)
    const dist = [0, 1, 3, 8, 18, 28, 32, 30, 22, 12, 6, 3, 2, 2, 3, 4, 5, 4, 3, 2];
    return dist;
  }, []);

  const visible = phase === 'sim' || phase === 'cluster' || phase === 'done';
  const W = 800, H = 76, M = 18;
  const maxV = Math.max(...bins);
  const barW = (W - 2 * M) / bins.length;

  return (
    <div>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 4 }}>
        distribution of pairwise cosine similarity {visible && <span style={{ color: 'var(--text-3)', textTransform: 'none', letterSpacing: 0 }}>· 182 pairs</span>}
      </div>
      <svg width="100%" height={H + 18} viewBox={`0 0 ${W} ${H + 18}`} preserveAspectRatio="none" style={{ maxWidth: '100%' }}>
        {/* threshold line */}
        {visible && (
          <g>
            <line x1={M + barW * 13} x2={M + barW * 13} y1={4} y2={H} stroke="var(--warn)" strokeWidth="1" strokeDasharray="3 2" />
            <text x={M + barW * 13 + 4} y={12} fontSize="9" fontFamily="var(--font-mono)" fill="var(--warn)">thr=0.65</text>
          </g>
        )}
        {/* bars */}
        {bins.map((v, i) => {
          const h = visible ? (v / maxV) * (H - 8) : 0;
          const x = M + i * barW;
          const y = H - h;
          const isMatch = i >= 13;
          return (
            <rect key={i}
              x={x + 1} y={y} width={barW - 2} height={h}
              fill={isMatch ? 'var(--cluster)' : 'var(--row)'}
              opacity={visible ? (isMatch ? 0.9 : 0.5) : 0}
              style={{ transition: `all 0.3s ease ${i * 12}ms` }} />
          );
        })}
        {/* x-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
          <text key={v}
            x={M + v * (W - 2 * M)} y={H + 12}
            fontSize="9" textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--text-4)">
            {v.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  );
}

Object.assign(window, { ScreenTraining });
