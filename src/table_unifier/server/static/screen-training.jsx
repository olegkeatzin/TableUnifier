// Screen 3 — Inference: run pre-trained GAT via real backend, stream phases via WS.

function ScreenTraining({ onContinue, onBack }) {
  const [phase, setPhase] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [pulsePhase, setPulsePhase] = useState(0);
  const [pulseLayer, setPulseLayer] = useState(null);
  const [graphProgress, setGraphProgress] = useState(0);
  const [activeTab, setActiveTab] = useState('split');
  const [hovered, setHovered] = useState(null);
  const [logs, setLogs] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);
  const [metrics, setMetrics] = useState({});
  const wsRef = useRef(null);

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

  const startInference = async (force = false) => {
    setPhase('idle'); setProgress(0); setGraphProgress(0); setPulseLayer(null);
    setLogs([]); setErrorMsg(null);
    const rid = window.__STATE__.runId;
    if (!rid) {
      setErrorMsg('no run_id — need to build graph first');
      setPhase('error');
      return;
    }
    try {
      // Не плодим повторных вызовов /api/infer/run при навигации назад/вперёд.
      // Если inference уже стартовал на этом run — просто переподписываемся
      // на WS, buffered события догонят состояние.
      if (force || window.__STATE__.inferRunId !== rid) {
        await window.API.runInference({ runId: rid });
        window.__STATE__.inferRunId = rid;
      } else {
        addLog('info', `reattaching to inference on ${rid}`);
      }
      wsRef.current = window.API.subscribeRun(rid, (ev) => {
        if (ev.type === 'phase') {
          setPhase(ev.phase);
          setProgress(0);
          if (ev.phase === 'l1') { setPulseLayer(0); setGraphProgress(0.4); }
          else if (ev.phase === 'l2') { setPulseLayer(1); setGraphProgress(0.75); }
          else if (ev.phase === 'sim') { setPulseLayer(null); setGraphProgress(0.85); }
          else if (ev.phase === 'cluster') { setGraphProgress(0.95); }
          addLog('info', ev.label || ev.phase);
        } else if (ev.type === 'progress') {
          if (typeof ev.progress === 'number') setProgress(ev.progress);
        } else if (ev.type === 'log') {
          addLog(ev.level, ev.msg);
        } else if (ev.type === 'metric') {
          setMetrics((m) => ({ ...m, [ev.key]: ev.value }));
        } else if (ev.type === 'done') {
          setPhase('done'); setGraphProgress(1); setPulseLayer(null);
          addLog('ok', 'inference complete');
          // populate __DATA__ for downstream screens
          window.API.getClusters(rid).then((res) => {
            setMetrics((m) => ({ ...m, ...res.metrics }));
            window.__STATE__.inferDone = true;
          }).catch((e) => addLog('err', String(e.message || e)));
        } else if (ev.type === 'error') {
          setErrorMsg(ev.msg);
          setPhase('error');
          addLog('err', ev.msg);
        }
      });
    } catch (e) {
      setErrorMsg(String(e.message || e));
      setPhase('error');
      addLog('err', String(e.message || e));
    }
  };

  useEffect(() => {
    startInference();
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const phaseInfo = {
    idle:    { label: 'инициализация',                  hint: '' },
    load:    { label: 'загрузка модели',                hint: 'state_dict + warmup' },
    l1:      { label: 'forward · layer 1 (token→row)',  hint: 'mean-aggregate сообщений' },
    l2:      { label: 'forward · layer 2 (row→token)',  hint: 'обновление row embeddings' },
    sim:     { label: 'cosine similarity matrix',       hint: 'cross-table A × B' },
    cluster: { label: 'connected components',           hint: 'threshold + union-find' },
    done:    { label: 'инференс завершён',              hint: `${metrics.n_pairs_found || metrics.n_pairs || 0} кандидатов` },
    error:   { label: 'ошибка', hint: errorMsg || '' },
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Инференс модели</h1>
          <p>Forward pass предобученной GAT по гетерографу. Слева — структура графа с пульсацией message passing, справа — UMAP-проекция row-эмбеддингов.</p>
        </div>
        <div className="actions">
          <button className="btn ghost" onClick={() => startInference(true)}>↻ перезапустить</button>
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
          {(activeTab === 'graph' || activeTab === 'split') && (
            <div style={{ position: 'relative', display: 'flex', flexDirection: 'column',
                         borderRight: activeTab === 'split' ? '1px solid var(--border)' : 'none' }}>
              <ViewLabel title="Структурный граф" subtitle="row + token nodes · message passing" />
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
          {(activeTab === 'embed' || activeTab === 'split') && (
            <div style={{ padding: '14px 16px 16px', display: 'flex', flexDirection: 'column', gap: 10, position: 'relative' }}>
              <ViewLabel title="Embedding space" subtitle="UMAP · 1024 → 2" />
              <div style={{ flex: 1, minHeight: 0, marginTop: 24 }}>
                <EmbeddingSpace progress={graphProgress} hovered={hovered} onHover={setHovered} dims="1024" />
              </div>
            </div>
          )}

          <div style={{ borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div style={{ padding: '16px 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>
                inference pipeline
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[['load', 'load checkpoint'],
                  ['l1', 'GATv2Conv[0] · token→row'],
                  ['l2', 'GATv2Conv[1] · row→token'],
                  ['sim', 'cosine similarity'],
                  ['cluster', 'CC threshold']].map(([k, label]) => {
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
              {phase !== 'done' && phase !== 'idle' && phase !== 'error' && (
                <div style={{ marginTop: 10, height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
                </div>
              )}
            </div>

            <div style={{ padding: '0 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>Архитектура</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
                <div><span style={{ color: 'var(--text-4)' }}>model:</span> EntityResolutionGAT</div>
                <div><span style={{ color: 'var(--text-4)' }}>layers:</span> 2 × GATv2Conv (heads=4)</div>
                <div><span style={{ color: 'var(--text-4)' }}>row_dim:</span> 1024 (bge-m3)</div>
                <div><span style={{ color: 'var(--text-4)' }}>edge_attr:</span> 1024 (col_emb MRL)</div>
                <div><span style={{ color: 'var(--text-4)' }}>checkpoint:</span> v14_mrl_gat_model.pt</div>
              </div>
            </div>

            <div style={{ padding: '0 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>message passing</div>
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

        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '12px 16px 14px',
          display: 'grid', gridTemplateColumns: '1fr 360px', gap: 20, alignItems: 'center',
        }}>
          <SimHistogram phase={phase} bins={window.__DATA__.histogram} threshold={metrics.threshold} />
          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>
              {phase === 'done' ? 'результат инференса' : 'inference progress'}
            </div>
            {phase === 'done' ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div className="metric"><div className="k">пар найдено</div><div className="v cluster">{metrics.n_pairs_found ?? '—'}</div><div className="delta">sim ≥ {Number(metrics.threshold || 0).toFixed(2)}</div></div>
                <div className="metric"><div className="k">кластеров</div><div className="v">{metrics.n_clusters ?? '—'}</div><div className="delta">CC</div></div>
                <div className="metric"><div className="k">rows</div><div className="v">{metrics.n_input_rows ?? '—'}</div><div className="delta">A + B</div></div>
                <div className="metric"><div className="k">latency</div><div className="v">{metrics.latency_ms ?? '—'}<span style={{ fontSize: 12, color: 'var(--text-3)' }}>ms</span></div><div className="delta">forward</div></div>
              </div>
            ) : (
              <>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 500, color: phase === 'error' ? 'var(--bad)' : 'var(--text)' }}>
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
          ? <>Инференс завершён · <b style={{ color: 'var(--cluster)' }}>{metrics.n_pairs_found ?? 0} пар</b></>
          : phase === 'error'
            ? <span style={{ color: 'var(--bad)' }}>Ошибка: {errorMsg}</span>
            : <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp; {phaseInfo[phase]?.label || 'старт…'}</>}
      </ScreenFooter>
    </div>
  );
}

function LayerDiagram({ pulseLayer }) {
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
      </g>
      <g>
        <rect x="190" y="18" width="60" height="20" rx="4" fill="var(--row-soft)" stroke="var(--row)" />
        <text x="220" y="32" textAnchor="middle" fontSize="11" fontFamily="var(--font-mono)" fill="var(--row)">row</text>
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

function InferenceHud({ phase, progress, phaseInfo }) {
  if (phase === 'idle') return null;
  return (
    <div className="overlay-card" style={{ top: 56, left: 14, width: 240 }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', marginBottom: 4 }}>PHASE</div>
      <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 2 }}>{phaseInfo[phase]?.label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-4)', marginBottom: 8 }}>
        {phaseInfo[phase]?.hint}
      </div>
      {phase !== 'done' && phase !== 'error' && (
        <div style={{ height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
        </div>
      )}
    </div>
  );
}

function ViewLabel({ title, subtitle }) {
  return (
    <div style={{ position: 'absolute', top: 14, left: 16, zIndex: 2, pointerEvents: 'none' }}>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, fontFamily: 'var(--font-mono)' }}>
        {title}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{subtitle}</div>
    </div>
  );
}

function SimHistogram({ phase, bins, threshold }) {
  bins = (bins && bins.length === 20) ? bins : new Array(20).fill(0);
  const visible = phase === 'sim' || phase === 'cluster' || phase === 'done';
  const W = 800, H = 76, M = 18;
  const maxV = Math.max(1, ...bins);
  const barW = (W - 2 * M) / bins.length;
  const thrIdx = Math.round((threshold || 0.65) * 20);
  return (
    <div>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 4 }}>
        distribution of pairwise cosine similarity
      </div>
      <svg width="100%" height={H + 18} viewBox={`0 0 ${W} ${H + 18}`} preserveAspectRatio="none" style={{ maxWidth: '100%' }}>
        {visible && (
          <g>
            <line x1={M + barW * thrIdx} x2={M + barW * thrIdx} y1={4} y2={H} stroke="var(--warn)" strokeWidth="1" strokeDasharray="3 2" />
            <text x={M + barW * thrIdx + 4} y={12} fontSize="9" fontFamily="var(--font-mono)" fill="var(--warn)">thr={Number(threshold || 0).toFixed(2)}</text>
          </g>
        )}
        {bins.map((v, i) => {
          const h = visible ? (v / maxV) * (H - 8) : 0;
          const x = M + i * barW;
          const isMatch = i >= thrIdx;
          return (
            <rect key={i} x={x + 1} y={H - h} width={barW - 2} height={h}
              fill={isMatch ? 'var(--cluster)' : 'var(--row)'}
              opacity={visible ? (isMatch ? 0.9 : 0.5) : 0}
              style={{ transition: `all 0.3s ease ${i * 12}ms` }} />
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
          <text key={v} x={M + v * (W - 2 * M)} y={H + 12}
            fontSize="9" textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--text-4)">
            {v.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  );
}

Object.assign(window, { ScreenTraining });
