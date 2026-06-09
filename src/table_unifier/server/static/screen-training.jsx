// Screen 3 — Inference: run pre-trained GAT via real backend, stream phases via WS.

// Инференс уже отработан? (кандидаты лежат в памяти) — не пересчитываем при возврате.
function inferenceIsDone() {
  return ((window.__DATA__ && window.__DATA__.candidates) || []).length > 0;
}

function ScreenTraining({ onContinue, onBack, threshold }) {
  const _done = inferenceIsDone();
  const [phase, setPhase] = useState(_done ? 'done' : 'idle');
  const [progress, setProgress] = useState(0);
  const [pulsePhase, setPulsePhase] = useState(0);
  const [pulseLayer, setPulseLayer] = useState(null);
  const [graphProgress, setGraphProgress] = useState(_done ? 1 : 0);
  const [activeTab, setActiveTab] = useState('split');
  const [hovered, setHovered] = useState(null);
  const [logs, setLogs] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);
  const [metrics, setMetrics] = useState(() => _done ? ((window.__DATA__ && window.__DATA__.metrics) || {}) : {});
  const [bottomOpen, setBottomOpen] = useState(true);
  const wsRef = useRef(null);

  const addLog = (level, msg) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs((L) => [...L, { t: ts, level, msg }].slice(-200));
  };

  // pulse animation loop — работает только во время передачи сообщений (pulseLayer ≠ null),
  // чтобы готовый экран не перерисовывался каждый кадр впустую.
  useEffect(() => {
    if (pulseLayer === null) return;
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000; last = now;
      setPulsePhase((p) => (p + dt * 0.75) % 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pulseLayer]);

  const startInference = async (force = false) => {
    setPhase('idle'); setProgress(0); setGraphProgress(0); setPulseLayer(null);
    setLogs([]); setErrorMsg(null);
    const rid = window.__STATE__.runId;
    if (!rid) {
      setErrorMsg('Сессия потеряна (возможно, страница была перезагружена). '
                  + 'Вернитесь к загрузке таблиц и постройте граф заново.');
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
      }, { kind: 'infer' });
    } catch (e) {
      setErrorMsg(String(e.message || e));
      setPhase('error');
      addLog('err', String(e.message || e));
    }
  };

  useEffect(() => {
    // Если граф ещё не загружен в память (например, после перезагрузки страницы),
    // подтянем его из API чтобы отобразить в HeteroGraph и EmbeddingSpace.
    const rid = window.__STATE__.runId;
    if (rid && (!window.__DATA__.graph || !window.__DATA__.graph.rows)) {
      window.API.getGraph(rid).catch(() => {/* граф может быть ещё не готов */});
    }
  }, []);

  useEffect(() => {
    // Инференс уже отработан — при возврате на экран показываем результат
    // без перезапуска (явный перезапуск — кнопка «↻ перезапустить»).
    if (inferenceIsDone()) {
      addLog('ok', 'inference complete · результат из кэша');
      return;
    }
    startInference();
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const phaseInfo = {
    idle:    { label: 'инициализация',                  hint: '' },
    load:    { label: 'загрузка модели',                hint: 'state_dict + прогрев' },
    l1:      { label: 'прямой проход · слой 1 (токен→строка)',  hint: 'усреднение сообщений' },
    l2:      { label: 'прямой проход · слой 2 (строка→токен)',  hint: 'обновление эмбеддингов строк' },
    sim:     { label: 'матрица косинусного сходства',       hint: 'между таблицами A × B' },
    cluster: { label: 'компоненты связности',           hint: 'порог + union-find' },
    done:    { label: 'инференс завершён',              hint: `${metrics.n_pairs_found || metrics.n_pairs || 0} кандидатов` },
    error:   { label: 'ошибка', hint: errorMsg || '' },
  };

  // ---- тайминги стадий пайплайна + память графа ----
  // Поля приходят из metrics бэка (snake_case). Если бэк ещё не отдаёт стадию —
  // показываем «—». GAT-инференс fallback'ится на старое поле latency_ms.
  const graph = window.__DATA__.graph;
  const tColDesc = metrics.t_col_descriptions_ms;
  const tColEmb  = metrics.t_col_embeddings_ms;
  const tRowEmb  = metrics.t_row_embeddings_ms;
  const tGat     = metrics.t_gat_ms ?? metrics.latency_ms;
  const stageVals = [tColDesc, tColEmb, tRowEmb, tGat];
  const tTotal = metrics.t_total_ms ??
    (stageVals.every((x) => typeof x === 'number') ? stageVals.reduce((a, b) => a + b, 0) : undefined);
  const graphMemMb = metrics.graph_mem_mb ??
    (typeof metrics.graph_bytes === 'number' ? metrics.graph_bytes / 1048576 : undefined);

  // Эффективный порог: значение слайдера из окна «Проверка» (App-state), а пока
  // пользователь его не трогал — дефолт из инференса. Управляет и гистограммой,
  // и цветом рёбер «найденных пар» в пространстве эмбеддингов.
  const effThreshold = (typeof threshold === 'number') ? threshold : (metrics.threshold ?? 0.831);
  const candidates = window.__DATA__.candidates || [];

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Инференс модели</h1>
          <p>Прямой проход предобученной GAT по гетерографу. Слева — структура гетерографа: узлы строк и токенов, толщина рёбер = вес внимания. Справа — UMAP-проекция эмбеддингов строк; рёбра соединяют найденные пары (зелёные — авто-слияние, жёлтые — на проверку, цвет зависит от порога из окна «Проверка»).</p>
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

      <div className="screen-body" style={{ display: 'grid', gridTemplateRows: 'minmax(0, 1fr) auto', minHeight: 0 }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: activeTab === 'split' ? '1fr 1fr 280px' : '1fr 280px',
          minHeight: 0,
          overflow: 'hidden',
        }}>
          {(activeTab === 'graph' || activeTab === 'split') && (
            <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden',
                         borderRight: activeTab === 'split' ? '1px solid var(--border)' : 'none' }}>
              <ViewLabel title="Структурный граф" subtitle="узлы строк и токенов · толщина = вес рёбер" />
              <HeteroGraph
                progress={graphProgress}
                pulseLayer={pulseLayer}
                pulsePhase={pulsePhase}
                showTokens={true}
                selected={hovered}
                onSelectRow={setHovered}
                highlightClusters={false}
                clusterLayout={false}
              />
            </div>
          )}
          {(activeTab === 'embed' || activeTab === 'split') && (
            <div style={{ display: 'flex', flexDirection: 'column', position: 'relative', minHeight: 0, overflow: 'hidden' }}>
              <ViewLabel title="Пространство эмбеддингов" subtitle="UMAP · 1024 → 2 · рёбра = найденные пары" />
              <div style={{ flex: 1, minHeight: 0, padding: '14px 16px 16px' }}>
                <EmbeddingSpace progress={graphProgress} hovered={hovered} onHover={setHovered}
                  dims="1024" candidates={candidates} threshold={effThreshold} />
              </div>
            </div>
          )}

          <div style={{ borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', minHeight: 0, overflowY: 'auto' }}>
            <div style={{ padding: '16px 16px 12px' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>
                этапы инференса
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[['load', 'загрузка модели'],
                  ['l1', 'GATv2Conv[0] · токен→строка'],
                  ['l2', 'GATv2Conv[1] · строка→токен'],
                  ['sim', 'косинусное сходство'],
                  ['cluster', 'порог по компонентам']].map(([k, label]) => {
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
                        fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-accent)',
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

            <div style={{ flexShrink: 0, height: 190, borderTop: '1px solid var(--border)', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '8px 16px', fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, borderBottom: '1px solid var(--border)' }}>
                журнал инференса
              </div>
              <div className="logs" ref={(el) => el && (el.scrollTop = el.scrollHeight)}>
                {logs.map((l, i) => (
                  <div key={i} className="l"><span className="t">{l.t}</span><span className={`lvl-${l.level}`}>{l.msg}</span></div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--border)' }}>
          <button className={`strip-handle ${bottomOpen ? 'open' : ''}`}
            onClick={() => setBottomOpen((o) => !o)}
            title={bottomOpen ? 'Свернуть вниз' : 'Развернуть'}>
            <span>{phase === 'done' ? 'результат инференса' : 'распределение · ход'}</span>
            {phase === 'done' && !bottomOpen && (
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cluster)', textTransform: 'none', letterSpacing: 0 }}>
                {metrics.n_pairs_found ?? 0} пар · {metrics.n_clusters ?? 0} кл.
              </span>
            )}
            <span className="chev">
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3.5 5.5L7 9l3.5-3.5" /></svg>
            </span>
          </button>
          {bottomOpen && (
            <div style={{
              padding: '12px 16px 14px',
              display: 'grid',
              gridTemplateColumns: phase === 'done' ? 'minmax(240px, 1fr) auto auto' : '1fr 360px',
              gap: 24, alignItems: 'start',
            }}>
              <SimHistogram phase={phase} bins={window.__DATA__.histogram} threshold={effThreshold} />

              {phase === 'done' ? (
                <React.Fragment>
                  <div>
                    <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>результат</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div className="metric"><div className="k">пар найдено</div><div className="v cluster">{metrics.n_pairs_found ?? '—'}</div><div className="delta">сходство ≥ {Number(metrics.threshold || 0).toFixed(2)}</div></div>
                      <div className="metric"><div className="k">кластеров</div><div className="v">{metrics.n_clusters ?? '—'}</div><div className="delta">комп. связн.</div></div>
                      <div className="metric"><div className="k">строк</div><div className="v">{metrics.n_input_rows ?? '—'}</div><div className="delta">A + B</div></div>
                      <div className="metric"><div className="k">память графа</div><div className="v">{fmtMem(graphMemMb)}</div><div className="delta">{graph ? `${graph.stats?.n_edges ?? '—'} рёбер` : 'в памяти'}</div></div>
                    </div>
                  </div>

                  <div style={{ minWidth: 248 }}>
                    <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>время выполнения</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                      <TimingRow label="описания столбцов" hint="LLM" value={fmtDur(tColDesc)} />
                      <TimingRow label="эмбеддинги столбцов" value={fmtDur(tColEmb)} />
                      <TimingRow label="эмбеддинги строк" value={fmtDur(tRowEmb)} />
                      <TimingRow label="инференс GAT" hint="прямой проход" value={fmtDur(tGat)} />
                      <div style={{ borderTop: '1px solid var(--border)', marginTop: 6, paddingTop: 6 }}>
                        <TimingRow label="итого" value={fmtDur(tTotal)} total />
                      </div>
                    </div>
                  </div>
                </React.Fragment>
              ) : (
                <div>
                  <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>ход инференса</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 500, color: phase === 'error' ? 'var(--bad)' : 'var(--text)' }}>
                    {phaseInfo[phase]?.label || '—'}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                    {phaseInfo[phase]?.hint}
                  </div>
                </div>
              )}
            </div>
          )}
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

// ---- helpers for the result panel ----
function fmtDur(ms) {
  if (ms == null || isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} мс`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} с`;
}
function fmtMem(mb) {
  if (mb == null || isNaN(mb)) return '—';
  if (mb < 1) return `${Math.round(mb * 1024)} КБ`;
  if (mb < 1024) return `${mb.toFixed(1)} МБ`;
  return `${(mb / 1024).toFixed(2)} ГБ`;
}

function TimingRow({ label, hint, value, total }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, padding: '3px 0' }}>
      <span style={{ color: total ? 'var(--text)' : 'var(--text-3)', fontWeight: total ? 600 : 400 }}>{label}</span>
      {hint && <span style={{ color: 'var(--text-4)', fontSize: 10 }}>{hint}</span>}
      <span style={{ flex: 1, borderBottom: '1px dotted var(--border)', transform: 'translateY(-3px)', minWidth: 12 }}></span>
      <span style={{ color: total ? 'var(--row)' : 'var(--text)', fontWeight: total ? 600 : 500, whiteSpace: 'nowrap' }}>{value}</span>
    </div>
  );
}

function ViewLabel({ title, subtitle }) {
  return (
    <div style={{ padding: '12px 16px 10px', flexShrink: 0, borderBottom: '1px solid var(--border)', background: 'var(--bg-elev)' }}>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, fontFamily: 'var(--font-mono)' }}>
        {title}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{subtitle}</div>
    </div>
  );
}

function SimHistogram({ phase, bins, threshold }) {
  const wrapRef = useRef(null);
  const [w, setW] = useState(420);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // offsetWidth is layout px (immune to the --ui-zoom CSS zoom); getBoundingClientRect
    // would return zoom-scaled px and make the SVG overflow its column at high zoom.
    const measure = () => setW(el.offsetWidth);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  bins = (bins && bins.length === 20) ? bins : new Array(20).fill(0);
  const visible = phase === 'sim' || phase === 'cluster' || phase === 'done';
  const H = 92, padL = 4, padR = 4, padTop = 18, padBot = 16;
  const W = Math.max(140, w);
  const plotW = W - padL - padR;
  const plotTop = padTop, plotBot = H - padBot;
  const maxV = Math.max(1, ...bins);
  const barW = plotW / bins.length;
  const thr = (typeof threshold === 'number' && threshold > 0) ? threshold : 0.65;
  const thrX = padL + thr * plotW;
  const flip = thrX > W - 60;

  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>
        распределение попарного косинусного сходства
      </div>
      <div ref={wrapRef} style={{ width: '100%', maxWidth: 560 }}>
        <svg width={W} height={H} style={{ display: 'block' }}>
          {/* baseline */}
          <line x1={padL} x2={W - padR} y1={plotBot} y2={plotBot} stroke="var(--border)" strokeWidth="1" />
          {/* bars */}
          {bins.map((v, i) => {
            const h = visible ? (v / maxV) * (plotBot - plotTop) : 0;
            const x = padL + i * barW;
            const isMatch = (i + 0.5) / bins.length >= thr;
            return (
              <rect key={i} x={x + 0.5} y={plotBot - h} width={Math.max(1, barW - 1)} height={h}
                fill={isMatch ? 'var(--cluster)' : 'var(--row)'}
                opacity={visible ? (isMatch ? 0.9 : 0.5) : 0}
                style={{ transition: `all 0.3s ease ${i * 12}ms` }} />
            );
          })}
          {/* threshold line — drawn on top of the bars so it stays visible */}
          {visible && (
            <g>
              <line x1={thrX} x2={thrX} y1={plotTop - 8} y2={plotBot}
                stroke="var(--warn)" strokeWidth="1.5" strokeDasharray="3 2" />
              <text x={flip ? thrX - 5 : thrX + 5} y={plotTop - 1}
                textAnchor={flip ? 'end' : 'start'}
                fontSize="10" fontWeight="600" fontFamily="var(--font-mono)" fill="var(--warn)">
                порог {thr.toFixed(2)}
              </text>
            </g>
          )}
          {/* x-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
            <text key={v} x={padL + v * plotW} y={H - 3}
              fontSize="9" textAnchor={v === 0 ? 'start' : (v === 1 ? 'end' : 'middle')}
              fontFamily="var(--font-mono)" fill="var(--text-4)">
              {v.toFixed(2)}
            </text>
          ))}
        </svg>
      </div>
    </div>
  );
}

Object.assign(window, { ScreenTraining });
