// Screen 2 — Build heterogeneous graph via real backend.

// Граф уже построен в памяти? Тогда при возврате на экран ничего не
// пересчитываем и не переигрываем анимацию — показываем готовое состояние.
function graphIsBuilt() {
  const g = window.__DATA__ && window.__DATA__.graph;
  return !!(g && g.rows && g.rows.length > 0 && g.stats);
}

function ScreenGraph({ onContinue, onBack }) {
  const _built = graphIsBuilt();
  const [phase, setPhase] = useState(_built ? 'done' : 'idle');
  const [progress, setProgress] = useState(_built ? 1 : 0);
  const [selectedRow, setSelectedRow] = useState(null);
  const [showTokens, setShowTokens] = useState(true);
  const [idfMin, setIdfMin] = useState(2);
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(() => {
    const g = window.__DATA__ && window.__DATA__.graph;
    return _built && g.stats
      ? { n_rows: g.stats.n_rows, n_tokens: g.stats.n_tokens, n_edges: g.stats.n_edges, col_dim: g.stats.col_dim || 1024 }
      : { n_rows: 0, n_tokens: 0, n_edges: 0, col_dim: 1024 };
  });
  const [errorMsg, setErrorMsg] = useState(null);
  const [showPipeline, setShowPipeline] = useState(true);
  const wsRef = useRef(null);

  const addLog = (level, msg) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs((L) => [...L, { t: ts, level, msg }].slice(-200));
  };

  useEffect(() => {
    // Граф уже в памяти — не запускаем сборку повторно при возврате на экран.
    if (graphIsBuilt()) {
      const g = window.__DATA__.graph;
      addLog('ok', `graph ready · ${g.stats.n_rows} row · ${g.stats.n_tokens} token · ${g.stats.n_edges} edges`);
      window.__STATE__.graphReady = true;
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        // Если для текущей сессии уже запущен билд — переподписываемся,
        // а не запускаем второй. Bus реплеит все буферизованные события,
        // так что прогресс восстановится с того места, где был.
        let runId = window.__STATE__.runId;
        if (runId) {
          addLog('info', `reattaching to existing run ${runId}`);
        } else {
          addLog('info', 'starting graph build · model=bge-m3 · target_col_dim=1024');
          const res = await window.API.buildGraph({ idfMinDf: idfMin });
          if (cancelled) return;
          runId = res.run_id;
        }
        wsRef.current = window.API.subscribeRun(runId, (ev) => {

          if (ev.type === 'phase') {
            setPhase(ev.phase);
            setProgress(0);
            addLog('info', `phase: ${ev.label || ev.phase}`);
          } else if (ev.type === 'progress') {
            if (typeof ev.progress === 'number') setProgress(ev.progress);
          } else if (ev.type === 'log') {
            addLog(ev.level, ev.msg);
          } else if (ev.type === 'graph_done') {
            setStats((s) => ({ ...s, n_rows: ev.n_rows, n_tokens: ev.n_tokens, n_edges: ev.n_edges }));
            setPhase('done'); setProgress(1);
            window.__STATE__.graphReady = true;
            addLog('ok', `graph ready · ${ev.n_rows} row · ${ev.n_tokens} token · ${ev.n_edges} edges`);
            // Подтягиваем реальные узлы/рёбра для визуализации.
            window.API.getGraph(runId).then((g) => {
              setStats((s) => ({ ...s, col_dim: g.stats?.col_dim || s.col_dim }));
            }).catch((e) => addLog('err', `graph fetch: ${e.message || e}`));
          } else if (ev.type === 'error') {
            setErrorMsg(ev.msg);
            setPhase('error');
            addLog('err', ev.msg);
          }
        }, { kind: 'build' });
      } catch (e) {
        setErrorMsg(String(e.message || e));
        setPhase('error');
        addLog('err', String(e.message || e));
      }
    })();
    return () => {
      cancelled = true;
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const graphProgress = phase === 'done' ? 0.15 : Math.min(0.15, progress * 0.15);
  const phaseLabels = {
    idle: 'инициализация',
    embed: `строковые эмбеддинги · ${Math.round(progress * 100)}%`,
    tokenize: `эмбеддинги столбцов (Ollama) · ${Math.round(progress * 100)}%`,
    build: `построение HeteroData · ${Math.round(progress * 100)}%`,
    done: 'готово',
    error: 'ошибка',
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Гетерогенный граф</h1>
          <p>Строки таблиц → узлы-строки, токены ячеек → узлы-токены. Рёбра токен→строка несут эмбеддинги столбцов (qwen3, MRL-truncated) как атрибуты.</p>
        </div>
        <div className="actions">
          <div className="tabs">
            <div className={`tab ${showTokens ? 'active' : ''}`} onClick={() => setShowTokens(true)}>полный граф</div>
            <div className={`tab ${!showTokens ? 'active' : ''}`} onClick={() => setShowTokens(false)}>только строки</div>
          </div>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '1fr 320px', minHeight: 0 }}>
        <div style={{ position: 'relative', display: 'flex', flexDirection: 'column' }}>
          <HeteroGraph
            progress={graphProgress}
            showTokens={showTokens}
            showEdges={true}
            selected={selectedRow}
            onSelectRow={setSelectedRow}
            highlightClusters={false}
          />

          {showPipeline ? (
          <div className="overlay-card" style={{ top: 14, left: 14, width: 280 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-3)', marginBottom: 6 }}>ЭТАПЫ</div>
              <button className="panel-x" title="Закрыть" aria-label="Закрыть" onClick={() => setShowPipeline(false)}>
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><path d="M3 3l6 6M9 3l-6 6" /></svg>
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                ['embed', 'TokenEmbedder · bge-m3'],
                ['tokenize', 'Ollama qwen3-embedding · столбцы'],
                ['build', 'HeteroData (строки + токены + рёбра)'],
              ].map(([k, label]) => {
                const order = ['embed', 'tokenize', 'build', 'done'];
                const myIdx = order.indexOf(k);
                const curIdx = order.indexOf(phase);
                const isCur = phase === k;
                const isDone = phase === 'done' || myIdx < curIdx;
                return (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      width: 14, height: 14, borderRadius: 3,
                      background: isDone ? 'var(--cluster)' : (isCur ? 'var(--row)' : 'var(--surface-2)'),
                      display: 'grid', placeItems: 'center',
                      fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-accent)',
                    }}>{isDone ? '✓' : (isCur ? <span className="spinner" style={{ width: 8, height: 8, borderWidth: 1.2 }}></span> : '')}</div>
                    <div style={{ flex: 1, fontSize: 11.5, color: isDone || isCur ? 'var(--text)' : 'var(--text-4)' }}>{label}</div>
                  </div>
                );
              })}
            </div>
            {phase !== 'done' && phase !== 'error' && (
              <div style={{ marginTop: 10, height: 3, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress * 100}%`, background: 'var(--row)', transition: 'width 0.15s' }}></div>
              </div>
            )}
          </div>
          ) : (
            <button className="restore-chip" style={{ top: 14, left: 14 }}
              onClick={() => setShowPipeline(true)} title="Показать pipeline">
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4"><circle cx="6" cy="6" r="5" /><path d="M6 5.2v3M6 3.6h.01" strokeLinecap="round" /></svg>
              показать этапы
            </button>
          )}
        </div>

        <div style={{ borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', minHeight: 0, overflowY: 'auto' }}>
          <div style={{ padding: '16px 16px 12px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>Узлы графа</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div className="metric"><div className="k">строки</div><div className="v row">{stats.n_rows || '—'}</div><div className="delta">A + B</div></div>
              <div className="metric"><div className="k">токены</div><div className="v token">{stats.n_tokens || '—'}</div><div className="delta">idf ≥ {idfMin}</div></div>
              <div className="metric"><div className="k">рёбра</div><div className="v">{stats.n_edges || '—'}</div><div className="delta">токен → строка</div></div>
              <div className="metric"><div className="k">col_dim</div><div className="v">{stats.col_dim}</div><div className="delta">MRL</div></div>
            </div>
          </div>

          <div style={{ padding: '0 16px 12px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>Фильтр токенов по IDF</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <span>min_df</span>
              <input type="range" min={1} max={6} step={1} value={idfMin}
                onChange={(e) => setIdfMin(parseInt(e.target.value))} className="range" />
              <b>{idfMin}</b>
            </div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', marginTop: 4 }}>
              значение учитывается при следующей сборке графа
            </div>
          </div>

          <div style={{ flexShrink: 0, height: 200, borderTop: '1px solid var(--border)', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '8px 16px', fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, borderBottom: '1px solid var(--border)' }}>
              журнал сборки
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
          ? <>Граф построен. Прямой проход GAT на следующем шаге.</>
          : phase === 'error'
            ? <span style={{ color: 'var(--bad)' }}>Ошибка: {errorMsg}</span>
            : <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp; {phaseLabels[phase]}</>}
      </ScreenFooter>
    </div>
  );
}

Object.assign(window, { ScreenGraph });
