// Screen 5 — Final unified table + real export from backend.

function ScreenResult({ onBack, onRestart, decisions }) {
  const D = window.__DATA__;
  const [expanded, setExpanded] = useState(() => new Set());
  const [, force] = useState(0);

  // Подтягиваем кластеры, если их ещё нет (прямой переход / перезагрузка).
  useEffect(() => {
    if ((!D.clusters || D.clusters.length === 0) && window.__STATE__.runId) {
      window.API.getClusters(window.__STATE__.runId)
        .then(() => force((t) => t + 1))
        .catch(console.error);
    }
  }, []);

  // Применяем decisions: rejected пары делают мульти-кластеры синглтонами визуально.
  const clusters = useMemo(() => {
    const rejected = new Set(Object.entries(decisions || {})
      .filter(([, v]) => v === 'reject').map(([k]) => k));
    if (rejected.size === 0) return D.clusters || [];
    const rejPairs = (D.candidates || []).filter((c) => rejected.has(c.id));
    if (rejPairs.length === 0) return D.clusters || [];
    return (D.clusters || []).map((c) => {
      // Если хотя бы одна rejected пара содержит обе ноги этого кластера — режем.
      const aRows = c.members.filter((m) => m.source === 'A').map((m) => m.row);
      const bRows = c.members.filter((m) => m.source === 'B').map((m) => m.row);
      const hasReject = rejPairs.some((p) =>
        aRows.includes(p.a[1]) && bRows.includes(p.b[1]),
      );
      if (!hasReject) return c;
      // split: помечаем кластер чтобы отрендерить как синглтоны
      return { ...c, _split: true };
    });
  }, [decisions, D.clusters, D.candidates]);

  const totalIn = (D.tableA?.rows || 0) + (D.tableB?.rows || 0);
  const flatClusters = clusters.flatMap((c) =>
    c._split ? c.members.map((m, i) => ({
      id: `${c.id}-${i + 1}`, members: [m], similarity: 1.0,
    })) : [c],
  );
  const totalOut = flatClusters.length;
  const merged = flatClusters.filter((c) => c.members.length > 1).length;
  const reduction = totalIn ? ((1 - totalOut / totalIn) * 100).toFixed(1) : '0';

  const toggle = (id) => {
    const n = new Set(expanded);
    if (n.has(id)) n.delete(id); else n.add(id);
    setExpanded(n);
  };

  const runId = window.__STATE__.runId;
  const exportHref = (fmt) => runId ? window.API.exportUrl(runId, fmt) : '#';

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Унифицированная таблица</h1>
          <p>Из {totalIn} исходных строк осталось {totalOut} сущностей ({reduction}% дедупликации).</p>
        </div>
        <div className="actions">
          <a className="btn" href={exportHref('xlsx')} download>⤓ .xlsx</a>
          <a className="btn" href={exportHref('parquet')} download>⤓ .parquet</a>
          <a className="btn primary" href={exportHref('csv')} download>⤓ .csv</a>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '1fr 280px', minHeight: 0 }}>
        <div style={{ overflow: 'auto' }}>
          <UnifiedTable clusters={flatClusters} expanded={expanded} toggle={toggle} />
        </div>

        <div style={{ borderLeft: '1px solid var(--border)', padding: 16, display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto' }}>
          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>сводка</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div className="metric"><div className="k">всего на входе</div><div className="v">{totalIn}</div><div className="delta">2 таблицы</div></div>
              <div className="metric"><div className="k">всего на выходе</div><div className="v cluster">{totalOut}</div><div className="delta">−{reduction}%</div></div>
              <div className="metric"><div className="k">объединено</div><div className="v">{merged}</div><div className="delta">≥ 2 записи</div></div>
              <div className="metric"><div className="k">одиночные</div><div className="v">{totalOut - merged}</div><div className="delta">1 запись</div></div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>метрики</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {Object.entries(D.metrics || {}).slice(0, 8).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-3)' }}>{k}</span>
                  <b>{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}</b>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            padding: 12, background: 'var(--cluster-soft)',
            border: '1px solid color-mix(in oklch, var(--cluster) 35%, var(--border))',
            borderRadius: 8, color: 'var(--text)',
          }}>
            <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>✓ Конвейер завершён</div>
            <div style={{ fontSize: 11, color: 'var(--text-2)' }}>
              Запуск: <span className="mono">{runId || '—'}</span>
            </div>
          </div>

          <button className="btn ghost" style={{ marginTop: 'auto' }} onClick={onRestart}>↻ начать заново</button>
        </div>
      </div>

      <ScreenFooter onBack={onBack} onNext={onRestart} nextLabel="Новый конвейер">
        Готово. Экспорт сверху справа.
      </ScreenFooter>
    </div>
  );
}

function UnifiedTable({ clusters, expanded, toggle }) {
  const D = window.__DATA__;
  const tblA = D.tableA, tblB = D.tableB;
  if (!tblA || !tblB) {
    return <div style={{ padding: 24, color: 'var(--text-4)' }}>Нет данных. Запустите инференс.</div>;
  }
  const cols = Array.from(new Set([...tblA.cols, ...tblB.cols]));

  const memberValue = (m, col) => {
    const tbl = m.source === 'A' ? tblA : tblB;
    const idx = tbl.cols.indexOf(col);
    if (idx === -1) return '';
    const row = tbl.data[m.row];
    return row ? row[idx] : '';
  };

  const canonical = (c) => {
    if (c.members.length === 1) {
      return Object.fromEntries(cols.map((col) => [col, memberValue(c.members[0], col)]));
    }
    return Object.fromEntries(cols.map((col) => {
      const values = c.members.map((m) => memberValue(m, col)).filter((v) => v != null && v !== '');
      if (values.length === 0) return [col, ''];
      // numeric mean
      if (values.every((v) => typeof v === 'number' || (!isNaN(parseFloat(v)) && isFinite(v)))) {
        const nums = values.map((v) => parseFloat(v));
        const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
        return [col, Number.isInteger(mean) ? String(mean) : mean.toFixed(2)];
      }
      // mode
      const counts = {};
      for (const v of values) counts[v] = (counts[v] || 0) + 1;
      const mode = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      return [col, mode];
    }));
  };

  return (
    <table className="dt">
      <thead>
        <tr>
          <th style={{ width: 90 }}>кластер</th>
          {cols.map((c) => <th key={c}>{c}</th>)}
          <th style={{ width: 60 }}>записей</th>
        </tr>
      </thead>
      <tbody>
        {clusters.map((c) => {
          const canon = canonical(c);
          const isExp = expanded.has(c.id);
          const isMulti = c.members.length > 1;
          return (
            <React.Fragment key={c.id}>
              <tr className="cluster-row"
                  onClick={() => isMulti && toggle(c.id)}
                  style={{ cursor: isMulti ? 'pointer' : 'default' }}>
                <td>
                  {isMulti ? (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, display: 'inline-block', color: 'var(--text-4)' }}>{isExp ? '▼' : '▶'}</span>
                      {c.id}
                    </span>
                  ) : c.id}
                </td>
                {cols.map((col) => (
                  <td key={col}>
                    {/color/i.test(col) ? <ColorSwatch value={String(canon[col] ?? '')} /> : String(canon[col] ?? '')}
                  </td>
                ))}
                <td>
                  {isMulti
                    ? <span className="chip cluster" style={{ height: 16 }}><span className="dot"></span>{c.members.length}</span>
                    : <span className="chip" style={{ height: 16 }}>1</span>}
                </td>
              </tr>
              {isExp && c.members.map((m, i) => (
                <tr key={i} className="member-row">
                  <td>↳ {m.source}/{m.row + 1}</td>
                  {cols.map((col) => (
                    <td key={col}>
                      {/color/i.test(col) ? <ColorSwatch value={String(memberValue(m, col) ?? '')} /> : String(memberValue(m, col) ?? '')}
                    </td>
                  ))}
                  <td><span className="mono" style={{ color: 'var(--text-4)', fontSize: 10 }}>ист.</span></td>
                </tr>
              ))}
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

Object.assign(window, { ScreenResult });
