// Screen 5 — Final unified table + export

function ScreenResult({ onBack, onRestart, decisions }) {
  const D = window.__DATA__;
  const [view, setView] = useState('unified');   // unified | sankey
  const [expanded, setExpanded] = useState(() => new Set(['C-001', 'C-009']));

  // assemble unified clusters with user decisions applied
  const clusters = useMemo(() => {
    return D.clusters.map((c) => {
      // apply rejected pairs: if a pair was rejected, split that member out
      const rejected = Object.entries(decisions)
        .filter(([, v]) => v === 'reject')
        .map(([k]) => k);
      // simple model: don't split clusters in this prototype, just mark
      return c;
    });
  }, [decisions]);

  const totalIn = D.tableA.rows + D.tableB.rows;
  const totalOut = clusters.length;
  const merged = clusters.filter((c) => c.members.length > 1).length;
  const reduction = ((1 - totalOut / totalIn) * 100).toFixed(1);

  const toggle = (id) => {
    const n = new Set(expanded);
    if (n.has(id)) n.delete(id); else n.add(id);
    setExpanded(n);
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Унифицированная таблица</h1>
          <p>Из {totalIn} исходных строк осталось {totalOut} уникальных сущностей ({reduction}% дедупликация). Каждая сущность — кластер из 1+ оригинальных записей с разрешёнными конфликтами полей.</p>
        </div>
        <div className="actions">
          <Tabs
            active={view}
            setActive={setView}
            tabs={[
              { key: 'unified', label: 'Таблица' },
              { key: 'sankey',  label: 'Sankey' },
            ]}
          />
          <button className="btn">⤓ .xlsx</button>
          <button className="btn">⤓ .parquet</button>
          <button className="btn primary">⤓ .csv</button>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '1fr 280px', minHeight: 0 }}>
        {/* main: unified table */}
        <div style={{ overflow: 'auto', padding: view === 'sankey' ? 16 : 0 }}>
          {view === 'unified' ? (
            <UnifiedTable clusters={clusters} expanded={expanded} toggle={toggle} />
          ) : (
            <SankeyView clusters={clusters} totalIn={totalIn} />
          )}
        </div>

        {/* right summary */}
        <div style={{ borderLeft: '1px solid var(--border)', padding: '16px 16px 16px', display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto' }}>
          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>сводка</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div className="metric"><div className="k">всего in</div><div className="v">{totalIn}</div><div className="delta">2 таблицы</div></div>
              <div className="metric"><div className="k">всего out</div><div className="v cluster">{totalOut}</div><div className="delta">−{reduction}%</div></div>
              <div className="metric"><div className="k">merged</div><div className="v">{merged}</div><div className="delta">≥ 2 записи</div></div>
              <div className="metric"><div className="k">singletons</div><div className="v">{totalOut - merged}</div><div className="delta">1 запись</div></div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>конфликты полей</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {[
                ['color',      9, 'mode'],
                ['mileage',    7, 'avg'],
                ['price',      9, 'min'],
                ['bodyType',   8, 'mode'],
                ['engine',     6, 'mode'],
              ].map(([col, n, strat]) => (
                <div key={col} style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: 8, alignItems: 'center' }}>
                  <span style={{ color: 'var(--text-2)' }}>{col}</span>
                  <span style={{ color: 'var(--text-4)' }}>n={n}</span>
                  <span className="chip" style={{ fontSize: 10 }}>{strat}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>финальные метрики</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-3)' }}>F1 (pair)</span><b>0.913</b></div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-3)' }}>Precision</span><b>0.940</b></div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-3)' }}>Recall</span><b>0.890</b></div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-3)' }}>ROC-AUC</span><b>0.957</b></div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-3)' }}>ARI (cluster)</span><b>0.881</b></div>
            </div>
          </div>

          <div style={{
            padding: 12, background: 'var(--cluster-soft)',
            border: '1px solid color-mix(in oklch, var(--cluster) 35%, var(--border))',
            borderRadius: 8, color: 'var(--text)',
          }}>
            <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>✓ Pipeline завершён</div>
            <div style={{ fontSize: 11, color: 'var(--text-2)' }}>
              Артефакты сохранены в <span className="mono">output/bge-m3/</span>. Pipeline воспроизводим через <span className="mono">--config v17_views_threshold.yml</span>.
            </div>
          </div>

          <button className="btn ghost" style={{ marginTop: 'auto' }} onClick={onRestart}>↻ начать заново</button>
        </div>
      </div>

      <ScreenFooter
        onBack={onBack}
        onNext={onRestart}
        nextLabel="Новый pipeline">
        Готово. Чтобы запустить ER на других данных — нажмите "Новый pipeline" или экспортируйте текущий результат справа сверху.
      </ScreenFooter>
    </div>
  );
}

// ---- Unified table ----
function UnifiedTable({ clusters, expanded, toggle }) {
  const D = window.__DATA__;
  return (
    <table className="dt">
      <thead>
        <tr>
          <th style={{ width: 90 }}>cluster_id</th>
          <th>brand</th>
          <th>model</th>
          <th>year</th>
          <th>color</th>
          <th>bodyType</th>
          <th>mileage</th>
          <th>engine</th>
          <th>price</th>
          <th style={{ width: 60 }}>members</th>
        </tr>
      </thead>
      <tbody>
        {clusters.map((c) => {
          // resolve a canonical row for the cluster
          const canon = resolveCanonical(c);
          const isExp = expanded.has(c.id);
          return (
            <React.Fragment key={c.id}>
              <tr className="cluster-row"
                onClick={() => c.members.length > 1 && toggle(c.id)}
                style={{ cursor: c.members.length > 1 ? 'pointer' : 'default' }}>
                <td>
                  {c.members.length > 1 ? (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, display: 'inline-block', color: 'var(--text-4)' }}>{isExp ? '▼' : '▶'}</span>
                      {c.id}
                    </span>
                  ) : c.id}
                </td>
                <td>{canon.brand}</td>
                <td>{canon.model}</td>
                <td>{canon.year}</td>
                <td><ColorSwatch value={canon.color} /></td>
                <td>{canon.bodyType}</td>
                <td>{canon.mileage.toLocaleString('ru-RU')}</td>
                <td>{canon.engine}L</td>
                <td>{canon.price.toLocaleString('ru-RU')} ₽</td>
                <td>
                  {c.members.length > 1
                    ? <span className="chip cluster" style={{ height: 16 }}><span className="dot"></span>{c.members.length}</span>
                    : <span className="chip" style={{ height: 16 }}>1</span>}
                </td>
              </tr>
              {isExp && c.members.map((m, i) => {
                const tbl = m[0] === 'A' ? D.tableA : D.tableB;
                const row = tbl.data[m[1]];
                return (
                  <tr key={i} className="member-row">
                    <td>↳ {m[0]}/{m[1] + 1}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                    <td><ColorSwatch value={String(row[5])} /></td>
                    <td>{row[6]}</td>
                    <td>{row[4].toLocaleString('ru-RU')}</td>
                    <td>{m[0] === 'A' ? `${row[7]}L` : `${(row[7] / 1000).toFixed(1)}L`}</td>
                    <td>{m[0] === 'A' ? row[9].toLocaleString('ru-RU') : (row[9] * 1000).toLocaleString('ru-RU')} ₽</td>
                    <td><span className="mono" style={{ color: 'var(--text-4)', fontSize: 10 }}>src</span></td>
                  </tr>
                );
              })}
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

function resolveCanonical(cluster) {
  const D = window.__DATA__;
  if (cluster.members.length === 1) {
    const [src, idx] = cluster.members[0];
    const tbl = src === 'A' ? D.tableA : D.tableB;
    const row = tbl.data[idx];
    return {
      brand: row[1], model: row[2], year: row[3],
      mileage: row[4], color: String(row[5]),
      bodyType: src === 'A' ? row[6] : row[6][0] + row[6].slice(1).toLowerCase(),
      engine: src === 'A' ? row[7] : (row[7] / 1000).toFixed(1),
      price: src === 'A' ? row[9] : row[9] * 1000,
    };
  }
  // merge — prefer A for human-readable color/bodyType
  const [aSrc, aIdx] = cluster.members.find((m) => m[0] === 'A') || cluster.members[0];
  const tblA = aSrc === 'A' ? D.tableA : D.tableB;
  const rA = tblA.data[aIdx];
  // average numeric fields across members
  let totMileage = 0, totPrice = 0, n = 0;
  cluster.members.forEach(([s, i]) => {
    const t = s === 'A' ? D.tableA : D.tableB;
    const r = t.data[i];
    totMileage += r[4]; n++;
    totPrice += s === 'A' ? r[9] : r[9] * 1000;
  });
  return {
    brand: rA[1],
    model: rA[2],
    year: rA[3],
    mileage: Math.round(totMileage / n),
    color: String(rA[5]),
    bodyType: aSrc === 'A' ? rA[6] : rA[6][0] + rA[6].slice(1).toLowerCase(),
    engine: aSrc === 'A' ? rA[7] : (rA[7] / 1000).toFixed(1),
    price: Math.round(totPrice / n),
  };
}

// ---- Sankey visualization (custom SVG, no library) ----
function SankeyView({ clusters, totalIn }) {
  const D = window.__DATA__;
  // 3 columns: source tables → clusters → output
  const W = 900, H = 520;
  const colX = [80, W / 2 - 30, W - 80];
  const colW = 14;

  // sources block coordinates
  const srcA = { x: colX[0], y: 60, h: 200, label: 'Table A (14 rows)', color: 'var(--row)' };
  const srcB = { x: colX[0], y: 290, h: 180, label: 'Table B (13 rows)', color: 'var(--token)' };

  // clusters in middle column, stacked
  const sortedClusters = [...clusters].sort((a, b) => b.members.length - a.members.length);
  const midH = 460;
  let yCursor = 50;
  const clusterRects = sortedClusters.map((c, i) => {
    const h = Math.max(8, (c.members.length / totalIn) * midH * 1.4);
    const rect = { x: colX[1], y: yCursor, h, c, idx: i };
    yCursor += h + 3;
    return rect;
  });

  // output single block
  const outBlock = { x: colX[2], y: 60, h: 410, label: `Unified (${clusters.length} entities)`, color: 'var(--cluster)' };

  // build flows: source row → cluster
  const flows = [];
  clusterRects.forEach((cr) => {
    const c = cr.c;
    const aMembers = c.members.filter((m) => m[0] === 'A');
    const bMembers = c.members.filter((m) => m[0] === 'B');
    if (aMembers.length > 0) {
      flows.push({
        from: { x: srcA.x + colW, y: srcA.y + (aMembers[0][1] / 14) * srcA.h + 4 },
        to:   { x: cr.x, y: cr.y + cr.h * 0.3 },
        thick: Math.max(1, aMembers.length * 3),
        color: 'var(--row)',
        op: 0.45,
      });
    }
    if (bMembers.length > 0) {
      flows.push({
        from: { x: srcB.x + colW, y: srcB.y + (bMembers[0][1] / 13) * srcB.h + 4 },
        to:   { x: cr.x, y: cr.y + cr.h * 0.7 },
        thick: Math.max(1, bMembers.length * 3),
        color: 'var(--token)',
        op: 0.45,
      });
    }
    // cluster → output
    flows.push({
      from: { x: cr.x + colW, y: cr.y + cr.h / 2 },
      to:   { x: outBlock.x, y: outBlock.y + ((cr.idx + 0.5) / clusterRects.length) * outBlock.h },
      thick: Math.max(1, c.members.length * 2),
      color: c.members.length > 1 ? 'var(--cluster)' : 'var(--text-3)',
      op: 0.35,
    });
  });

  return (
    <div style={{ background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 12, padding: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>Поток объединения</div>
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 12 }}>
        Исходные строки → кластеры → итоговые сущности. Толщина потока ∝ количеству записей.
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {/* flows first */}
        {flows.map((f, i) => {
          const dx = (f.to.x - f.from.x) / 2;
          const d = `M ${f.from.x} ${f.from.y} C ${f.from.x + dx} ${f.from.y}, ${f.to.x - dx} ${f.to.y}, ${f.to.x} ${f.to.y}`;
          return (
            <path key={i} d={d} fill="none" stroke={f.color} strokeWidth={f.thick} opacity={f.op} strokeLinecap="round" />
          );
        })}
        {/* source A */}
        <rect x={srcA.x} y={srcA.y} width={colW} height={srcA.h} fill={srcA.color} rx={2} />
        <text x={srcA.x - 6} y={srcA.y + srcA.h / 2} textAnchor="end" fontSize="11" fontFamily="var(--font-mono)" fill={srcA.color}>{srcA.label}</text>
        {/* source B */}
        <rect x={srcB.x} y={srcB.y} width={colW} height={srcB.h} fill={srcB.color} rx={2} />
        <text x={srcB.x - 6} y={srcB.y + srcB.h / 2} textAnchor="end" fontSize="11" fontFamily="var(--font-mono)" fill={srcB.color}>{srcB.label}</text>
        {/* cluster rects */}
        {clusterRects.map((cr, i) => (
          <g key={i}>
            <rect x={cr.x} y={cr.y} width={colW} height={cr.h}
              fill={cr.c.members.length > 1 ? 'var(--cluster)' : 'var(--text-3)'} rx={2} opacity={cr.c.members.length > 1 ? 0.95 : 0.4} />
            {cr.h > 14 && (
              <text x={cr.x + colW + 4} y={cr.y + cr.h / 2 + 3} fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-4)">{cr.c.id}</text>
            )}
          </g>
        ))}
        {/* output */}
        <rect x={outBlock.x} y={outBlock.y} width={colW} height={outBlock.h} fill={outBlock.color} rx={2} />
        <text x={outBlock.x + colW + 6} y={outBlock.y + outBlock.h / 2} fontSize="11" fontFamily="var(--font-mono)" fill={outBlock.color}>{outBlock.label}</text>

        {/* column headers */}
        <text x={srcA.x + colW / 2} y={28} textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-3)">SOURCES</text>
        <text x={colX[1] + colW / 2} y={28} textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-3)">CLUSTERS (CC + thr)</text>
        <text x={outBlock.x + colW / 2} y={28} textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-3)">UNIFIED</text>
      </svg>
    </div>
  );
}

Object.assign(window, { ScreenResult });
