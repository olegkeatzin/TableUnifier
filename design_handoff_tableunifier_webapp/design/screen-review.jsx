// Screen 4 — manual review of candidate duplicate pairs

function ScreenReview({ onContinue, onBack, decisions, setDecisions }) {
  const D = window.__DATA__;
  const [filter, setFilter] = useState('all');  // all | review | auto | reject
  const [activePair, setActivePair] = useState(null);
  const [threshold, setThreshold] = useState(0.85);

  // candidates filtered + sorted
  const allCandidates = D.candidates;
  const filtered = allCandidates.filter((c) => {
    if (filter === 'all') return true;
    if (filter === 'review') return c.verdict === 'review';
    if (filter === 'auto') return c.verdict === 'auto';
    if (filter === 'reject') return c.verdict === 'reject';
    return true;
  });

  // counts
  const counts = {
    all: allCandidates.length,
    auto: allCandidates.filter((c) => c.verdict === 'auto').length,
    review: allCandidates.filter((c) => c.verdict === 'review').length,
    reject: allCandidates.filter((c) => c.verdict === 'reject').length,
  };

  const pendingReview = allCandidates
    .filter((c) => c.verdict === 'review')
    .filter((c) => !decisions[pairKey(c)])
    .length;

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Ревью пар-кандидатов</h1>
          <p>GNN выдал {allCandidates.length} пар с косинусной близостью ≥ 0.65. GA-настроенный порог + connected components автоматически разрешили {counts.auto} пар; {counts.review} требуют ручной проверки из-за пограничной метрики или конфликтующих полей.</p>
        </div>
        <div className="actions">
          <button className="btn ghost">⤓ экспорт пар</button>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '300px 1fr', minHeight: 0 }}>
        {/* left panel — filters + threshold */}
        <div style={{ borderRight: '1px solid var(--border)', padding: '14px 14px 0', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>фильтр</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              { k: 'all',    l: 'Все пары',         c: counts.all,    color: 'var(--text)' },
              { k: 'auto',   l: 'Автоматически',    c: counts.auto,   color: 'var(--cluster)' },
              { k: 'review', l: 'Требует ревью',    c: counts.review, color: 'var(--warn)' },
              { k: 'reject', l: 'Отклонено',        c: counts.reject, color: 'var(--bad)' },
            ].map((b) => (
              <button key={b.k}
                onClick={() => setFilter(b.k)}
                className={`btn ${filter === b.k ? '' : 'ghost'}`}
                style={{
                  justifyContent: 'space-between', height: 32, width: '100%',
                  fontWeight: 500, color: filter === b.k ? 'var(--text)' : b.color,
                }}>
                <span>{b.l}</span>
                <span className="mono" style={{ fontSize: 11 }}>{b.c}</span>
              </button>
            ))}
          </div>

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>
              similarity threshold
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 500, marginBottom: 4 }}>
              {threshold.toFixed(2)}
            </div>
            <input type="range" min={0.5} max={0.99} step={0.01} value={threshold}
              className="range"
              onChange={(e) => setThreshold(parseFloat(e.target.value))} />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)' }}>
              <span>0.50</span><span>GA: 0.831</span><span>0.99</span>
            </div>
          </div>

          <div style={{ marginTop: 18, padding: 12, background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 8 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>GA-tuned threshold</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
              <div><span style={{ color: 'var(--text-4)' }}>clustering:</span> connected components</div>
              <div><span style={{ color: 'var(--text-4)' }}>metric:</span> cosine</div>
              <div><span style={{ color: 'var(--text-4)' }}>GA pop:</span> 50 · gen=40</div>
              <div><span style={{ color: 'var(--text-4)' }}>fitness:</span> F1=0.913 @ thr=0.831</div>
            </div>
          </div>

          <div style={{ marginTop: 'auto', padding: '12px 0 14px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>прогресс ревью</div>
            <div style={{ height: 4, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${100 - (pendingReview / counts.review || 0) * 100}%`, background: 'var(--cluster)' }}></div>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>
              {counts.review - pendingReview}/{counts.review} проверено
            </div>
          </div>
        </div>

        {/* right panel — pair list */}
        <div style={{ overflow: 'auto', padding: '14px 16px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {filtered.map((c, i) => (
              <PairCard
                key={i}
                cand={c}
                threshold={threshold}
                decision={decisions[pairKey(c)]}
                onDecide={(verdict) => {
                  setDecisions({ ...decisions, [pairKey(c)]: verdict });
                }}
                onSelect={() => setActivePair(c)}
                active={activePair === c}
              />
            ))}
            {filtered.length === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-4)', fontSize: 12 }}>
                Нет пар в этой категории.
              </div>
            )}
          </div>
        </div>
      </div>

      <ScreenFooter
        onBack={onBack}
        onNext={onContinue}
        nextLabel="Собрать унифицированную таблицу"
        nextDisabled={false}>
        {pendingReview > 0
          ? <>Осталось {pendingReview} пар на проверку. Можно продолжить — непринятые решения трактуются как авто-вердикт алгоритма.</>
          : <>Всё проверено. Готово к финальной сборке.</>}
      </ScreenFooter>
    </div>
  );
}

function pairKey(c) {
  return `${c.a[0]}${c.a[1]}-${c.b[0]}${c.b[1]}`;
}

function PairCard({ cand, threshold, decision, onDecide, active }) {
  const D = window.__DATA__;
  const A = D.tableA.data[cand.a[1]];
  const B = D.tableB.data[cand.b[1]];

  // map fields for comparison: pick a unified set of "logical" columns
  const fields = [
    ['ID',           A[0],            B[0]],
    ['Марка',        A[1],            B[1]],
    ['Модель',       A[2],            B[2]],
    ['Год',          A[3],            B[3]],
    ['Пробег, км',   A[4],            B[4]],
    ['Цвет',         A[5],            B[5]],
    ['Кузов',        A[6],            B[6]],
    ['Двигатель',    `${A[7]}L`,       `${(B[7] / 1000).toFixed(1)}L`],
    ['Коробка',      A[8],            B[8]],
    ['Цена, ₽',      A[9].toLocaleString('ru-RU'), (B[9] * 1000).toLocaleString('ru-RU')],
  ];

  // detect divergent fields (string-different even if same logical value)
  const divergent = fields.map(([k, a, b]) => {
    if (a === b) return false;
    const sa = String(a).toLowerCase().replace(/[#\s-]/g, '');
    const sb = String(b).toLowerCase().replace(/[#\s-]/g, '');
    return sa !== sb;
  });

  // user decision overrides verdict
  const effective = decision || (cand.sim >= threshold ? cand.verdict : 'reject');

  const verdictChip = {
    auto:    <span className="chip cluster"><span className="dot"></span>auto-match</span>,
    review:  <span className="chip warn"><span className="dot"></span>требует ревью</span>,
    reject:  <span className="chip" style={{ color: 'var(--bad)', borderColor: 'color-mix(in oklch, var(--bad) 40%, var(--border))' }}><span className="dot" style={{ background: 'var(--bad)' }}></span>отклонено</span>,
  }[cand.verdict];

  return (
    <div className="pair" style={{
      borderColor: decision === 'approve' ? 'color-mix(in oklch, var(--cluster) 50%, var(--border))'
                : decision === 'reject'  ? 'color-mix(in oklch, var(--bad) 50%, var(--border))'
                : 'var(--border)',
    }}>
      <div className="pair-h">
        <div className="left">
          <span className="mono" style={{ color: 'var(--text-3)' }}>
            {cand.a[0]}/{cand.a[1] + 1} ↔ {cand.b[0]}/{cand.b[1] + 1}
          </span>
          {verdictChip}
          {cand.cluster && <span className="chip"><span className="mono" style={{ opacity: 0.6 }}>cluster</span> {cand.cluster}</span>}
        </div>
        <div className="right">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)' }}>
            sim = <b style={{
              color: cand.sim > 0.85 ? 'var(--cluster)' : cand.sim > 0.7 ? 'var(--warn)' : 'var(--bad)',
              fontSize: 13,
            }}>{cand.sim.toFixed(3)}</b>
          </div>
          <div style={{ width: 1, height: 16, background: 'var(--border)' }}></div>
          {decision === 'approve' && <span className="chip cluster"><span className="dot"></span>approved</span>}
          {decision === 'reject' && <span className="chip" style={{ color: 'var(--bad)', borderColor: 'color-mix(in oklch, var(--bad) 40%, var(--border))' }}><span className="dot" style={{ background: 'var(--bad)' }}></span>rejected</span>}
          <button className="btn"
            style={decision === 'approve' ? { background: 'var(--cluster-soft)', borderColor: 'var(--cluster)', color: 'var(--cluster)' } : {}}
            onClick={() => onDecide(decision === 'approve' ? null : 'approve')}>
            ✓ объединить
          </button>
          <button className="btn danger"
            style={decision === 'reject' ? { background: 'color-mix(in oklch, var(--bad) 18%, transparent)', borderColor: 'var(--bad)' } : {}}
            onClick={() => onDecide(decision === 'reject' ? null : 'reject')}>
            ✗ разделить
          </button>
        </div>
      </div>
      <div className="pair-body">
        <div className="pair-side">
          <h5><span className="chip row" style={{ height: 16, padding: '0 5px' }}><span className="dot"></span>{D.tableA.name}</span><span style={{ color: 'var(--text-4)' }}>· row #{cand.a[1] + 1}</span></h5>
          {fields.map(([k, a], i) => (
            <div key={i} className={`field ${divergent[i] ? 'div' : ''}`}>
              <span className="k">{k}</span>
              <span className="v">{D.tableA.cols[i] === 'color' ? <ColorSwatch value={String(a)} /> : String(a)}</span>
            </div>
          ))}
        </div>
        <div className="pair-mid">
          <div className="sim">SIMILARITY {cand.sim.toFixed(2)}</div>
        </div>
        <div className="pair-side">
          <h5><span className="chip token" style={{ height: 16, padding: '0 5px' }}><span className="dot"></span>{D.tableB.name}</span><span style={{ color: 'var(--text-4)' }}>· row #{cand.b[1] + 1}</span></h5>
          {fields.map(([k, , b], i) => (
            <div key={i} className={`field ${divergent[i] ? 'div' : ''}`}>
              <span className="k">{k}</span>
              <span className="v">{D.tableB.cols[i] === 'color_hex' ? <ColorSwatch value={String(b)} /> : String(b)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { ScreenReview });
