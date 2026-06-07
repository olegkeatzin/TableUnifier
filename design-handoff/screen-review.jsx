// Screen 4 — manual review of candidate duplicate pairs (driven by real API).

function pairKey(c) { return c.id || `${c.a[0]}${c.a[1]}-${c.b[0]}${c.b[1]}`; }
function pairSim(c) { return c.sim ?? c.similarity ?? 0; }

// Порог — это отсечка по косинусному сходству (само сходство фиксировано).
// Классифицируем пары относительно порога: уверенно выше → авто-слияние,
// в полосе ±band вокруг порога → ручная проверка, ниже → отклонение.
// Так слайдер реально влияет на раскладку, без обращения к бэку.
const VERDICT_BAND = 0.04;
function classifyPair(sim, threshold) {
  if (sim >= threshold + VERDICT_BAND) return 'auto';
  if (sim >= threshold - VERDICT_BAND) return 'review';
  return 'reject';
}

function ScreenReview({ onContinue, onBack, decisions, setDecisions }) {
  const D = window.__DATA__;
  const [filter, setFilter] = useState('all');
  const [activePair, setActivePair] = useState(null);
  const [threshold, setThreshold] = useState(D.metrics?.threshold || 0.831);

  // Если данных нет — попробуем подтянуть.
  useEffect(() => {
    if ((!D.candidates || D.candidates.length === 0) && window.__STATE__.runId) {
      window.API.getClusters(window.__STATE__.runId).catch(console.error);
    }
  }, []);

  const allCandidates = D.candidates || [];
  const verdictOf = (c) => classifyPair(pairSim(c), threshold);
  const filtered = allCandidates.filter((c) => filter === 'all' || verdictOf(c) === filter);

  const counts = {
    all: allCandidates.length,
    auto: allCandidates.filter((c) => verdictOf(c) === 'auto').length,
    review: allCandidates.filter((c) => verdictOf(c) === 'review').length,
    reject: allCandidates.filter((c) => verdictOf(c) === 'reject').length,
  };

  const pendingReview = allCandidates
    .filter((c) => verdictOf(c) === 'review')
    .filter((c) => !decisions[pairKey(c)])
    .length;

  const submit = async () => {
    if (window.__STATE__.runId && Object.keys(decisions).length > 0) {
      try {
        await window.API.postDecisions(window.__STATE__.runId, decisions);
      } catch (e) { console.error(e); }
    }
    onContinue();
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Проверка пар-кандидатов</h1>
          <p>GNN выдал {allCandidates.length} пар. Компоненты связности автоматически разрешили {counts.auto}; {counts.review} требуют ручной проверки.</p>
        </div>
      </div>

      <div className="screen-body" style={{ display: 'grid', gridTemplateColumns: '300px 1fr', minHeight: 0 }}>
        <div style={{ borderRight: '1px solid var(--border)', padding: '14px 14px 0', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 10 }}>фильтр</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              { k: 'all', l: 'Все пары', c: counts.all, color: 'var(--text)' },
              { k: 'auto', l: 'Автоматически', c: counts.auto, color: 'var(--cluster)' },
              { k: 'review', l: 'Требует проверки', c: counts.review, color: 'var(--warn)' },
              { k: 'reject', l: 'Отклонено', c: counts.reject, color: 'var(--bad)' },
            ].map((b) => (
              <button key={b.k} onClick={() => setFilter(b.k)}
                className={`btn ${filter === b.k ? '' : 'ghost'}`}
                style={{ justifyContent: 'space-between', height: 32, width: '100%',
                         fontWeight: 500, color: filter === b.k ? 'var(--text)' : b.color }}>
                <span>{b.l}</span>
                <span className="mono" style={{ fontSize: 11 }}>{b.c}</span>
              </button>
            ))}
          </div>

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 8 }}>порог сходства</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 500, marginBottom: 4 }}>{threshold.toFixed(2)}</div>
            <input type="range" min={0.5} max={0.99} step={0.01} value={threshold}
              className="range" onChange={(e) => setThreshold(parseFloat(e.target.value))} />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-4)' }}>
              <span>0.50</span><span>порог</span><span>0.99</span>
            </div>
          </div>

          <div style={{ marginTop: 18, padding: 12, background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 8 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>кластеризация</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
              <div><span style={{ color: 'var(--text-4)' }}>метод:</span> компоненты связности</div>
              <div><span style={{ color: 'var(--text-4)' }}>метрика:</span> косинусная</div>
              <div><span style={{ color: 'var(--text-4)' }}>порог:</span> <b style={{ color: 'var(--text)' }}>{threshold.toFixed(3)}</b></div>
            </div>
          </div>

          <div style={{ marginTop: 'auto', padding: '12px 0 14px' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 0.06, marginBottom: 6 }}>ход проверки</div>
            <div style={{ height: 4, background: 'var(--surface)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${counts.review ? (1 - pendingReview / counts.review) * 100 : 0}%`, background: 'var(--cluster)' }}></div>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>
              {counts.review - pendingReview}/{counts.review} проверено
            </div>
          </div>
        </div>

        <div style={{ overflow: 'auto', padding: '14px 16px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {filtered.map((c) => (
              <PairCard
                key={pairKey(c)}
                cand={c}
                threshold={threshold}
                decision={decisions[pairKey(c)]}
                onDecide={(v) => setDecisions({ ...decisions, [pairKey(c)]: v })}
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
        onNext={submit}
        nextLabel="Собрать унифицированную таблицу"
        nextDisabled={false}>
        {pendingReview > 0
          ? <>Осталось {pendingReview} пар. Можно продолжить — нерешённые остаются с авто-решением.</>
          : <>Всё проверено.</>}
      </ScreenFooter>
    </div>
  );
}

function PairCard({ cand, threshold, decision, onDecide }) {
  const D = window.__DATA__;
  const tblA = D.tableA, tblB = D.tableB;
  if (!tblA || !tblB) return null;
  const aRow = tblA.data[cand.a[1]] || [];
  const bRow = tblB.data[cand.b[1]] || [];
  const colsA = tblA.cols;
  const colsB = tblB.cols;
  const divergent = new Set(cand.divergence || []);
  const sim = cand.sim ?? cand.similarity ?? 0;
  const verdict = classifyPair(sim, threshold);
  const delta = sim - threshold;
  const verdictColor = verdict === 'auto' ? 'var(--cluster)' : verdict === 'review' ? 'var(--warn)' : 'var(--bad)';

  const verdictChip = {
    auto:    <span className="chip cluster"><span className="dot"></span>авто-совпадение</span>,
    review:  <span className="chip warn"><span className="dot"></span>требует проверки</span>,
    reject:  <span className="chip" style={{ color: 'var(--bad)', borderColor: 'color-mix(in oklch, var(--bad) 40%, var(--border))' }}><span className="dot" style={{ background: 'var(--bad)' }}></span>отклонено</span>,
  }[verdict] || null;

  return (
    <div className="pair" style={{
      borderColor: decision === 'approve' ? 'color-mix(in oklch, var(--cluster) 50%, var(--border))'
                : decision === 'reject' ? 'color-mix(in oklch, var(--bad) 50%, var(--border))'
                : 'var(--border)',
    }}>
      <div className="pair-h">
        <div className="left">
          <span className="mono" style={{ color: 'var(--text-3)' }}>
            {cand.a[0]}/{cand.a[1] + 1} ↔ {cand.b[0]}/{cand.b[1] + 1}
          </span>
          {verdictChip}
          {cand.cluster && <span className="chip"><span className="mono" style={{ opacity: 0.6 }}>кластер</span> {cand.cluster}</span>}
        </div>
        <div className="right">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)' }}>
            сходство = <b style={{ color: verdictColor, fontSize: 13 }}>{sim.toFixed(3)}</b>
          </div>
          <span className="mono" style={{ fontSize: 10, color: delta >= 0 ? 'var(--cluster)' : 'var(--bad)', whiteSpace: 'nowrap' }}
                title={`порог ${threshold.toFixed(2)}`}>
            {delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)} к порогу
          </span>
          <div style={{ width: 1, height: 16, background: 'var(--border)' }}></div>
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
          <h5>
            <span className="chip row" style={{ height: 16, padding: '0 5px' }}><span className="dot"></span>{tblA.name}</span>
            <span style={{ color: 'var(--text-4)' }}>· строка №{cand.a[1] + 1}</span>
          </h5>
          {colsA.map((col, i) => (
            <div key={col} className={`field ${divergent.has(col) ? 'div' : ''}`}>
              <span className="k">{col}</span>
              <span className="v">{/color/i.test(col) ? <ColorSwatch value={String(aRow[i] ?? '')} /> : String(aRow[i] ?? '')}</span>
            </div>
          ))}
        </div>
        <div className="pair-mid">
          <div className="sim">СХОДСТВО {sim.toFixed(2)}</div>
        </div>
        <div className="pair-side">
          <h5>
            <span className="chip token" style={{ height: 16, padding: '0 5px' }}><span className="dot"></span>{tblB.name}</span>
            <span style={{ color: 'var(--text-4)' }}>· строка №{cand.b[1] + 1}</span>
          </h5>
          {colsB.map((col, i) => (
            <div key={col} className={`field ${divergent.has(col) ? 'div' : ''}`}>
              <span className="k">{col}</span>
              <span className="v">{/color/i.test(col) ? <ColorSwatch value={String(bRow[i] ?? '')} /> : String(bRow[i] ?? '')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { ScreenReview });
