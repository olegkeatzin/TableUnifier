// Shared UI primitives

const { useState, useEffect, useRef, useMemo, useCallback } = React;

// ---------- Brand mark ----------
function BrandMark() {
  return (
    <div className="brand">
      <div className="brand-mark"></div>
      <div>TableUnifier</div>
      <span style={{ color: 'var(--text-4)', fontFamily: 'var(--font-mono)', fontSize: 10.5, marginLeft: 4 }}>
        v0.18.2
      </span>
    </div>
  );
}

// ---------- Sidebar steps ----------
const STEPS = [
  { key: 'upload',  label: 'Источники',     sub: 'Загрузка .xlsx' },
  { key: 'graph',   label: 'Граф',          sub: 'row + token nodes' },
  { key: 'infer',   label: 'Инференс GNN',  sub: 'message passing · L=2' },
  { key: 'review',  label: 'Ревью пар',     sub: 'кандидаты-дубликаты' },
  { key: 'result',  label: 'Результат',     sub: 'unified table + export' },
];

function Sidebar({ current, setCurrent, completed, stats }) {
  return (
    <aside className="sidebar">
      <h6>Pipeline</h6>
      <div className="steps">
        {STEPS.map((s, i) => {
          const isActive = current === i;
          const isDone = completed.has(i);
          return (
            <div
              key={s.key}
              className={`step ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}
              onClick={() => (isDone || isActive || completed.has(i - 1)) && setCurrent(i)}
            >
              <div className="num">{isDone ? '✓' : i + 1}</div>
              <div>
                <div className="label">{s.label}</div>
                <div className="sub">{s.sub}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="sidebar-section">
        <h6 style={{ margin: '0 0 8px' }}>Параметры</h6>
        {Object.entries(stats).map(([k, v]) => (
          <div className="stat" key={k}>
            <span>{k}</span><b>{v}</b>
          </div>
        ))}
      </div>
    </aside>
  );
}

// ---------- Topbar ----------
function Topbar({ stepIdx }) {
  return (
    <header className="topbar">
      <BrandMark />
      <div style={{ width: 1, height: 16, background: 'var(--border)' }}></div>
      <div className="crumbs">
        <span>er_pipeline</span>
        <span className="sep">/</span>
        <span>auto_ru</span>
        <span className="sep">/</span>
        <span className="cur">{STEPS[stepIdx]?.label.toLowerCase()}</span>
      </div>
      <div className="spacer"></div>
      <div className="meta">
        <span className="pill">model: bge-m3</span>
        <span className="pill">L=2 GNN</span>
        <span className="pill">device: cuda:0</span>
      </div>
      <div className="avatar">ОК</div>
    </header>
  );
}

// ---------- Status bar ----------
function StatusBar({ stepIdx, running }) {
  return (
    <div className="statusbar">
      <span className="ok">●</span>
      <span>ollama @ nvidia-server:11434</span>
      <span style={{ color: 'var(--border-strong)' }}>│</span>
      <span>mlflow run: er_v17_views_gat_2026-05-26</span>
      <span style={{ color: 'var(--border-strong)' }}>│</span>
      <span>{running ? <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp;running</> : 'idle'}</span>
      <div className="spacer"></div>
      <span>step {stepIdx + 1}/{STEPS.length}</span>
      <span style={{ color: 'var(--border-strong)' }}>│</span>
      <span>↑↓ navigate · ↵ continue</span>
    </div>
  );
}

// ---------- Footer with primary continue button ----------
function ScreenFooter({ onBack, onNext, nextLabel = 'Продолжить', nextDisabled, children }) {
  return (
    <div style={{
      padding: '12px 24px', borderTop: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', gap: 10, background: 'var(--bg)',
    }}>
      <div style={{ flex: 1, color: 'var(--text-3)', fontSize: 12 }}>
        {children}
      </div>
      {onBack && <button className="btn ghost" onClick={onBack}>← Назад</button>}
      <button className="btn primary" disabled={nextDisabled} onClick={onNext}>
        {nextLabel} <span style={{ opacity: 0.5, marginLeft: 4 }}>→</span>
      </button>
    </div>
  );
}

// ---------- Tabs ----------
function Tabs({ tabs, active, setActive }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <div
          key={t.key}
          className={`tab ${active === t.key ? 'active' : ''}`}
          onClick={() => setActive(t.key)}
        >
          {t.label}
        </div>
      ))}
    </div>
  );
}

// ---------- Color swatch (for hex preview) ----------
function ColorSwatch({ value }) {
  if (!value) return null;
  const isHex = /^#[0-9a-f]{6}$/i.test(value);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      {isHex && (
        <span style={{
          display: 'inline-block', width: 9, height: 9, borderRadius: 2,
          background: value, border: '1px solid var(--border-strong)',
        }}></span>
      )}
      <span>{value}</span>
    </span>
  );
}

// expose
Object.assign(window, {
  Sidebar, Topbar, StatusBar, BrandMark, ScreenFooter, Tabs, ColorSwatch, STEPS,
});
