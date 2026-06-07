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
  { key: 'graph',   label: 'Граф',          sub: 'узлы строк и токенов' },
  { key: 'infer',   label: 'Инференс GNN',  sub: 'передача сообщений · L=2' },
  { key: 'review',  label: 'Проверка пар',     sub: 'кандидаты-дубликаты' },
  { key: 'result',  label: 'Результат',     sub: 'единая таблица + экспорт' },
];

function Sidebar({ current, setCurrent, completed, stats }) {
  return (
    <aside className="sidebar">
      <h6>Этапы</h6>
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

// ---- UI scale (projector legibility) ----
function UIScale() {
  const MIN = 1.0, MAX = 1.8, STEP = 0.1, DEF = 1.25;
  const read = () => {
    let v = NaN;
    try { v = parseFloat(localStorage.getItem('tableunifier:uiscale')); } catch (_e) { /* noop */ }
    if (isNaN(v)) {
      v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--ui-zoom')) || DEF;
    }
    return v;
  };
  const [z, setZ] = useState(read);
  // Read the live applied value (not the React closure) so rapid clicks compound.
  const bump = (delta) => {
    const cur = parseFloat(
      document.documentElement.style.getPropertyValue('--ui-zoom')
    ) || z;
    const nz = Math.round(Math.min(MAX, Math.max(MIN, cur + delta)) * 100) / 100;
    document.documentElement.style.setProperty('--ui-zoom', String(nz));
    try { localStorage.setItem('tableunifier:uiscale', String(nz)); } catch (_e) { /* noop */ }
    setZ(nz);
  };
  const btnStyle = {
    height: '100%', width: 24, padding: 0, borderRadius: 0,
    justifyContent: 'center', fontSize: 15, color: 'var(--text-2)',
  };
  return (
    <div title="Масштаб интерфейса"
      style={{ display: 'flex', alignItems: 'center', height: 24,
               border: '1px solid var(--border)', borderRadius: 6,
               background: 'var(--surface)', overflow: 'hidden' }}>
      <button className="btn ghost" style={btnStyle} onClick={() => bump(-STEP)}
        disabled={z <= MIN + 1e-6} aria-label="Меньше">−</button>
      <span className="mono" style={{ minWidth: 40, textAlign: 'center', fontSize: 11, color: 'var(--text-2)' }}>
        {Math.round(z * 100)}%
      </span>
      <button className="btn ghost" style={btnStyle} onClick={() => bump(STEP)}
        disabled={z >= MAX - 1e-6} aria-label="Больше">+</button>
    </div>
  );
}

// ---- Theme toggle ----
function ThemeToggle() {
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute('data-theme') || 'dark'
  );
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('tableunifier:theme', next); } catch (_e) { /* noop */ }
    setTheme(next);
  };
  const isDark = theme === 'dark';
  return (
    <button
      className="btn icon"
      onClick={toggle}
      title={isDark ? 'Переключить на светлую тему' : 'Переключить на тёмную тему'}
      aria-label="Сменить тему"
      style={{ color: 'var(--text-2)' }}>
      {isDark ? (
        // sun (action: switch to light)
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="4.4" fill="currentColor" stroke="none" />
          <line x1="12" y1="1.5" x2="12" y2="4" />
          <line x1="12" y1="20" x2="12" y2="22.5" />
          <line x1="1.5" y1="12" x2="4" y2="12" />
          <line x1="20" y1="12" x2="22.5" y2="12" />
          <line x1="4.3" y1="4.3" x2="6" y2="6" />
          <line x1="18" y1="18" x2="19.7" y2="19.7" />
          <line x1="19.7" y1="4.3" x2="18" y2="6" />
          <line x1="6" y1="18" x2="4.3" y2="19.7" />
        </svg>
      ) : (
        // moon (action: switch to dark) — crescent from two circles
        <svg width="15" height="15" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="8.5" fill="currentColor" />
          <circle cx="15.5" cy="9.2" r="7" fill="var(--bg)" />
        </svg>
      )}
    </button>
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
      <UIScale />
      <ThemeToggle />
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
      <span>{running ? <><span className="spinner" style={{ verticalAlign: '-2px' }}></span> &nbsp;выполняется</> : 'ожидание'}</span>
      <div className="spacer"></div>
      <span>шаг {stepIdx + 1}/{STEPS.length}</span>
      <span style={{ color: 'var(--border-strong)' }}>│</span>
      <span>↑↓ навигация · ↵ далее</span>
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
  Sidebar, Topbar, StatusBar, BrandMark, ScreenFooter, Tabs, ColorSwatch, ThemeToggle, UIScale, STEPS,
});
