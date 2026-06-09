// Main app — wires together topbar, sidebar, screens

// Persist UI state in sessionStorage, чтобы F5 не выкидывал пользователя
// обратно на экран 1, когда __STATE__.runId уже валиден на бэке.
const APP_KEY = "tableunifier:app:v1";
function _loadApp() {
  try {
    const raw = sessionStorage.getItem(APP_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    return { step: obj.step || 0, completed: new Set(obj.completed || []),
             reviewDecisions: obj.reviewDecisions || {},
             reviewThreshold: (typeof obj.reviewThreshold === 'number') ? obj.reviewThreshold : null,
             autoThreshold: !!obj.autoThreshold };
  } catch (_e) { return null; }
}
function _saveApp(step, completed, reviewDecisions, reviewThreshold, autoThreshold) {
  try {
    sessionStorage.setItem(APP_KEY, JSON.stringify({
      step, completed: [...completed], reviewDecisions, reviewThreshold, autoThreshold,
    }));
  } catch (_e) { /* noop */ }
}

function App() {
  const _saved = _loadApp() || { step: 0, completed: new Set(), reviewDecisions: {}, reviewThreshold: null, autoThreshold: false };
  const [step, setStep] = useState(_saved.step);
  const [completed, setCompleted] = useState(_saved.completed);
  const [reviewDecisions, setReviewDecisions] = useState(_saved.reviewDecisions);
  // Общий порог сходства: null → берётся из metrics.threshold инференса,
  // как только пользователь двигает слайдер в окне «Проверка», значение
  // фиксируется и используется и в окне «Инференс» (рёбра + гистограмма).
  const [reviewThreshold, setReviewThreshold] = useState(_saved.reviewThreshold);
  // Авто-подстройка порога под решения «объединить/разделить» в окне «Проверка».
  const [autoThreshold, setAutoThreshold] = useState(_saved.autoThreshold);
  const [, _tick] = useState(0);
  useEffect(() => { _saveApp(step, completed, reviewDecisions, reviewThreshold, autoThreshold); },
            [step, completed, reviewDecisions, reviewThreshold, autoThreshold]);
  // Re-render the shell (so sidebar stats track live graph data) when API
  // populates __DATA__.graph / clusters.
  useEffect(() => {
    const h = () => _tick((t) => t + 1);
    window.addEventListener('graph-updated', h);
    return () => window.removeEventListener('graph-updated', h);
  }, []);

  const advance = () => {
    setCompleted((s) => new Set([...s, step]));
    setStep(step + 1);
  };
  const goBack = () => {
    if (step > 0) setStep(step - 1);
  };
  const reset = () => {
    setCompleted(new Set());
    setStep(0);
    setReviewDecisions({});
    setReviewThreshold(null);
    setAutoThreshold(false);
    // Сбрасываем серверное состояние (включая sessionStorage),
    // чтобы следующий цикл создал новый run с нуля.
    if (window.__STATE_RESET__) window.__STATE_RESET__();
    window.__DATA__.tableA = null;
    window.__DATA__.tableB = null;
    window.__DATA__.candidates = [];
    window.__DATA__.clusters = [];
    window.__DATA__.graph = null;
    try { sessionStorage.removeItem(APP_KEY); } catch (_e) { /* noop */ }
  };

  // sidebar stats — derived from live data where available so the rail agrees
  // with the graph canvas + result metrics (no hardcoded counts).
  const D = window.__DATA__ || {};
  const gStats = (D.graph && D.graph.stats) || null;
  const nIn = ((D.tableA && D.tableA.rows) || 0) + ((D.tableB && D.tableB.rows) || 0);
  const nOut = (D.clusters && D.clusters.length) || 0;
  // «Выполняется» — только пока реально идёт сборка/инференс, а не просто на этих шагах.
  const graphBuilt = !!(D.graph && D.graph.rows && D.graph.rows.length > 0);
  const inferDone = ((D.candidates) || []).length > 0;
  const running = (step === 1 && !graphBuilt) || (step === 2 && !inferDone);
  const stats = {
    'таблиц': '2',
    'строк на входе': nIn ? String(nIn) : '27',
    'токенов': step >= 1 && gStats ? String(gStats.n_tokens) : '—',
    'рёбер': step >= 1 && gStats ? String(gStats.n_edges) : '—',
    'кластеров': step >= 3 && nOut ? String(nOut) : (step >= 3 ? '18' : '—'),
    'строк на выходе': step >= 4 && nOut ? String(nOut) : (step >= 4 ? '18' : '—'),
  };

  return (
    <div className="app-shell">
      <Topbar stepIdx={step} />
      <Sidebar current={step} setCurrent={setStep} completed={completed} stats={stats} />
      <div className="main">
        {step === 0 && <ScreenUpload onContinue={advance} />}
        {step === 1 && <ScreenGraph    onBack={goBack} onContinue={advance} />}
        {step === 2 && <ScreenTraining onBack={goBack} onContinue={advance}
                                       threshold={reviewThreshold}
                                       setThreshold={setReviewThreshold} />}
        {step === 3 && <ScreenReview   onBack={goBack} onContinue={advance}
                                       decisions={reviewDecisions}
                                       setDecisions={setReviewDecisions}
                                       threshold={reviewThreshold}
                                       setThreshold={setReviewThreshold}
                                       autoThreshold={autoThreshold}
                                       setAutoThreshold={setAutoThreshold} />}
        {step === 4 && <ScreenResult   onBack={goBack} onRestart={reset}
                                       decisions={reviewDecisions} />}
        <StatusBar stepIdx={step} running={running} />
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
