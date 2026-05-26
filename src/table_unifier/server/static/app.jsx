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
             reviewDecisions: obj.reviewDecisions || {} };
  } catch (_e) { return null; }
}
function _saveApp(step, completed, reviewDecisions) {
  try {
    sessionStorage.setItem(APP_KEY, JSON.stringify({
      step, completed: [...completed], reviewDecisions,
    }));
  } catch (_e) { /* noop */ }
}

function App() {
  const _saved = _loadApp() || { step: 0, completed: new Set(), reviewDecisions: {} };
  const [step, setStep] = useState(_saved.step);
  const [completed, setCompleted] = useState(_saved.completed);
  const [reviewDecisions, setReviewDecisions] = useState(_saved.reviewDecisions);
  useEffect(() => { _saveApp(step, completed, reviewDecisions); },
            [step, completed, reviewDecisions]);

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

  // sidebar stats — dependent on current step
  const stats = {
    'tables': '2',
    'rows in': '27',
    'tokens': step >= 1 ? '41' : '—',
    'edges': step >= 1 ? '108' : '—',
    'clusters': step >= 3 ? '18' : '—',
    'unified rows': step >= 4 ? '18' : '—',
  };

  return (
    <div className="app-shell">
      <Topbar stepIdx={step} />
      <Sidebar current={step} setCurrent={setStep} completed={completed} stats={stats} />
      <div className="main">
        {step === 0 && <ScreenUpload onContinue={advance} />}
        {step === 1 && <ScreenGraph    onBack={goBack} onContinue={advance} />}
        {step === 2 && <ScreenTraining onBack={goBack} onContinue={advance} />}
        {step === 3 && <ScreenReview   onBack={goBack} onContinue={advance}
                                       decisions={reviewDecisions}
                                       setDecisions={setReviewDecisions} />}
        {step === 4 && <ScreenResult   onBack={goBack} onRestart={reset}
                                       decisions={reviewDecisions} />}
        <StatusBar stepIdx={step} running={step === 1 || step === 2} />
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
