// Main app — wires together topbar, sidebar, screens

function App() {
  const [step, setStep] = useState(0);
  const [completed, setCompleted] = useState(new Set());
  const [reviewDecisions, setReviewDecisions] = useState({});

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
    // Сбрасываем серверное состояние, чтобы следующий цикл создал новый run.
    window.__STATE__.runId = null;
    window.__STATE__.sessionId = null;
    window.__STATE__.graphReady = false;
    window.__STATE__.inferDone = false;
    window.__STATE__.inferRunId = null;
    window.__DATA__.tableA = null;
    window.__DATA__.tableB = null;
    window.__DATA__.candidates = [];
    window.__DATA__.clusters = [];
    window.__DATA__.graph = null;
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
