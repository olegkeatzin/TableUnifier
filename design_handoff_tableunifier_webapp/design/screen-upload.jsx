// Screen 1 — Upload Excel files + preview tables

function TablePreview({ tbl, accentColor }) {
  return (
    <div className="table-card">
      <div className="table-card-h">
        <span className="dot" style={{ background: accentColor }}></span>
        <span className="name">{tbl.name}</span>
        <span className="meta">{tbl.rows} rows · {tbl.cols.length} cols</span>
      </div>
      <div className="table-card-body">
        <table className="dt">
          <thead>
            <tr>
              <th className="nrow">#</th>
              {tbl.cols.map((c) => <th key={c}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {tbl.data.map((row, i) => (
              <tr key={i}>
                <td className="nrow">{i + 1}</td>
                {row.map((v, j) => {
                  const col = tbl.cols[j];
                  if (col === 'color_hex' || col === 'color') {
                    return <td key={j}><ColorSwatch value={String(v)} /></td>;
                  }
                  return <td key={j}>{String(v)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ScreenUpload({ onContinue }) {
  const D = window.__DATA__;
  const [files, setFiles] = useState([
    { name: D.tableA.name, size: 18420, ok: true, ref: 'A' },
    { name: D.tableB.name, size: 16880, ok: true, ref: 'B' },
  ]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    // for prototype: pretend we accepted whatever was dropped
    const dropped = Array.from(e.dataTransfer?.files || []);
    if (dropped.length > 0) {
      const newFiles = dropped.map((f) => ({
        name: f.name, size: f.size, ok: true, ref: null,
      }));
      setFiles([...files, ...newFiles]);
    }
  };

  const sources = files.filter((f) => f.ref).map((f) => f.ref);
  const tableA = sources.includes('A') ? D.tableA : null;
  const tableB = sources.includes('B') ? D.tableB : null;

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Источники данных</h1>
          <p>Загрузите Excel-файлы для объединения. TableUnifier автоматически выявит общие сущности через embedding-based entity resolution, даже если колонки названы по-разному.</p>
        </div>
        <div className="actions">
          <button className="btn ghost">📋 Из буфера</button>
          <button className="btn ghost">⚙ Парсер...</button>
        </div>
      </div>

      <div className="screen-body" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 18 }}>

        {/* Dropzone + file list row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current && inputRef.current.click()}
          >
            <div className="icon"></div>
            <div className="big">Перетащите .xlsx файлы сюда</div>
            <div className="small">или нажмите чтобы выбрать. Поддерживается .xlsx · .csv · .parquet</div>
            <input ref={inputRef} type="file" multiple accept=".xlsx,.csv,.parquet" hidden
              onChange={(e) => {
                const dropped = Array.from(e.target.files || []);
                if (dropped.length > 0) {
                  const newFiles = dropped.map((f) => ({ name: f.name, size: f.size, ok: true, ref: null }));
                  setFiles([...files, ...newFiles]);
                }
              }} />
          </div>

          <div className="panel">
            <div className="panel-h">
              <h3>Загружено</h3>
              <span className="sub">{files.length} файлов</span>
            </div>
            <div style={{ padding: '4px 0' }}>
              {files.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '8px 14px', fontSize: 12,
                  borderBottom: i < files.length - 1 ? '1px solid var(--border)' : 'none',
                }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: 5,
                    background: f.ref === 'A' ? 'var(--row-soft)' : f.ref === 'B' ? 'var(--token-soft)' : 'var(--surface)',
                    border: '1px solid var(--border-strong)',
                    display: 'grid', placeItems: 'center',
                    fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 600,
                    color: f.ref === 'A' ? 'var(--row)' : f.ref === 'B' ? 'var(--token)' : 'var(--text-3)',
                  }}>{f.ref || '?'}</div>
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div style={{
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      fontFamily: 'var(--font-mono)', fontSize: 11.5,
                    }}>{f.name}</div>
                    <div style={{ color: 'var(--text-4)', fontSize: 10.5, fontFamily: 'var(--font-mono)' }}>
                      {(f.size / 1024).toFixed(1)} KB · {f.ok ? 'parsed ok' : 'parsing...'}
                    </div>
                  </div>
                  <button className="btn ghost icon" style={{ width: 22, height: 22, fontSize: 11 }}
                    onClick={() => setFiles(files.filter((_, j) => j !== i))}>×</button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* schema divergence callout */}
        {tableA && tableB && (
          <div className="panel" style={{ padding: '12px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
              <div style={{
                width: 26, height: 26, borderRadius: 6, flexShrink: 0,
                background: 'var(--warn-soft)', color: 'var(--warn)',
                display: 'grid', placeItems: 'center', fontFamily: 'var(--font-mono)', fontWeight: 600,
              }}>!</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, fontWeight: 500, marginBottom: 3 }}>
                  Обнаружено расхождение схем — 7 колонок имеют разные имена
                </div>
                <div style={{ color: 'var(--text-3)', fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>
                  mark↔brand · model↔model_name · mileage↔probeg_km · color↔color_hex (text vs HEX) · bodyType↔body_type (case)
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <span className="chip">qwen3-emb · 4096d</span>
                <span className="chip warn"><span className="dot"></span>auto-match</span>
              </div>
            </div>
          </div>
        )}

        {/* Preview tables */}
        {tableA && tableB && (
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16,
            minHeight: 0, flex: 1,
          }}>
            <TablePreview tbl={tableA} accentColor="var(--row)" />
            <TablePreview tbl={tableB} accentColor="var(--token)" />
          </div>
        )}
      </div>

      <ScreenFooter
        onNext={onContinue}
        nextLabel="Перейти к графу"
        nextDisabled={!(tableA && tableB)}>
        {tableA && tableB
          ? <>Готово к индексации: <b style={{ color: 'var(--text-2)' }}>{tableA.rows + tableB.rows} строк</b>, общий словарь токенов будет построен на следующем шаге.</>
          : 'Загрузите как минимум 2 таблицы для запуска entity resolution.'}
      </ScreenFooter>
    </div>
  );
}

Object.assign(window, { ScreenUpload });
