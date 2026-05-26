// Screen 1 — Upload Excel/CSV/Parquet files via real backend API.

function TablePreview({ tbl, accentColor }) {
  if (!tbl) return null;
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
            {(tbl.data || []).slice(0, 50).map((row, i) => (
              <tr key={i}>
                <td className="nrow">{i + 1}</td>
                {tbl.cols.map((c, j) => {
                  const v = Array.isArray(row) ? row[j] : row[c];
                  if (/^color/i.test(c)) return <td key={j}><ColorSwatch value={String(v ?? '')} /></td>;
                  return <td key={j}>{v == null ? '' : String(v)}</td>;
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
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState(null);
  const [divergence, setDivergence] = useState([]);
  const [, force] = useState(0);
  const inputRef = useRef(null);

  const uploadBatch = async (chosen) => {
    setUploading(true);
    setUploadErr(null);
    try {
      const res = await window.API.uploadFiles(chosen);
      setFiles(res.sources.map((s, i) => ({
        name: s.name, size: s.size_bytes, ref: i === 0 ? 'A' : i === 1 ? 'B' : null,
        id: s.id,
      })));
      setDivergence(res.divergence || []);
      force((t) => t + 1);
    } catch (e) {
      setUploadErr(String(e.message || e));
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer?.files || []);
    if (dropped.length) uploadBatch(dropped);
  };

  const D = window.__DATA__;
  const tableA = D.tableA;
  const tableB = D.tableB;

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <h1>Источники данных</h1>
          <p>Загрузите Excel/CSV-файлы для объединения. TableUnifier автоматически выявит общие сущности через embedding-based entity resolution.</p>
        </div>
      </div>

      <div className="screen-body" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current && inputRef.current.click()}>
            <div className="icon"></div>
            <div className="big">{uploading ? 'Загрузка…' : 'Перетащите .xlsx / .csv / .parquet'}</div>
            <div className="small">или нажмите чтобы выбрать</div>
            <input ref={inputRef} type="file" multiple accept=".xlsx,.csv,.parquet" hidden
              onChange={(e) => {
                const chosen = Array.from(e.target.files || []);
                if (chosen.length) uploadBatch(chosen);
              }} />
          </div>

          <div className="panel">
            <div className="panel-h">
              <h3>Загружено</h3>
              <span className="sub">{files.length} файлов</span>
            </div>
            <div style={{ padding: '4px 0' }}>
              {files.length === 0 && (
                <div style={{ padding: 18, fontSize: 11.5, color: 'var(--text-4)' }}>
                  Нет загруженных файлов.
                </div>
              )}
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
                    <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                  fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{f.name}</div>
                    <div style={{ color: 'var(--text-4)', fontSize: 10.5, fontFamily: 'var(--font-mono)' }}>
                      {(f.size / 1024).toFixed(1)} KB · parsed
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {uploadErr && (
          <div className="panel" style={{ padding: '10px 14px', color: 'var(--bad)', fontSize: 12 }}>
            Ошибка: {uploadErr}
          </div>
        )}

        {divergence.length > 0 && (
          <div className="panel" style={{ padding: '12px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
              <div style={{
                width: 26, height: 26, borderRadius: 6, flexShrink: 0,
                background: 'var(--warn-soft)', color: 'var(--warn)',
                display: 'grid', placeItems: 'center', fontFamily: 'var(--font-mono)', fontWeight: 600,
              }}>!</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, fontWeight: 500, marginBottom: 3 }}>
                  Расхождение схем — {divergence.length} соответствий
                </div>
                <div style={{ color: 'var(--text-3)', fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>
                  {divergence.slice(0, 6).map((d, i) => (
                    <span key={i}>
                      {d.a_col}↔{d.b_col} <span style={{ opacity: 0.6 }}>({d.kind})</span>
                      {i < Math.min(divergence.length, 6) - 1 ? ' · ' : ''}
                    </span>
                  ))}
                  {divergence.length > 6 && '…'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <span className="chip">qwen3-emb · 4096d</span>
                <span className="chip warn"><span className="dot"></span>auto-match</span>
              </div>
            </div>
          </div>
        )}

        {tableA && tableB && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, minHeight: 0, flex: 1 }}>
            <TablePreview tbl={tableA} accentColor="var(--row)" />
            <TablePreview tbl={tableB} accentColor="var(--token)" />
          </div>
        )}
      </div>

      <ScreenFooter
        onNext={onContinue}
        nextLabel="Перейти к графу"
        nextDisabled={!(tableA && tableB) || uploading}>
        {tableA && tableB
          ? <>Готово: <b style={{ color: 'var(--text-2)' }}>{tableA.rows + tableB.rows} строк</b>, общий словарь токенов будет построен дальше.</>
          : 'Загрузите как минимум 2 таблицы.'}
      </ScreenFooter>
    </div>
  );
}

Object.assign(window, { ScreenUpload });
