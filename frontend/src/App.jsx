import { useState, useEffect } from 'react'

const API = '/api'

const RESOLUTIONS = [256, 512, 800, 1600]
const GRIDS = [2, 3, 4]

export default function App() {
  const [models, setModels] = useState([])
  const [samples, setSamples] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedSample, setSelectedSample] = useState('')
  const [resolution, setResolution] = useState(800)
  const [split, setSplit] = useState(false)
  const [splitGrid, setSplitGrid] = useState(2)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5)
  const [showClasses, setShowClasses] = useState({ crack: true, shape: true })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overlay')

  useEffect(() => {
    fetch(`${API}/models`)
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models)
        if (data.models.length > 0) setSelectedModel(data.models[0])
      })
      .catch(() => setError('Could not connect to backend. Is the server running?'))

    fetch(`${API}/samples`)
      .then((r) => r.json())
      .then((data) => {
        setSamples(data.samples)
        if (data.samples.length > 0) setSelectedSample(data.samples[0])
      })
      .catch(() => {})
  }, [])

  const handlePredict = async () => {
    if (!selectedModel || !selectedSample) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          sample: selectedSample,
          resolution,
          split,
          split_grid: splitGrid,
          confidence_threshold: confidenceThreshold,
          show_classes: Object.keys(showClasses).filter((k) => showClasses[k]),
        }),
      })

      if (!res.ok) {
        let errMessage = 'Prediction failed'
        try {
          const err = await res.json()
          errMessage = err.detail || errMessage
        } catch {
          errMessage = (await res.text().catch(() => '')) || errMessage
        }
        throw new Error(errMessage)
      }

      const data = await res.json()
      setResult(data)
      setActiveTab('overlay')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const tabImage = result
    ? { overlay: result.overlay, mask: result.mask, original: result.original }[activeTab]
    : null

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div>
            <h1 className="header-title">Crack Detection</h1>
            <p className="header-sub">UNet-based segmentation — select options and predict</p>
          </div>
          {result && (
            <div className="header-badges">
              {result.stats.has_crack && <span className="badge crack-badge">Crack detected</span>}
              {result.stats.has_shape && <span className="badge shape-badge">Shape detected</span>}
              {!result.stats.has_crack && !result.stats.has_shape && (
                <span className="badge ok-badge">No defects</span>
              )}
            </div>
          )}
        </div>
      </header>

      <div className="layout">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <Section label="Model">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m.replace('.pt', '')}
                </option>
              ))}
            </select>
          </Section>

          <Section label="Sample">
            <select
              value={selectedSample}
              onChange={(e) => setSelectedSample(e.target.value)}
              disabled={loading}
            >
              {samples.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </Section>

          <Section label="Inference Resolution">
            <div className="chip-group">
              {RESOLUTIONS.map((r) => (
                <button
                  key={r}
                  className={`chip ${resolution === r ? 'chip-active' : ''}`}
                  onClick={() => setResolution(r)}
                  disabled={loading}
                >
                  {r}×{r}
                </button>
              ))}
            </div>
          </Section>

          <Section label="Split & Predict">
            <div className="toggle-row">
              <span className="toggle-desc">Split image into tiles</span>
              <button
                className={`toggle ${split ? 'toggle-on' : ''}`}
                onClick={() => setSplit((v) => !v)}
                disabled={loading}
                aria-pressed={split}
              >
                <span className="toggle-thumb" />
              </button>
            </div>
            <p className="hint">
              Divides the image into tiles, predicts each part at the chosen resolution, then
              stitches results together — useful for very large images.
            </p>
          </Section>

          {split && (
            <Section label="Grid Size">
              <div className="chip-group">
                {GRIDS.map((g) => (
                  <button
                    key={g}
                    className={`chip ${splitGrid === g ? 'chip-active' : ''}`}
                    onClick={() => setSplitGrid(g)}
                    disabled={loading}
                  >
                    {g}×{g}
                  </button>
                ))}
              </div>
            </Section>
          )}

          <Section label="Show Classes">
            <div className="class-toggles">
              <button
                className={`class-btn class-btn-crack ${showClasses.crack ? 'class-btn-active' : ''}`}
                onClick={() => setShowClasses((s) => ({ ...s, crack: !s.crack }))}
                disabled={loading}
              >
                <span className="class-dot" style={{ background: 'var(--red)' }} />
                Crack
              </button>
              <button
                className={`class-btn class-btn-shape ${showClasses.shape ? 'class-btn-active-shape' : ''}`}
                onClick={() => setShowClasses((s) => ({ ...s, shape: !s.shape }))}
                disabled={loading}
              >
                <span className="class-dot" style={{ background: 'var(--gold)' }} />
                Shape
              </button>
            </div>
            <p className="hint">Toggle which defect classes appear in the output mask and overlay.</p>
          </Section>

          <Section label={`Confidence Threshold — ${confidenceThreshold.toFixed(2)}`}>
            <input
              type="range"
              min="0.1"
              max="0.99"
              step="0.01"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              disabled={loading}
              className="threshold-slider"
            />
            <div className="threshold-labels">
              <span>Low (0.10)</span>
              <span>High (0.99)</span>
            </div>
            <p className="hint">
              Only label a pixel as crack/shape when the model's softmax
              confidence exceeds this value. Lower = more detections, higher =
              only high-confidence predictions.
            </p>
          </Section>

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading || !selectedModel || !selectedSample}
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                Running…
              </>
            ) : (
              'Predict'
            )}
          </button>
        </aside>

        {/* ── Main ── */}
        <main className="main">
          {error && (
            <div className="alert alert-error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="loading-screen">
              <div className="loading-spinner" />
              <p>Running inference…</p>
              {split && (
                <p className="loading-sub">
                  Processing {splitGrid}×{splitGrid} tiles at {resolution}×{resolution}
                </p>
              )}
            </div>
          )}

          {result && !loading && (
            <>
              {/* Stats row */}
              <div className="stats-row">
                <StatCard label="Crack Coverage" value={`${result.stats.line_percentage.toFixed(2)}%`} accent="var(--red)" />
                <StatCard label="Shape Coverage" value={`${result.stats.shape_percentage.toFixed(2)}%`} accent="var(--gold)" />
                <StatCard label="Total Defect" value={`${result.stats.defect_percentage.toFixed(2)}%`} accent="var(--orange)" />
                <StatCard
                  label="Image Size"
                  value={`${result.stats.image_size.width}×${result.stats.image_size.height}`}
                  accent="var(--blue)"
                />
              </div>

              {/* Legend */}
              <div className="legend">
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: 'var(--red)' }} />
                  Crack (line)
                </span>
                <span className="legend-item">
                  <span className="legend-dot" style={{ background: 'var(--gold)' }} />
                  Shape
                </span>
              </div>

              {/* Image viewer */}
              <div className="viewer">
                <div className="viewer-tabs">
                  {['overlay', 'mask', 'original'].map((tab) => (
                    <button
                      key={tab}
                      className={`viewer-tab ${activeTab === tab ? 'viewer-tab-active' : ''}`}
                      onClick={() => setActiveTab(tab)}
                    >
                      {tab.charAt(0).toUpperCase() + tab.slice(1)}
                    </button>
                  ))}
                </div>
                <div className="viewer-body">
                  <img
                    key={activeTab}
                    src={`data:image/png;base64,${tabImage}`}
                    alt={activeTab}
                    className="viewer-img"
                  />
                </div>
              </div>

              {/* Side-by-side comparison strip */}
              <div className="strip-grid">
                {['original', 'mask', 'overlay'].map((key) => (
                  <div
                    key={key}
                    className={`strip-card ${activeTab === key ? 'strip-card-active' : ''}`}
                    onClick={() => setActiveTab(key)}
                  >
                    <img
                      src={`data:image/png;base64,${result[key]}`}
                      alt={key}
                      className="strip-img"
                    />
                    <span className="strip-label">
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {!result && !loading && !error && (
            <div className="placeholder">
              <div className="placeholder-icon">
                <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  <path d="M11 8v6M8 11h6" />
                </svg>
              </div>
              <p>
                Choose a <strong>model</strong>, a <strong>sample</strong>, and your inference
                settings — then hit <strong>Predict</strong>.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

function Section({ label, children }) {
  return (
    <div className="section">
      <p className="section-label">{label}</p>
      {children}
    </div>
  )
}

function StatCard({ label, value, accent }) {
  return (
    <div className="stat-card" style={{ borderTopColor: accent }}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}
