import { useState, useEffect, useRef } from 'react'

const API = import.meta.env.VITE_API_URL ?? '/api'

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
  const [intensityMin, setIntensityMin] = useState(0)
  const [intensityMax, setIntensityMax] = useState(255)
  const canvasRef = useRef(null)
  const [viewerHeight, setViewerHeight] = useState(480)
  const dragState = useRef(null)
  const [stripHeight, setStripHeight] = useState(160)

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

  useEffect(() => {
    if (!result || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    let cancelled = false

    if (activeTab === 'original') {
      const img = new Image()
      img.onload = () => {
        if (cancelled) return
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        ctx.drawImage(img, 0, 0)
      }
      img.src = `data:image/png;base64,${result.original}`
      return () => { cancelled = true }
    }

    const origImg = new Image()
    const maskImg = new Image()
    let origLoaded = false
    let maskLoaded = false

    const render = () => {
      if (!origLoaded || !maskLoaded || cancelled) return
      const w = origImg.naturalWidth
      const h = origImg.naturalHeight
      canvas.width = w
      canvas.height = h

      const offO = new OffscreenCanvas(w, h)
      const ctxO = offO.getContext('2d')
      ctxO.drawImage(origImg, 0, 0)
      const origPx = ctxO.getImageData(0, 0, w, h).data

      const offM = new OffscreenCanvas(w, h)
      const ctxM = offM.getContext('2d')
      ctxM.drawImage(maskImg, 0, 0)
      const maskPx = ctxM.getImageData(0, 0, w, h).data

      const out = ctx.createImageData(w, h)
      for (let i = 0; i < w * h; i++) {
        const p = i * 4
        const gray = origPx[p]
        const mr = maskPx[p], mg = maskPx[p + 1], mb = maskPx[p + 2]
        // Use nearest-colour matching so LANCZOS-blended edge pixels still classify correctly
        const dCrack = (mr-255)**2 +  mg**2       +  mb**2
        const dShape = (mr-255)**2 + (mg-215)**2  +  mb**2
        const dBg    =  mr**2      +  mg**2       +  mb**2
        const isCrack = dCrack < dShape && dCrack < dBg && dCrack < 40000
        const isShape = dShape < dCrack && dShape < dBg && dShape < 40000
        const showLabel = (isCrack && showClasses.crack) || (isShape && showClasses.shape)
        const inRange = gray >= intensityMin && gray <= intensityMax

        if (showLabel && inRange) {
          if (activeTab === 'overlay') {
            out.data[p]     = Math.round(gray * 0.6 + mr * 0.4)
            out.data[p + 1] = Math.round(gray * 0.6 + mg * 0.4)
            out.data[p + 2] = Math.round(gray * 0.6 + mb * 0.4)
          } else {
            out.data[p] = mr; out.data[p + 1] = mg; out.data[p + 2] = mb
          }
        } else if (activeTab === 'overlay') {
          out.data[p] = gray; out.data[p + 1] = gray; out.data[p + 2] = gray
        }
        out.data[p + 3] = 255
      }
      ctx.putImageData(out, 0, 0)
    }

    origImg.onload = () => { origLoaded = true; render() }
    maskImg.onload = () => { maskLoaded = true; render() }
    origImg.src = `data:image/png;base64,${result.original}`
    maskImg.src = `data:image/png;base64,${result.mask}`

    return () => { cancelled = true }
  }, [result, activeTab, intensityMin, intensityMax, showClasses])

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
          show_classes: ['crack', 'shape'],
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
      setIntensityMin(0)
      setIntensityMax(255)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

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
          <div className="sidebar-scroll">
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
          </div>

          <div className="predict-btn-wrap">
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
          </div>
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
            <div className="result-wrap">
              {/* Stats row */}}
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

              {/* Intensity filter */}
              <div className="intensity-section">
                <div className="intensity-header">
                  <span className="intensity-title">Intensity Filter</span>
                  <span className="intensity-value">{intensityMin} – {intensityMax}</span>
                </div>
                <div className="intensity-range-wrap">
                  <div className="intensity-track">
                    <div className="intensity-fill" style={{
                      left: `${(intensityMin / 255) * 100}%`,
                      width: `${((intensityMax - intensityMin) / 255) * 100}%`,
                    }} />
                  </div>
                  <input type="range" min="0" max="255" value={intensityMin}
                    onChange={(e) => setIntensityMin(Math.min(+e.target.value, intensityMax - 1))}
                    className="intensity-slider" />
                  <input type="range" min="0" max="255" value={intensityMax}
                    onChange={(e) => setIntensityMax(Math.max(+e.target.value, intensityMin + 1))}
                    className="intensity-slider" />
                </div>
                <div className="intensity-labels">
                  <span>Dark (0)</span>
                  <span>Bright (255)</span>
                </div>
                <p className="hint" style={{ marginTop: '6px' }}>
                  Only show labeled pixels where the original pixel intensity is within this range. Updates instantly without re-predicting.
                </p>
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
                  <canvas ref={canvasRef} className="viewer-img" />
                </div>
              </div>

              {/* Resize handle */}
              <div
                className="resize-handle"
                onMouseDown={(e) => {
                  e.preventDefault()
                  dragState.current = { startY: e.clientY, startH: stripHeight }
                  const onMove = (ev) => {
                    const delta = ev.clientY - dragState.current.startY
                    setStripHeight(Math.max(100, Math.min(700, dragState.current.startH - delta)))
                  }
                  const onUp = () => {
                    window.removeEventListener('mousemove', onMove)
                    window.removeEventListener('mouseup', onUp)
                    dragState.current = null
                  }
                  window.addEventListener('mousemove', onMove)
                  window.addEventListener('mouseup', onUp)
                }}
              >
                <span className="resize-handle-dots" />
              </div>

              {/* Side-by-side comparison strip */}
              <div className="strip-grid" style={{ height: stripHeight }}>
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
            </div>
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
