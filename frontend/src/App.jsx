import { useState, useEffect, useRef } from 'react'

const API = import.meta.env.VITE_API_URL ?? '/api'

const RESOLUTIONS = [256, 512, 800, 1600]
const GRIDS = [2, 3, 4]

/**
 * Draws a dimmed region outside the detected circle onto an existing canvas context.
 * Also draws a thin circle boundary ring.
 */
function drawCircleOverlay(ctx, w, h, circle, opacity) {
  const { cx, cy, radius } = circle
  ctx.save()
  // Darken area outside the circle using evenodd fill rule
  ctx.beginPath()
  ctx.rect(0, 0, w, h)
  ctx.arc(cx, cy, radius, 0, Math.PI * 2, true) // true = counterclockwise (cuts a hole)
  ctx.fillStyle = `rgba(0, 0, 0, ${opacity})`
  ctx.fill('evenodd')
  // Draw the circle boundary as a thin ring
  ctx.beginPath()
  ctx.arc(cx, cy, radius, 0, Math.PI * 2)
  ctx.strokeStyle = 'rgba(120, 210, 255, 0.85)'
  ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) / 400))
  ctx.stroke()
  ctx.restore()
}

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
  // Post-processing / crack filter
  const [closeGapSize, setCloseGapSize] = useState(0)
  const [minCrackArea, setMinCrackArea] = useState(0)
  const [maxCircularity, setMaxCircularity] = useState(1.0)
  const [circleMask, setCircleMask] = useState(false)
  const [circleMaskMargin, setCircleMaskMargin] = useState(0)
  const [circleMaskOffsetX, setCircleMaskOffsetX] = useState(0)
  const [circleMaskOffsetY, setCircleMaskOffsetY] = useState(0)
  const [circleOverlayOpacity, setCircleOverlayOpacity] = useState(0.35)
  const hasResultRef = useRef(false)  // guards filter effect from running before first predict
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

  // Debounced filter: re-apply post-processing on the cached mask whenever
  // filter params change, without re-running model inference.
  useEffect(() => {
    if (!hasResultRef.current) return  // no prediction run yet
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/filter`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            close_gap_size: closeGapSize,
            min_crack_area: minCrackArea,
            max_circularity: maxCircularity,
            circle_mask: circleMask,
            circle_mask_margin: circleMaskMargin,
            circle_mask_offset_x: circleMaskOffsetX,
            circle_mask_offset_y: circleMaskOffsetY,
            intensity_min: intensityMin,
            intensity_max: intensityMax,
          }),
        })
        if (!res.ok) return
        const data = await res.json()
        setResult((prev) => prev ? { ...prev, mask: data.mask, overlay: data.overlay, stats: data.stats, circle: data.circle } : prev)
      } catch {}
    }, 350)
    return () => clearTimeout(timer)
  }, [closeGapSize, minCrackArea, maxCircularity, circleMask, circleMaskMargin, circleMaskOffsetX, circleMaskOffsetY, intensityMin, intensityMax])  // eslint-disable-line react-hooks/exhaustive-deps

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
        if (circleMask && result.circle && circleOverlayOpacity > 0) {
          drawCircleOverlay(ctx, canvas.width, canvas.height, result.circle, circleOverlayOpacity)
        }
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

        if (showLabel) {
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
      if (circleMask && result.circle && circleOverlayOpacity > 0) {
        drawCircleOverlay(ctx, w, h, result.circle, circleOverlayOpacity)
      }
    }

    origImg.onload = () => { origLoaded = true; render() }
    maskImg.onload = () => { maskLoaded = true; render() }
    origImg.src = `data:image/png;base64,${result.original}`
    maskImg.src = `data:image/png;base64,${result.mask}`

    return () => { cancelled = true }
  }, [result, activeTab, intensityMin, intensityMax, showClasses, circleMask, circleOverlayOpacity])

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
          close_gap_size: closeGapSize,
          min_crack_area: minCrackArea,
          max_circularity: maxCircularity,
          circle_mask: circleMask,
          circle_mask_margin: circleMaskMargin,
          circle_mask_offset_x: circleMaskOffsetX,
          circle_mask_offset_y: circleMaskOffsetY,
          intensity_min: intensityMin,
          intensity_max: intensityMax,
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
      hasResultRef.current = true
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

          <div className="filter-section-divider" />

          <Section label="Crack Post-Processing">
            <p className="hint" style={{ marginBottom: 6 }}>
              Filters applied to crack predictions only — removes blobs and noise,
              keeps linear structures. Updates instantly after predict.
            </p>

            {/* 1. Intensity filter */}
            <div className="filter-row">
              <div className="intensity-header">
                <span className="filter-label">Intensity Range</span>
                <span className="filter-value">{intensityMin} – {intensityMax}</span>
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
                  disabled={loading}
                  className="intensity-slider" />
                <input type="range" min="0" max="255" value={intensityMax}
                  onChange={(e) => setIntensityMax(Math.max(+e.target.value, intensityMin + 1))}
                  disabled={loading}
                  className="intensity-slider" />
              </div>
              <div className="intensity-labels">
                <span>Dark (0)</span>
                <span>Bright (255)</span>
              </div>
              <p className="hint" style={{ marginTop: 4 }}>
                Only keep crack labels where the underlying pixel brightness is
                within this range. Runs first — before gap-closing and area filters.
              </p>
            </div>

            {/* 2. Gap closing */}
            <div className="filter-row">
              <div className="filter-row-header">
                <span className="filter-label">Connect Gaps</span>
                <span className="filter-value">{closeGapSize === 0 ? 'off' : `r=${closeGapSize}`}</span>
              </div>
              <input
                type="range"
                min="0"
                max="15"
                step="1"
                value={closeGapSize}
                onChange={(e) => setCloseGapSize(parseInt(e.target.value))}
                disabled={loading}
                className="threshold-slider"
              />
              <div className="threshold-labels">
                <span>Off</span>
                <span>r=15</span>
              </div>
              <p className="hint">
                Merges crack segments that are within this radius of each other — runs
                before other filters so nearby spots survive the area/circularity gates.
              </p>
            </div>

            {/* 3. Min area */}
            <div className="filter-row">
              <div className="filter-row-header">
                <span className="filter-label">Min Crack Area</span>
                <span className="filter-value">{minCrackArea === 0 ? 'off' : `${minCrackArea} px`}</span>
              </div>
              <input
                type="range"
                min="0"
                max="500"
                step="5"
                value={minCrackArea}
                onChange={(e) => setMinCrackArea(parseInt(e.target.value))}
                disabled={loading}
                className="threshold-slider"
              />
              <div className="threshold-labels">
                <span>Off</span>
                <span>500 px</span>
              </div>
              <p className="hint">
                Discard connected components smaller than this pixel area — removes tiny spots.
              </p>
            </div>

            {/* 4. Circularity */}
            <div className="filter-row">
              <div className="filter-row-header">
                <span className="filter-label">Max Circularity</span>
                <span className="filter-value">{maxCircularity >= 1.0 ? 'off' : maxCircularity.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={maxCircularity}
                onChange={(e) => setMaxCircularity(parseFloat(e.target.value))}
                disabled={loading}
                className="threshold-slider"
              />
              <div className="threshold-labels">
                <span>Strict (0.0)</span>
                <span>Off (1.0)</span>
              </div>
              <p className="hint">
                Discard components where circularity (4π·area/perimeter²) exceeds this value.
                Round blobs score ≈1.0; cracks, lines, and crack networks (crosses, grids) score
                much lower and are always preserved.
              </p>
            </div>

            {/* 5. Circle mask */}
            <div className="filter-row">
              <div className="toggle-row">
                <span className="filter-label">Circle Sample Mask</span>
                <button
                  className={`toggle ${circleMask ? 'toggle-on' : ''}`}
                  onClick={() => setCircleMask((v) => !v)}
                  disabled={loading}
                  aria-pressed={circleMask}
                >
                  <span className="toggle-thumb" />
                </button>
              </div>
              <p className="hint">
                Auto-detects the circular sample boundary and removes all crack
                labels outside it.
              </p>
              {circleMask && (
                <>
                  <div className="filter-row-header" style={{ marginTop: 8 }}>
                    <span className="filter-label">Offset X</span>
                    <span className="filter-value">{circleMaskOffsetX > 0 ? `+${circleMaskOffsetX}` : circleMaskOffsetX} px</span>
                  </div>
                  <input
                    type="range"
                    min="-200"
                    max="200"
                    step="5"
                    value={circleMaskOffsetX}
                    onChange={(e) => setCircleMaskOffsetX(parseInt(e.target.value))}
                    disabled={loading}
                    className="threshold-slider"
                  />
                  <div className="threshold-labels">
                    <span>← Left</span>
                    <span>Right →</span>
                  </div>
                  <div className="filter-row-header" style={{ marginTop: 8 }}>
                    <span className="filter-label">Offset Y</span>
                    <span className="filter-value">{circleMaskOffsetY > 0 ? `+${circleMaskOffsetY}` : circleMaskOffsetY} px</span>
                  </div>
                  <input
                    type="range"
                    min="-200"
                    max="200"
                    step="5"
                    value={circleMaskOffsetY}
                    onChange={(e) => setCircleMaskOffsetY(parseInt(e.target.value))}
                    disabled={loading}
                    className="threshold-slider"
                  />
                  <div className="threshold-labels">
                    <span>↑ Up</span>
                    <span>Down ↓</span>
                  </div>
                  <div className="filter-row-header" style={{ marginTop: 8 }}>
                    <span className="filter-label">Circle Margin</span>
                    <span className="filter-value">{circleMaskMargin > 0 ? `+${circleMaskMargin}` : circleMaskMargin} px</span>
                  </div>
                  <input
                    type="range"
                    min="-100"
                    max="100"
                    step="5"
                    value={circleMaskMargin}
                    onChange={(e) => setCircleMaskMargin(parseInt(e.target.value))}
                    disabled={loading}
                    className="threshold-slider"
                  />
                  <div className="threshold-labels">
                    <span>Shrink (−100)</span>
                    <span>Expand (+100)</span>
                  </div>
                  <div className="filter-row-header" style={{ marginTop: 8 }}>
                    <span className="filter-label">Outside Opacity</span>
                    <span className="filter-value">{Math.round(circleOverlayOpacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={circleOverlayOpacity}
                    onChange={(e) => setCircleOverlayOpacity(parseFloat(e.target.value))}
                    disabled={loading}
                    className="threshold-slider"
                  />
                  <div className="threshold-labels">
                    <span>Transparent</span>
                    <span>Opaque</span>
                  </div>
                </>
              )}
            </div>
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

              {/* Intensity filter — now in sidebar post-processing section */}

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
