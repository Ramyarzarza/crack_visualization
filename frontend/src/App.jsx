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

  // Detection mode
  const [detectionMode, setDetectionMode] = useState('synthetic') // 'synthetic' | 'unsupervised'
  const [unsupParams, setUnsupParams] = useState({
    method: 'frangi_sato',
    // Frangi / Sato
    fs_filter: 'sato', fs_sigma_min: 1, fs_sigma_max: 4,
    fs_threshold_method: 'percentile', fs_percentile: 93, fs_min_component_size: 100,
    // Matched Filter
    mf_n_orientations: 12, mf_sigma_x: 1.5, mf_sigma_y: 6.0, mf_kernel_size: 25,
    mf_threshold_method: 'percentile', mf_percentile: 97, mf_min_component_size: 100,
    // Top-Hat
    th_line_length: 40, th_n_orientations: 20,
    th_threshold_method: 'percentile', th_percentile: 97,
    th_min_component_size: 100, th_min_aspect_ratio: 2.0,
    // Attribute Filter
    af_bg_disk_radius: 15, af_threshold_method: 'otsu', af_adaptive_block: 51,
    af_min_area: 50, af_min_eccentricity: 0.90, af_min_axis_ratio: 3.0,
    af_max_circularity: 0.40, af_min_skeleton_length: 15,
  })
  const setP = (key, val) => setUnsupParams(p => ({ ...p, [key]: val }))

  // Test-set evaluation
  const [evalResult, setEvalResult]     = useState(null)
  const [evalLoading, setEvalLoading]   = useState(false)
  const [evalError, setEvalError]       = useState(null)
  const [showEvalModal, setShowEvalModal] = useState(false)
  const [evalPredKind, setEvalPredKind] = useState('filtered_pred')
  const [evalGtKind,   setEvalGtKind]   = useState('filtered_gt')
  const [evalDataset,  setEvalDataset]  = useState('labeling')
  const [benchmarks,   setBenchmarks]   = useState([])
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

    fetch(`${API}/evaluate/benchmarks`)
      .then((r) => r.json())
      .then((data) => setBenchmarks(data.benchmarks ?? []))
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
        if (!res.ok) {
          if (res.status === 400) {
            // Server cache was cleared (e.g. restart) — stop calling /filter until next predict
            hasResultRef.current = false
          }
          return
        }
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
          const imgScale = canvas.width / result.stats.image_size.width
          const scaledCircle = { cx: result.circle.cx * imgScale, cy: result.circle.cy * imgScale, radius: result.circle.radius * imgScale }
          drawCircleOverlay(ctx, canvas.width, canvas.height, scaledCircle, circleOverlayOpacity)
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
        const imgScale = w / result.stats.image_size.width
        const scaledCircle = { cx: result.circle.cx * imgScale, cy: result.circle.cy * imgScale, radius: result.circle.radius * imgScale }
        drawCircleOverlay(ctx, w, h, scaledCircle, circleOverlayOpacity)
      }
    }

    origImg.onload = () => { origLoaded = true; render() }
    maskImg.onload = () => { maskLoaded = true; render() }
    origImg.src = `data:image/png;base64,${result.original}`
    maskImg.src = `data:image/png;base64,${result.mask}`

    return () => { cancelled = true }
  }, [result, activeTab, intensityMin, intensityMax, showClasses, circleMask, circleOverlayOpacity])

  const handlePredict = async () => {
    if (!selectedSample) return
    if (detectionMode === 'synthetic' && !selectedModel) return
    setLoading(true)
    setError(null)
    setResult(null)

    const commonPostProc = {
      close_gap_size: closeGapSize, min_crack_area: minCrackArea,
      max_circularity: maxCircularity, circle_mask: circleMask,
      circle_mask_margin: circleMaskMargin, circle_mask_offset_x: circleMaskOffsetX,
      circle_mask_offset_y: circleMaskOffsetY,
      intensity_min: intensityMin, intensity_max: intensityMax,
    }

    try {
      let endpoint, body
      if (detectionMode === 'synthetic') {
        endpoint = `${API}/predict`
        body = {
          model: selectedModel, sample: selectedSample,
          resolution, split, split_grid: splitGrid,
          confidence_threshold: confidenceThreshold,
          show_classes: ['crack', 'shape'],
          ...commonPostProc,
        }
      } else {
        endpoint = `${API}/predict/unsupervised`
        body = {
          sample: selectedSample,
          method: unsupParams.method,
          ...commonPostProc,
          // Frangi / Sato
          fs_filter: unsupParams.fs_filter,
          fs_sigma_min: unsupParams.fs_sigma_min,
          fs_sigma_max: unsupParams.fs_sigma_max,
          fs_threshold_method: unsupParams.fs_threshold_method,
          fs_percentile: unsupParams.fs_percentile,
          fs_min_component_size: unsupParams.fs_min_component_size,
          // Matched Filter
          mf_n_orientations: unsupParams.mf_n_orientations,
          mf_sigma_x: unsupParams.mf_sigma_x,
          mf_sigma_y: unsupParams.mf_sigma_y,
          mf_kernel_size: unsupParams.mf_kernel_size,
          mf_threshold_method: unsupParams.mf_threshold_method,
          mf_percentile: unsupParams.mf_percentile,
          mf_min_component_size: unsupParams.mf_min_component_size,
          // Top-Hat
          th_line_length: unsupParams.th_line_length,
          th_n_orientations: unsupParams.th_n_orientations,
          th_threshold_method: unsupParams.th_threshold_method,
          th_percentile: unsupParams.th_percentile,
          th_min_component_size: unsupParams.th_min_component_size,
          th_min_aspect_ratio: unsupParams.th_min_aspect_ratio,
          // Attribute Filter
          af_bg_disk_radius: unsupParams.af_bg_disk_radius,
          af_threshold_method: unsupParams.af_threshold_method,
          af_adaptive_block: unsupParams.af_adaptive_block,
          af_min_area: unsupParams.af_min_area,
          af_min_eccentricity: unsupParams.af_min_eccentricity,
          af_min_axis_ratio: unsupParams.af_min_axis_ratio,
          af_max_circularity: unsupParams.af_max_circularity,
          af_min_skeleton_length: unsupParams.af_min_skeleton_length,
        }
      }
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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

  const handleEvaluate = async () => {
    if (detectionMode === 'synthetic' && !selectedModel) return
    setEvalLoading(true)
    setEvalError(null)
    try {
      const commonPostProc = {
        close_gap_size: closeGapSize, min_crack_area: minCrackArea,
        max_circularity: maxCircularity, circle_mask: circleMask,
        circle_mask_margin: circleMaskMargin, circle_mask_offset_x: circleMaskOffsetX,
        circle_mask_offset_y: circleMaskOffsetY,
        intensity_min: intensityMin, intensity_max: intensityMax,
      }
      let endpoint, body
      if (detectionMode === 'synthetic' && evalDataset !== 'labeling') {
        // Benchmark dataset evaluation
        endpoint = `${API}/evaluate/benchmark`
        body = {
          dataset: evalDataset, model: selectedModel, resolution, split, split_grid: splitGrid,
          confidence_threshold: confidenceThreshold,
          ...commonPostProc,
        }
      } else if (detectionMode === 'synthetic') {
        endpoint = `${API}/evaluate`
        body = {
          model: selectedModel, resolution, split, split_grid: splitGrid,
          confidence_threshold: confidenceThreshold,
          ...commonPostProc,
        }
      } else {
        endpoint = `${API}/evaluate/unsupervised`
        body = {
          method: unsupParams.method,
          ...commonPostProc,
          fs_filter: unsupParams.fs_filter, fs_sigma_min: unsupParams.fs_sigma_min, fs_sigma_max: unsupParams.fs_sigma_max,
          fs_threshold_method: unsupParams.fs_threshold_method, fs_percentile: unsupParams.fs_percentile,
          fs_min_component_size: unsupParams.fs_min_component_size,
          mf_n_orientations: unsupParams.mf_n_orientations, mf_sigma_x: unsupParams.mf_sigma_x,
          mf_sigma_y: unsupParams.mf_sigma_y, mf_kernel_size: unsupParams.mf_kernel_size,
          mf_threshold_method: unsupParams.mf_threshold_method, mf_percentile: unsupParams.mf_percentile,
          mf_min_component_size: unsupParams.mf_min_component_size,
          th_line_length: unsupParams.th_line_length, th_n_orientations: unsupParams.th_n_orientations,
          th_threshold_method: unsupParams.th_threshold_method, th_percentile: unsupParams.th_percentile,
          th_min_component_size: unsupParams.th_min_component_size, th_min_aspect_ratio: unsupParams.th_min_aspect_ratio,
          af_bg_disk_radius: unsupParams.af_bg_disk_radius, af_threshold_method: unsupParams.af_threshold_method,
          af_adaptive_block: unsupParams.af_adaptive_block, af_min_area: unsupParams.af_min_area,
          af_min_eccentricity: unsupParams.af_min_eccentricity, af_min_axis_ratio: unsupParams.af_min_axis_ratio,
          af_max_circularity: unsupParams.af_max_circularity, af_min_skeleton_length: unsupParams.af_min_skeleton_length,
        }
      }
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Evaluation failed')
      }
      const data = await res.json()
      // For benchmark mode, fix gtKind so modal defaults to 'raw_gt'
      if (evalDataset !== 'labeling') setEvalGtKind('raw_gt')
      setEvalResult(data)
      setShowEvalModal(true)
    } catch (err) {
      setEvalError(err.message)
    } finally {
      setEvalLoading(false)
    }
  }

  return (
    <div className="app-inner">
      {/* Status badges in a slim subheader */}
      {result && (
        <div className="subheader">
          <div className="header-badges">
            {result.stats.has_crack && <span className="badge crack-badge">Crack detected</span>}
            {result.stats.has_shape && <span className="badge shape-badge">Shape detected</span>}
            {!result.stats.has_crack && !result.stats.has_shape && (
              <span className="badge ok-badge">No defects</span>
            )}
          </div>
        </div>
      )}

      <div className="layout">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-scroll">
          <Section label="Detection Mode">
            <div className="chip-group">
              {[
                { k: 'synthetic',    l: 'Synthetic Model' },
                { k: 'unsupervised', l: 'Unsupervised' },
              ].map(({ k, l }) => (
                <button
                  key={k}
                  className={`chip ${detectionMode === k ? 'chip-active' : ''}`}
                  onClick={() => { setDetectionMode(k); setResult(null); hasResultRef.current = false }}
                  disabled={loading}
                >{l}</button>
              ))}
            </div>
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

          {detectionMode === 'synthetic' && (<>
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
          </>)}

          {detectionMode === 'unsupervised' && (<>
            <Section label="Method">
              <div className="chip-group">
                {[
                  { k: 'frangi_sato',    l: 'Frangi / Sato' },
                  { k: 'matched_filter', l: 'Matched Filter' },
                  { k: 'tophat',         l: 'Top-Hat' },
                  { k: 'attribute',      l: 'Attribute Filter' },
                ].map(({ k, l }) => (
                  <button
                    key={k}
                    className={`chip ${unsupParams.method === k ? 'chip-active' : ''}`}
                    onClick={() => setP('method', k)}
                    disabled={loading}
                  >{l}</button>
                ))}
              </div>
              <p className="hint" style={{ marginTop: 4 }}>
                {unsupParams.method === 'frangi_sato'    && 'Hessian-based multiscale ridge filter — enhances thin curvilinear structures.'}
                {unsupParams.method === 'matched_filter' && 'Oriented Gaussian kernels — high response where the image matches a line-shaped template.'}
                {unsupParams.method === 'tophat'         && 'Morphological white top-hat with line structuring elements at multiple orientations.'}
                {unsupParams.method === 'attribute'      && 'Threshold + keep only components with crack-like shape attributes (elongated, low circularity).'}
              </p>
            </Section>

            {/* ── Frangi / Sato ── */}
            {unsupParams.method === 'frangi_sato' && (<>
              <Section label="Filter Type">
                <div className="chip-group">
                  {[{k:'sato',l:'Sato'},{k:'frangi',l:'Frangi'}].map(({k,l}) => (
                    <button key={k} className={`chip ${unsupParams.fs_filter===k?'chip-active':''}`} onClick={()=>setP('fs_filter',k)} disabled={loading}>{l}</button>
                  ))}
                </div>
                <p className="hint">Sato is more robust to noise; Frangi emphasises vessel-like ridges.</p>
              </Section>
              <Section label={`Scale Range — σ ${unsupParams.fs_sigma_min}–${unsupParams.fs_sigma_max}`}>
                <div className="filter-row">
                  <div className="filter-row-header"><span className="filter-label">Min σ</span><span className="filter-value">{unsupParams.fs_sigma_min}</span></div>
                  <input type="range" min="1" max="6" step="1" value={unsupParams.fs_sigma_min}
                    onChange={(e)=>setP('fs_sigma_min', Math.min(+e.target.value, unsupParams.fs_sigma_max-1))}
                    disabled={loading} className="threshold-slider" />
                  <div className="filter-row-header" style={{marginTop:6}}><span className="filter-label">Max σ</span><span className="filter-value">{unsupParams.fs_sigma_max}</span></div>
                  <input type="range" min="2" max="10" step="1" value={unsupParams.fs_sigma_max}
                    onChange={(e)=>setP('fs_sigma_max', Math.max(+e.target.value, unsupParams.fs_sigma_min+1))}
                    disabled={loading} className="threshold-slider" />
                </div>
                <p className="hint">Probes structures at these pixel radii. Increase max σ for wider cracks.</p>
              </Section>
              <Section label="Threshold Method">
                <div className="chip-group">
                  {[{k:'percentile',l:'Percentile'},{k:'otsu',l:'Otsu'},{k:'adaptive',l:'Adaptive'}].map(({k,l})=>(
                    <button key={k} className={`chip ${unsupParams.fs_threshold_method===k?'chip-active':''}`} onClick={()=>setP('fs_threshold_method',k)} disabled={loading}>{l}</button>
                  ))}
                </div>
              </Section>
              {unsupParams.fs_threshold_method === 'percentile' && (
                <Section label={`Percentile — ${unsupParams.fs_percentile}`}>
                  <input type="range" min="70" max="99.5" step="0.5" value={unsupParams.fs_percentile}
                    onChange={(e)=>setP('fs_percentile',+e.target.value)} disabled={loading} className="threshold-slider" />
                  <div className="threshold-labels"><span>Sensitive (70)</span><span>Strict (99.5)</span></div>
                  <p className="hint">Keep top {(100-unsupParams.fs_percentile).toFixed(1)}% of the filter response.</p>
                </Section>
              )}
              <Section label={`Min Component Size — ${unsupParams.fs_min_component_size===0?'off':unsupParams.fs_min_component_size+' px'}`}>
                <input type="range" min="0" max="500" step="10" value={unsupParams.fs_min_component_size}
                  onChange={(e)=>setP('fs_min_component_size',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>Off</span><span>500 px</span></div>
              </Section>
            </>)}

            {/* ── Matched Filter ── */}
            {unsupParams.method === 'matched_filter' && (<>
              <Section label={`Orientations — ${unsupParams.mf_n_orientations}`}>
                <input type="range" min="4" max="24" step="2" value={unsupParams.mf_n_orientations}
                  onChange={(e)=>setP('mf_n_orientations',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>4</span><span>24</span></div>
                <p className="hint">More orientations = better coverage, slower inference.</p>
              </Section>
              <Section label={`σ_x (crack width) — ${unsupParams.mf_sigma_x}`}>
                <input type="range" min="0.5" max="5" step="0.5" value={unsupParams.mf_sigma_x}
                  onChange={(e)=>setP('mf_sigma_x',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>0.5</span><span>5</span></div>
              </Section>
              <Section label={`σ_y (crack length) — ${unsupParams.mf_sigma_y}`}>
                <input type="range" min="2" max="20" step="1" value={unsupParams.mf_sigma_y}
                  onChange={(e)=>setP('mf_sigma_y',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>2</span><span>20</span></div>
              </Section>
              <Section label={`Kernel Size — ${unsupParams.mf_kernel_size}`}>
                <input type="range" min="11" max="51" step="2" value={unsupParams.mf_kernel_size}
                  onChange={(e)=>setP('mf_kernel_size',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>11</span><span>51</span></div>
              </Section>
              <Section label="Threshold Method">
                <div className="chip-group">
                  {[{k:'percentile',l:'Percentile'},{k:'otsu',l:'Otsu'}].map(({k,l})=>(
                    <button key={k} className={`chip ${unsupParams.mf_threshold_method===k?'chip-active':''}`} onClick={()=>setP('mf_threshold_method',k)} disabled={loading}>{l}</button>
                  ))}
                </div>
              </Section>
              {unsupParams.mf_threshold_method === 'percentile' && (
                <Section label={`Percentile — ${unsupParams.mf_percentile}`}>
                  <input type="range" min="80" max="99.5" step="0.5" value={unsupParams.mf_percentile}
                    onChange={(e)=>setP('mf_percentile',+e.target.value)} disabled={loading} className="threshold-slider" />
                  <div className="threshold-labels"><span>Sensitive (80)</span><span>Strict (99.5)</span></div>
                  <p className="hint">Keep top {(100-unsupParams.mf_percentile).toFixed(1)}% of the filter response.</p>
                </Section>
              )}
              <Section label={`Min Component Size — ${unsupParams.mf_min_component_size===0?'off':unsupParams.mf_min_component_size+' px'}`}>
                <input type="range" min="0" max="500" step="10" value={unsupParams.mf_min_component_size}
                  onChange={(e)=>setP('mf_min_component_size',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>Off</span><span>500 px</span></div>
              </Section>
            </>)}

            {/* ── Top-Hat ── */}
            {unsupParams.method === 'tophat' && (<>
              <Section label={`Line Length — ${unsupParams.th_line_length} px`}>
                <input type="range" min="10" max="80" step="5" value={unsupParams.th_line_length}
                  onChange={(e)=>setP('th_line_length',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>10</span><span>80</span></div>
                <p className="hint">Length of the line structuring element. Increase for longer cracks.</p>
              </Section>
              <Section label={`Orientations — ${unsupParams.th_n_orientations}`}>
                <input type="range" min="4" max="36" step="2" value={unsupParams.th_n_orientations}
                  onChange={(e)=>setP('th_n_orientations',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>4</span><span>36</span></div>
              </Section>
              <Section label="Threshold Method">
                <div className="chip-group">
                  {[{k:'percentile',l:'Percentile'},{k:'otsu',l:'Otsu'}].map(({k,l})=>(
                    <button key={k} className={`chip ${unsupParams.th_threshold_method===k?'chip-active':''}`} onClick={()=>setP('th_threshold_method',k)} disabled={loading}>{l}</button>
                  ))}
                </div>
              </Section>
              {unsupParams.th_threshold_method === 'percentile' && (
                <Section label={`Percentile — ${unsupParams.th_percentile}`}>
                  <input type="range" min="80" max="99.5" step="0.5" value={unsupParams.th_percentile}
                    onChange={(e)=>setP('th_percentile',+e.target.value)} disabled={loading} className="threshold-slider" />
                  <div className="threshold-labels"><span>Sensitive (80)</span><span>Strict (99.5)</span></div>
                  <p className="hint">Keep top {(100-unsupParams.th_percentile).toFixed(1)}% of the filter response.</p>
                </Section>
              )}
              <Section label={`Min Component Size — ${unsupParams.th_min_component_size===0?'off':unsupParams.th_min_component_size+' px'}`}>
                <input type="range" min="0" max="500" step="10" value={unsupParams.th_min_component_size}
                  onChange={(e)=>setP('th_min_component_size',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>Off</span><span>500 px</span></div>
              </Section>
              <Section label={`Min Aspect Ratio — ${unsupParams.th_min_aspect_ratio}`}>
                <input type="range" min="1" max="10" step="0.5" value={unsupParams.th_min_aspect_ratio}
                  onChange={(e)=>setP('th_min_aspect_ratio',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>1 (any)</span><span>10 (elongated)</span></div>
                <p className="hint">Keep components with major/minor axis ratio ≥ this. Removes round blobs.</p>
              </Section>
            </>)}

            {/* ── Attribute Filter ── */}
            {unsupParams.method === 'attribute' && (<>
              <Section label={`BG Disk Radius — ${unsupParams.af_bg_disk_radius} px`}>
                <input type="range" min="5" max="50" step="5" value={unsupParams.af_bg_disk_radius}
                  onChange={(e)=>setP('af_bg_disk_radius',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>5</span><span>50</span></div>
                <p className="hint">Background subtraction disk radius — should be larger than crack width.</p>
              </Section>
              <Section label="Threshold Method">
                <div className="chip-group">
                  {[{k:'otsu',l:'Otsu'},{k:'adaptive',l:'Adaptive'}].map(({k,l})=>(
                    <button key={k} className={`chip ${unsupParams.af_threshold_method===k?'chip-active':''}`} onClick={()=>setP('af_threshold_method',k)} disabled={loading}>{l}</button>
                  ))}
                </div>
              </Section>
              <Section label={`Min Area — ${unsupParams.af_min_area} px`}>
                <input type="range" min="10" max="500" step="10" value={unsupParams.af_min_area}
                  onChange={(e)=>setP('af_min_area',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>10</span><span>500</span></div>
              </Section>
              <Section label={`Min Eccentricity — ${unsupParams.af_min_eccentricity.toFixed(2)}`}>
                <input type="range" min="0.5" max="0.99" step="0.01" value={unsupParams.af_min_eccentricity}
                  onChange={(e)=>setP('af_min_eccentricity',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>0.5</span><span>0.99</span></div>
                <p className="hint">0 = circle, 1 = line. Cracks should have high eccentricity.</p>
              </Section>
              <Section label={`Min Axis Ratio — ${unsupParams.af_min_axis_ratio}`}>
                <input type="range" min="1" max="10" step="0.5" value={unsupParams.af_min_axis_ratio}
                  onChange={(e)=>setP('af_min_axis_ratio',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>1</span><span>10</span></div>
              </Section>
              <Section label={`Max Circularity — ${unsupParams.af_max_circularity.toFixed(2)}`}>
                <input type="range" min="0.05" max="1" step="0.05" value={unsupParams.af_max_circularity}
                  onChange={(e)=>setP('af_max_circularity',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>Strict (0.05)</span><span>Off (1.0)</span></div>
                <p className="hint">4π·area/perimeter². Cracks score low; round blobs score near 1.</p>
              </Section>
              <Section label={`Min Skeleton Length — ${unsupParams.af_min_skeleton_length} px`}>
                <input type="range" min="5" max="100" step="5" value={unsupParams.af_min_skeleton_length}
                  onChange={(e)=>setP('af_min_skeleton_length',+e.target.value)} disabled={loading} className="threshold-slider" />
                <div className="threshold-labels"><span>5</span><span>100</span></div>
                <p className="hint">Minimum medial-axis length. Filters out short isolated blobs.</p>
              </Section>
            </>)}
          </>)}

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
          <div className="eval-btn-group">
            {detectionMode === 'synthetic' && (
              <select
                className="eval-dataset-select"
                value={evalDataset}
                onChange={(e) => setEvalDataset(e.target.value)}
                disabled={evalLoading}
                title="Select evaluation dataset"
              >
                <option value="labeling">My Labeled Set</option>
                {benchmarks.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            )}
            <button
              className="eval-btn"
              onClick={handleEvaluate}
              disabled={evalLoading || loading || (detectionMode === 'synthetic' && !selectedModel)}
            >
              {evalLoading ? (
                <>
                  <span className="btn-spinner" />
                  Evaluating…
                </>
              ) : (
                'Evaluate on Test Set'
              )}
            </button>
          </div>
          {evalError && <p className="eval-btn-error">{evalError}</p>}
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

      {/* Evaluation modal */}
      {showEvalModal && evalResult && (
        <EvalModal
          result={evalResult}
          predKind={evalPredKind}
          gtKind={evalGtKind}
          onPredKind={setEvalPredKind}
          onGtKind={setEvalGtKind}
          isBenchmark={!!evalResult.benchmark}
          onClose={() => setShowEvalModal(false)}
        />
      )}
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

const EVAL_METRICS = [
  { key: 'iou',         label: 'Jaccard' },
  { key: 'dice',        label: 'Dice' },
  { key: 'accuracy',    label: 'Acc.' },
  { key: 'recall',      label: 'Sn.' },
  { key: 'specificity', label: 'Sp.' },
  { key: 'auc',         label: 'AUC' },
]

function metricColor(v) {
  if (v == null) return 'var(--muted)'
  if (v >= 0.8)  return 'var(--green)'
  if (v >= 0.5)  return 'var(--gold)'
  return 'var(--red)'
}

function EvalModal({ result, predKind, gtKind, onPredKind, onGtKind, isBenchmark, onClose }) {
  // For benchmark mode, GT is always 'raw_gt'
  const effectiveGtKind = isBenchmark ? 'raw_gt' : gtKind
  const aggKey = `${predKind}_vs_${effectiveGtKind}`
  const agg    = result.aggregate[aggKey]

  const [selectedRow,    setSelectedRow]    = useState(null)
  const [compareData,    setCompareData]    = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)

  const handleRowClick = async (filename) => {
    if (selectedRow === filename) {
      setSelectedRow(null)
      setCompareData(null)
      return
    }
    setSelectedRow(filename)
    setCompareLoading(true)
    setCompareData(null)
    try {
      const res = await fetch(`${API}/evaluate/compare/${encodeURIComponent(filename)}`)
      if (res.ok) setCompareData(await res.json())
    } catch {}
    finally { setCompareLoading(false) }
  }

  const getRowMetrics = (row) => {
    const gtData = row[effectiveGtKind]
    return gtData ? gtData[predKind] : null
  }

  return (
    <div className="eval-backdrop" onClick={onClose}>
      <div className="eval-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="eval-modal-header">
          <div>
            <h2 className="eval-modal-title">
              {isBenchmark ? `Benchmark: ${result.benchmark}` : 'Test Set Evaluation'}
            </h2>
            <span className="eval-modal-sub">
              {result.n_evaluated} image{result.n_evaluated !== 1 ? 's' : ''} evaluated · crack class only
            </span>
          </div>
          <button className="eval-modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* Toggle controls */}
        <div className="eval-controls">
          <div className="eval-control-group">
            <span className="eval-control-label">Prediction</span>
            <div className="eval-pill-group">
              {[
                { k: 'filtered_pred', label: 'Filtered (current settings)' },
                { k: 'raw_pred',      label: 'Raw (no post-processing)' },
              ].map(({ k, label }) => (
                <button
                  key={k}
                  className={`eval-pill ${predKind === k ? 'eval-pill-active' : ''}`}
                  onClick={() => onPredKind(k)}
                >{label}</button>
              ))}
            </div>
          </div>
          {!isBenchmark && (
            <div className="eval-control-group">
              <span className="eval-control-label">Ground Truth</span>
              <div className="eval-pill-group">
                {[
                  { k: 'filtered_gt', label: 'Filtered Labels' },
                  { k: 'raw_gt',      label: 'Original Labels' },
                ].map(({ k, label }) => (
                  <button
                    key={k}
                    className={`eval-pill ${gtKind === k ? 'eval-pill-active' : ''}`}
                    onClick={() => onGtKind(k)}
                  >{label}</button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Aggregate metric cards */}
        {agg ? (
          <div className="eval-agg-row">
            {EVAL_METRICS.map(({ key, label }) => (
              <div key={key} className="eval-agg-card">
                <div className="eval-agg-value" style={{ color: metricColor(agg[key]) }}>
                  {(agg[key] * 100).toFixed(1)}%
                </div>
                <div className="eval-agg-label">{label}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="eval-no-data">No data available for this combination.</div>
        )}

        {/* Comparison panel — shown when a row is selected */}
        {(selectedRow) && (
          <div className="eval-compare">
            <div className="eval-compare-header">
              <span className="eval-compare-name">{selectedRow}</span>
              <button className="eval-compare-dismiss" onClick={() => { setSelectedRow(null); setCompareData(null) }}>✕</button>
            </div>
            {compareLoading ? (
              <div className="eval-compare-loading"><span className="btn-spinner" /></div>
            ) : compareData ? (
              <>
                <div className="eval-compare-single">
                  {compareData[`${predKind}_vs_${effectiveGtKind}`] ? (
                    <img
                      src={`data:image/png;base64,${compareData[`${predKind}_vs_${effectiveGtKind}`]}`}
                      alt="Comparison"
                      className="eval-compare-img-full"
                    />
                  ) : (
                    <div className="eval-compare-nogt">No data for this combination</div>
                  )}
                </div>
                <div className="eval-compare-legend">
                  <span className="eval-legend-item">
                    <span className="eval-legend-dot" style={{ background: '#00ff00' }} />
                    Correct (TP)
                  </span>
                  <span className="eval-legend-item">
                    <span className="eval-legend-dot" style={{ background: '#ff3333' }} />
                    False positive
                  </span>
                  <span className="eval-legend-item">
                    <span className="eval-legend-dot" style={{ background: '#00ffff' }} />
                    Missed (FN)
                  </span>
                </div>
              </>
            ) : null}
          </div>
        )}

        {/* Per-image table */}
        <div className="eval-table-wrap">
          <table className="eval-table">
            <thead>
              <tr>
                <th>Image</th>
                {EVAL_METRICS.map(({ key, label }) => (
                  <th key={key}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.results.map((row) => {
                const m = getRowMetrics(row)
                const isSelected = selectedRow === row.image
                return (
                  <tr
                    key={row.image}
                    className={`eval-table-row${isSelected ? ' eval-row-selected' : ''}`}
                    onClick={() => handleRowClick(row.image)}
                  >
                    <td className="eval-table-filename">{row.image}</td>
                    {EVAL_METRICS.map(({ key }) => (
                      <td key={key} style={{ color: m ? metricColor(m[key]) : 'var(--muted)' }}>
                        {m ? `${(m[key] * 100).toFixed(1)}%` : '—'}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
