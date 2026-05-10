import { useState, useEffect, useRef, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL ?? '/api'

// Label classes — value is the grayscale pixel value stored in the mask
const CLASSES = [
  { id: 'crack', label: 'Crack', color: '#ff4d4d', maskValue: 255 },
]

const CURSOR_SIZE_SCALE = 1  // canvas-px to screen-px ratio tracking

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert mask grayscale value → RGBA for the overlay canvas */
function maskValueToRGBA(v) {
  if (v === 255) return [255, 77, 77, 180]  // crack — red
  return [0, 0, 0, 0]                        // background — transparent
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Labeler() {
  const [images, setImages]               = useState([])
  const [selectedImage, setSelectedImage] = useState(null)
  const [imageData, setImageData]         = useState(null)  // { image, width, height }
  const [activeClass, setActiveClass]     = useState('crack')
  const [brushSize, setBrushSize]         = useState(5)
  const [tool, setTool]                   = useState('brush')  // 'brush' | 'eraser' | 'pan'
  const toolRef = useRef('brush')          // always current — avoids stale closures in pointer handlers
  const [saving, setSaving]               = useState(false)
  const [saveStatus, setSaveStatus]       = useState(null)  // 'saved' | 'error'
  const [loadingImage, setLoadingImage]   = useState(false)
  const [maskProgress, setMaskProgress]   = useState({})   // filename → boolean (has mask)
  const [showMask, setShowMask]           = useState(true)
  const showMaskRef = useRef(true)

  // Noise filter — non-destructive: hides small regions visually, mask data always preserved
  const [filtersOn, setFiltersOn]         = useState(false)
  const [noiseMinArea, setNoiseMinArea]   = useState(100)
  const noiseFilterRef   = useRef(false)
  const noiseMinAreaRef  = useRef(100)
  const noiseHideRef     = useRef(null)   // Uint8Array — 1 = pixel hidden by noise filter
  const noiseDebounceRef = useRef(null)
  // Fill holes
  const [fillHoles, setFillHoles]         = useState(false)
  const fillHolesRef = useRef(false)
  const holeFillRef  = useRef(null)       // Uint8Array — class value for enclosed hole pixels

  // Intensity filter — non-destructive: hides labels visually, mask data is always preserved
  const [intensityMin, setIntensityMin]       = useState(0)
  const [intensityMax, setIntensityMax]       = useState(255)
  // Refs so paint / redrawOverlay always read the latest values without stale closures
  const intensityFilterRef = useRef(false)
  const intensityMinRef    = useRef(0)
  const intensityMaxRef    = useRef(255)

  // Canvas refs
  const baseCanvasRef = useRef(null)   // shows the original image (read-only)
  const maskCanvasRef = useRef(null)   // the actual label mask (grayscale, hidden)
  const drawCanvasRef = useRef(null)   // coloured overlay drawn on top for display

  // Interaction state
  const isDrawing  = useRef(false)
  const lastPos    = useRef(null)
  const undoStack  = useRef([])        // array of ImageData snapshots of maskCanvas

  // Keep toolRef, intensity refs, and noise refs in sync with state
  useEffect(() => { toolRef.current = tool }, [tool])
  // Both filters share the same on/off toggle (filtersOn)
  useEffect(() => {
    intensityFilterRef.current = filtersOn
    noiseFilterRef.current     = filtersOn
    fillHolesRef.current       = filtersOn && fillHoles
  }, [filtersOn, fillHoles])
  useEffect(() => { intensityMinRef.current = intensityMin }, [intensityMin])
  useEffect(() => { intensityMaxRef.current = intensityMax }, [intensityMax])
  useEffect(() => { noiseMinAreaRef.current = noiseMinArea }, [noiseMinArea])
  useEffect(() => { fillHolesRef.current = filtersOn && fillHoles }, [filtersOn, fillHoles])
  useEffect(() => { showMaskRef.current = showMask }, [showMask])

  // ── Compute noise-hide bitmap using client-side BFS ──────────────────────
  // Runs BFS only on pixels that are BOTH labeled AND pass the intensity filter,
  // so fragments left by intensity trimming are correctly detected as small.
  const computeNoiseHide = useCallback(() => {
    const mask = maskCanvasRef.current
    const base = baseCanvasRef.current
    if (!mask) return
    const w = mask.width
    const h = mask.height
    const n = w * h
    const hide = new Uint8Array(n)
    if (noiseFilterRef.current && noiseMinAreaRef.current > 0) {
      const maskPx = mask.getContext('2d').getImageData(0, 0, w, h).data
      // Only read base pixels when the intensity filter is actually on
      const basePx = (intensityFilterRef.current && base)
        ? base.getContext('2d').getImageData(0, 0, w, h).data
        : null
      const iMin   = intensityMinRef.current
      const iMax   = intensityMaxRef.current
      // A pixel "counts" for BFS only if it has a label AND passes intensity
      const isActive = (i) => {
        const v = maskPx[i * 4]
        if (v === 0) return false
        if (basePx) {
          const br = basePx[i * 4]
          if (br < iMin || br > iMax) return false
        }
        return true
      }
      const visited = new Uint8Array(n)
      const queue   = new Int32Array(n)
      const minArea = noiseMinAreaRef.current
      for (let start = 0; start < n; start++) {
        if (!isActive(start) || visited[start]) continue
        const classVal = maskPx[start * 4]
        let head = 0, tail = 0
        queue[tail++] = start
        visited[start] = 1
        while (head < tail) {
          const idx = queue[head++]
          const x = idx % w
          const y = (idx / w) | 0
          for (let dy = -1; dy <= 1; dy++) {
            const ny = y + dy
            if (ny < 0 || ny >= h) continue
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue
              const nx = x + dx
              if (nx < 0 || nx >= w) continue
              const ni = ny * w + nx
              if (visited[ni] || maskPx[ni * 4] !== classVal || !isActive(ni)) continue
              visited[ni] = 1
              queue[tail++] = ni
            }
          }
        }
        if (tail < minArea) {
          for (let j = 0; j < tail; j++) hide[queue[j]] = 1
        }
      }
    }
    noiseHideRef.current = hide
  }, [])  // stable — reads only from refs and canvas refs

  // ── Compute hole-fill bitmap ─────────────────────────────────────────────
  // Finds background pixels fully enclosed by visible labeled regions and
  // assigns them the class of the nearest surrounding labeled pixel.
  // Must run AFTER computeNoiseHide (reads noiseHideRef).
  const computeHoleFill = useCallback(() => {
    const mask = maskCanvasRef.current
    const base = baseCanvasRef.current
    if (!mask) return
    const w = mask.width
    const h = mask.height
    const n = w * h
    const holeFill = new Uint8Array(n)

    if (fillHolesRef.current) {
      const maskPx    = mask.getContext('2d').getImageData(0, 0, w, h).data
      const basePx    = (intensityFilterRef.current && base)
        ? base.getContext('2d').getImageData(0, 0, w, h).data
        : null
      const iMin      = intensityMinRef.current
      const iMax      = intensityMaxRef.current
      const noiseHide = noiseHideRef.current

      // Build visible-foreground bitmap: labeled pixels that pass all active filters.
      // Pixels hidden by the intensity filter are treated as voids/holes — this is
      // intentional so that intensity-created pores inside brush strokes get filled.
      const fg = new Uint8Array(n)
      for (let i = 0; i < n; i++) {
        const v = maskPx[i * 4]
        if (v === 0) continue
        if (basePx && (basePx[i * 4] < iMin || basePx[i * 4] > iMax)) continue
        if (noiseHide && noiseHide[i]) continue
        fg[i] = v
      }

      // Step 1: flood-fill from every border pixel to find "outside" voids.
      // Traverses fg===0 pixels (both unlabeled background and intensity-filtered labels).
      const outside = new Uint8Array(n)
      const bfsQ    = new Int32Array(n)
      let head = 0, tail = 0
      const seedOutside = (i) => { if (!outside[i] && fg[i] === 0) { outside[i] = 1; bfsQ[tail++] = i } }
      for (let x = 0; x < w; x++) { seedOutside(x); seedOutside((h - 1) * w + x) }
      for (let y = 1; y < h - 1; y++) { seedOutside(y * w); seedOutside(y * w + w - 1) }
      while (head < tail) {
        const idx = bfsQ[head++]
        const x = idx % w, y = (idx / w) | 0
        if (x > 0     && !outside[idx - 1] && fg[idx - 1] === 0) { outside[idx - 1] = 1; bfsQ[tail++] = idx - 1 }
        if (x < w - 1 && !outside[idx + 1] && fg[idx + 1] === 0) { outside[idx + 1] = 1; bfsQ[tail++] = idx + 1 }
        if (y > 0     && !outside[idx - w] && fg[idx - w] === 0) { outside[idx - w] = 1; bfsQ[tail++] = idx - w }
        if (y < h - 1 && !outside[idx + w] && fg[idx + w] === 0) { outside[idx + w] = 1; bfsQ[tail++] = idx + w }
      }

      // Step 2: BFS from visible labeled pixels into enclosed voids.
      // holeFill captures both unlabeled background holes AND intensity-filtered pores.
      const assigned = new Uint8Array(n)
      const expQ     = new Int32Array(n)
      head = 0; tail = 0
      for (let i = 0; i < n; i++) { if (fg[i] > 0) { assigned[i] = fg[i]; expQ[tail++] = i } }
      while (head < tail) {
        const idx = expQ[head++]
        const cv  = assigned[idx]
        const x = idx % w, y = (idx / w) | 0
        const expand = (ni) => {
          if (fg[ni] !== 0 || outside[ni] || assigned[ni]) return
          assigned[ni] = cv; holeFill[ni] = cv; expQ[tail++] = ni
        }
        if (x > 0)     expand(idx - 1)
        if (x < w - 1) expand(idx + 1)
        if (y > 0)     expand(idx - w)
        if (y < h - 1) expand(idx + w)
      }
    }
    holeFillRef.current = holeFill
  }, [])  // stable — reads only from refs and canvas refs

  // ── Run both spatial filters in the correct order ────────────────────────
  // Hole fill reads noiseHideRef, so noise must run first.
  const computeFilters = useCallback(() => {
    computeNoiseHide()
    computeHoleFill()
  }, [computeNoiseHide, computeHoleFill])

  // ── Load image list ──────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/labeling/images`)
      .then((r) => r.json())
      .then((data) => {
        setImages(data.images)
        if (data.images.length > 0) loadImage(data.images[0])
      })
      .catch(() => {})
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load a single image + its existing mask ──────────────────────────────
  const loadImage = useCallback(async (filename) => {
    setLoadingImage(true)
    setSaveStatus(null)
    undoStack.current = []

    try {
      const [imgRes, maskRes] = await Promise.all([
        fetch(`${API}/labeling/image/${encodeURIComponent(filename)}`).then((r) => r.json()),
        fetch(`${API}/labeling/mask/${encodeURIComponent(filename)}`).then((r) => r.json()),
      ])

      setSelectedImage(filename)
      setImageData({ image: imgRes.image, width: imgRes.width, height: imgRes.height })

      // Paint base canvas once dimensions are known — deferred to next tick so
      // refs have been re-attached after a potential re-render.
      requestAnimationFrame(() => {
        const base  = baseCanvasRef.current
        const mask  = maskCanvasRef.current
        const draw  = drawCanvasRef.current
        if (!base || !mask || !draw) return

        base.width = mask.width = draw.width  = imgRes.width
        base.height = mask.height = draw.height = imgRes.height

        // Draw original image on base canvas
        const img = new Image()
        img.onload = () => base.getContext('2d').drawImage(img, 0, 0)
        img.src = `data:image/png;base64,${imgRes.image}`

        // Restore or clear mask canvas
        const maskCtx = mask.getContext('2d')
        maskCtx.clearRect(0, 0, mask.width, mask.height)

        if (maskRes.mask) {
          const maskImg = new Image()
          maskImg.onload = () => {
            maskCtx.drawImage(maskImg, 0, 0)
            computeFilters()
            redrawOverlay()
          }
          maskImg.src = `data:image/png;base64,${maskRes.mask}`
          setMaskProgress((p) => ({ ...p, [filename]: true }))
        } else {
          computeFilters()
          redrawOverlay()
          setMaskProgress((p) => ({ ...p, [filename]: false }))
        }
      })
    } finally {
      setLoadingImage(false)
    }
  }, [])

  // ── Redraw overlay from the mask canvas, applying the intensity filter visually ──
  // Uses refs so it never needs to be recreated — safe to call from any callback.
  const redrawOverlay = useCallback(() => {
    const base = baseCanvasRef.current
    const mask = maskCanvasRef.current
    const draw = drawCanvasRef.current
    if (!base || !mask || !draw) return

    if (!showMaskRef.current) {
      draw.getContext('2d').clearRect(0, 0, draw.width, draw.height)
      return
    }

    const w = mask.width
    const h = mask.height
    const basePx  = base.getContext('2d').getImageData(0, 0, w, h).data
    const maskPx  = mask.getContext('2d').getImageData(0, 0, w, h).data
    const drawCtx = draw.getContext('2d')
    const out     = drawCtx.createImageData(w, h)

    const filterOn  = intensityFilterRef.current
    const iMin      = intensityMinRef.current
    const iMax      = intensityMaxRef.current
    const noiseOn   = noiseFilterRef.current
    const noiseHide = noiseHideRef.current
    const fillOn    = fillHolesRef.current
    const holeFill  = holeFillRef.current

    for (let i = 0; i < w * h; i++) {
      const p = i * 4
      const v = maskPx[p]  // grayscale mask value
      if (v === 0) {
        // Render enclosed holes filled with the surrounding class colour
        if (fillOn && holeFill && holeFill[i] > 0) {
          const [r, g, b, a] = maskValueToRGBA(holeFill[i])
          out.data[p] = r; out.data[p + 1] = g; out.data[p + 2] = b; out.data[p + 3] = a
        }
        continue
      }
      // Hide pixel if intensity filter active and brightness out of range.
      // Even when hidden, if fill-holes is on and this pixel is an enclosed pore,
      // render it with the fill colour (intensity-created pores inside brush strokes).
      if (filterOn) {
        const brightness = basePx[p]
        if (brightness < iMin || brightness > iMax) {
          if (fillOn && holeFill && holeFill[i] > 0) {
            const [r, g, b, a] = maskValueToRGBA(holeFill[i])
            out.data[p] = r; out.data[p + 1] = g; out.data[p + 2] = b; out.data[p + 3] = a
          }
          continue
        }
      }
      // Hide pixel if noise filter active and it's in a small component
      if (noiseOn && noiseHide && noiseHide[i]) continue
      const [r, g, b, a] = maskValueToRGBA(v)
      out.data[p] = r; out.data[p + 1] = g; out.data[p + 2] = b; out.data[p + 3] = a
    }
    drawCtx.putImageData(out, 0, 0)
  }, [])  // stable — reads only from refs and canvas refs

  // ── Canvas coordinate helpers ─────────────────────────────────────────────
  const getCanvasPos = (e, canvas) => {
    const rect  = canvas.getBoundingClientRect()
    const scaleX = canvas.width  / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    }
  }

  // ── Painting logic ────────────────────────────────────────────────────────
  // Writes every stroke unconditionally to the mask (source of truth).
  // The overlay display is handled by redrawOverlay, which respects the filter.
  const paint = useCallback((pos) => {
    const mask = maskCanvasRef.current
    if (!mask) return

    const value = toolRef.current === 'eraser' ? 0 : CLASSES.find((c) => c.id === activeClass).maskValue
    const maskCtx = mask.getContext('2d')
    maskCtx.fillStyle   = `rgb(${value},${value},${value})`
    maskCtx.strokeStyle = `rgb(${value},${value},${value})`
    maskCtx.lineWidth   = brushSize * 2
    maskCtx.lineCap     = 'round'
    maskCtx.lineJoin    = 'round'
    if (lastPos.current) {
      maskCtx.beginPath()
      maskCtx.moveTo(lastPos.current.x, lastPos.current.y)
      maskCtx.lineTo(pos.x, pos.y)
      maskCtx.stroke()
    } else {
      maskCtx.beginPath()
      maskCtx.arc(pos.x, pos.y, brushSize, 0, Math.PI * 2)
      maskCtx.fill()
    }

    redrawOverlay()
    lastPos.current = pos
  }, [activeClass, brushSize, redrawOverlay])

  // ── Pointer event handlers ────────────────────────────────────────────────
  const onPointerDown = useCallback((e) => {
    if (toolRef.current === 'pan') return
    e.preventDefault()
    const canvas = drawCanvasRef.current
    if (!canvas) return
    // Save undo snapshot before first stroke
    const maskCtx = maskCanvasRef.current.getContext('2d')
    const snap    = maskCtx.getImageData(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height)
    undoStack.current.push(snap)
    if (undoStack.current.length > 40) undoStack.current.shift()

    isDrawing.current = true
    lastPos.current   = null
    paint(getCanvasPos(e, canvas))
  }, [paint])

  const onPointerMove = useCallback((e) => {
    if (toolRef.current === 'pan') return
    e.preventDefault()
    if (!isDrawing.current) return
    paint(getCanvasPos(e, drawCanvasRef.current))
  }, [paint])

  const onPointerUp = useCallback(() => {
    isDrawing.current = false
    lastPos.current   = null
    // Recompute filters after the stroke modified the mask
    if (noiseFilterRef.current || fillHolesRef.current) {
      computeFilters()
      redrawOverlay()
    }
  }, [computeFilters, redrawOverlay])

  // ── Undo ─────────────────────────────────────────────────────────────────
  const undo = useCallback(() => {
    if (undoStack.current.length === 0) return
    maskCanvasRef.current.getContext('2d').putImageData(undoStack.current.pop(), 0, 0)
    computeFilters()
    redrawOverlay()
  }, [computeFilters, redrawOverlay])

  // ── Recompute filters + redraw for any filter param change ─────────────────
  useEffect(() => {
    if (!selectedImage) return
    clearTimeout(noiseDebounceRef.current)
    noiseDebounceRef.current = setTimeout(() => {
      computeFilters()
      redrawOverlay()
    }, 80)
    return () => clearTimeout(noiseDebounceRef.current)
  }, [filtersOn, fillHoles, intensityMin, intensityMax, noiseMinArea, selectedImage, computeFilters, redrawOverlay])

  // ── Clear ─────────────────────────────────────────────────────────────────
  const clearMask = useCallback(() => {
    const mask = maskCanvasRef.current
    const draw = drawCanvasRef.current
    if (!mask || !draw) return
    undoStack.current.push(mask.getContext('2d').getImageData(0, 0, mask.width, mask.height))
    mask.getContext('2d').clearRect(0, 0, mask.width, mask.height)
    draw.getContext('2d').clearRect(0, 0, draw.width, draw.height)
    computeFilters()
  }, [computeFilters])

  // ── Save ─────────────────────────────────────────────────────────────────
  const saveMask = useCallback(async () => {
    if (!selectedImage) return
    setSaving(true)
    setSaveStatus(null)
    try {
      const mask = maskCanvasRef.current
      const w = mask.width
      const h = mask.height

      // Raw mask — always the unfiltered source of truth (used for reloading)
      const rawB64 = mask.toDataURL('image/png').split(',')[1]

      // Filtered mask — apply intensity / noise / hole-fill on a copy
      const intensityActive = intensityFilterRef.current
      const noiseActive     = noiseFilterRef.current
      const fillActive      = fillHolesRef.current
      const holeFill        = holeFillRef.current
      const copy   = mask.getContext('2d').getImageData(0, 0, w, h)
      const px     = copy.data
      const basePx = intensityActive
        ? baseCanvasRef.current.getContext('2d').getImageData(0, 0, w, h).data
        : null
      const iMin      = intensityMinRef.current
      const iMax      = intensityMaxRef.current
      const noiseHide = noiseHideRef.current
      for (let i = 0; i < w * h; i++) {
        const p = i * 4
        if (px[p] === 0) {
          if (fillActive && holeFill && holeFill[i] > 0) {
            px[p] = px[p + 1] = px[p + 2] = holeFill[i]
            px[p + 3] = 255
          }
          continue
        }
        const hiddenByIntensity = intensityActive && basePx && (basePx[p] < iMin || basePx[p] > iMax)
        const hiddenByNoise     = noiseActive && noiseHide && noiseHide[i]
        if (hiddenByIntensity || hiddenByNoise) {
          px[p] = px[p + 1] = px[p + 2] = px[p + 3] = 0
        }
      }
      const tmp = document.createElement('canvas')
      tmp.width = w; tmp.height = h
      tmp.getContext('2d').putImageData(copy, 0, 0)
      const filteredB64 = tmp.toDataURL('image/png').split(',')[1]

      const res = await fetch(`${API}/labeling/save`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ filename: selectedImage, mask: rawB64, mask_filtered: filteredB64 }),
      })
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Save failed')
      setSaveStatus('saved')
      setMaskProgress((p) => ({ ...p, [selectedImage]: true }))
    } catch (err) {
      setSaveStatus('error')
      console.error(err)
    } finally {
      setSaving(false)
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }, [selectedImage])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); undo() }
      if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); saveMask() }
      if (e.key === 'b') setTool('brush')
      if (e.key === 'e') setTool('eraser')
      if (e.key === 'p') setTool('pan')
      if (e.key === 'v') setShowMask((v) => { const next = !v; showMaskRef.current = next; redrawOverlay(); return next })
      if (e.key === '1') setActiveClass('crack')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undo, saveMask])

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="labeler-page">
      {/* ── Toolbar ── */}
      <aside className="labeler-sidebar">
        <div className="labeler-sidebar-scroll">

          {/* Image list */}
          <div className="lab-section">
            <p className="lab-section-label">Images ({images.length})</p>
            <div className="lab-image-list">
              {images.length === 0 && (
                <p className="lab-empty">No images in Labeling/Images</p>
              )}
              {images.map((name) => (
                <button
                  key={name}
                  className={`lab-image-item ${selectedImage === name ? 'lab-image-item-active' : ''}`}
                  onClick={() => loadImage(name)}
                  disabled={loadingImage}
                  title={name}
                >
                  <span className="lab-image-name">{name}</span>
                  {maskProgress[name] && <span className="lab-mask-dot" title="Mask saved" />}
                </button>
              ))}
            </div>
          </div>

          {/* Class selector */}
          <div className="lab-section">
            <p className="lab-section-label">Label Class</p>
            <div className="lab-class-list">
              {CLASSES.map((c, i) => (
                <button
                  key={c.id}
                  className={`lab-class-btn ${activeClass === c.id && tool === 'brush' ? 'lab-class-btn-active' : ''}`}
                  style={{ '--class-color': c.color }}
                  onClick={() => { setActiveClass(c.id); setTool('brush') }}
                  title={`Shortcut: ${i + 1}`}
                >
                  <span className="lab-class-dot" style={{ background: c.color }} />
                  {c.label}
                  <span className="lab-shortcut">{i + 1}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Tool */}
          <div className="lab-section">
            <p className="lab-section-label">Tool</p>
            <div className="lab-tool-row">
              <button
                className={`lab-tool-btn ${tool === 'brush' ? 'lab-tool-btn-active' : ''}`}
                onClick={() => setTool('brush')}
                title="Brush (B)"
              >
                Brush <span className="lab-shortcut">B</span>
              </button>
              <button
                className={`lab-tool-btn ${tool === 'eraser' ? 'lab-tool-btn-active' : ''}`}
                onClick={() => setTool('eraser')}
                title="Eraser (E)"
              >
                Eraser <span className="lab-shortcut">E</span>
              </button>
              <button
                className={`lab-tool-btn ${tool === 'pan' ? 'lab-tool-btn-active' : ''}`}
                onClick={() => setTool('pan')}
                title="Pan / Scroll (P)"
              >
                Pan <span className="lab-shortcut">P</span>
              </button>
            </div>
            {tool === 'pan' && (
              <p className="hint">Drag to scroll the canvas. Tap B or E to resume drawing.</p>
            )}
          </div>

          {/* Brush size */}
          <div className="lab-section">
            <p className="lab-section-label">Brush Size — {brushSize}px</p>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="threshold-slider"
            />
            <div className="threshold-labels">
              <span>1 px</span>
              <span>10 px</span>
            </div>
          </div>

          {/* Actions */}
          <div className="lab-section">
            <p className="lab-section-label">Actions</p>
            <div className="lab-action-row">
              <button className="lab-action-btn" onClick={undo} title="Undo (⌘Z)">
                ↩ Undo
              </button>
              <button
                className={`lab-action-btn ${!showMask ? 'lab-action-btn-active' : ''}`}
                onClick={() => setShowMask((v) => { const next = !v; showMaskRef.current = next; redrawOverlay(); return next })}
                title="Toggle mask visibility (V)"
              >
                {showMask ? '◑ Hide Mask' : '◑ Show Mask'}
              </button>
              <button className="lab-action-btn lab-action-btn-danger" onClick={clearMask} title="Clear mask">
                ✕ Clear
              </button>
            </div>
          </div>

          {/* Post-processing Filters */}
          <div className="lab-section">
            <div className="toggle-row">
              <p className="lab-section-label" style={{ marginBottom: 0 }}>Post-processing Filters</p>
              <button
                className={`toggle ${filtersOn ? 'toggle-on' : ''}`}
                onClick={() => setFiltersOn((v) => !v)}
                aria-pressed={filtersOn}
              >
                <span className="toggle-thumb" />
              </button>
            </div>
            {filtersOn && (
              <>
                <p className="hint">Filters are non-destructive — mask data is always kept. Save exports only visible labels.</p>

                {/* Intensity filter */}
                <div className="filter-row-header" style={{ marginTop: 8 }}>
                  <span className="filter-label">Intensity range</span>
                  <span className="filter-value">{intensityMin}–{intensityMax}</span>
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

                {/* Noise filter */}
                <div className="filter-row-header" style={{ marginTop: 12 }}>
                  <span className="filter-label">Min region area</span>
                  <span className="filter-value">{noiseMinArea} px²</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={noiseMinArea}
                  onChange={(e) => setNoiseMinArea(+e.target.value)}
                  className="threshold-slider"
                />
                <div className="threshold-labels">
                  <span>0</span>
                  <span>100 px²</span>
                </div>

                {/* Fill holes */}
                <div className="toggle-row" style={{ marginTop: 12 }}>
                  <span className="filter-label">Fill enclosed pores</span>
                  <button
                    className={`toggle ${fillHoles ? 'toggle-on' : ''}`}
                    onClick={() => setFillHoles((v) => !v)}
                    aria-pressed={fillHoles}
                  >
                    <span className="toggle-thumb" />
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Hint */}
          <p className="hint" style={{ padding: '0 4px' }}>
            Shortcuts: <strong>1</strong> crack · <strong>B</strong> brush ·
            <strong> E</strong> eraser · <strong>P</strong> pan · <strong>V</strong> mask · <strong>⌘Z</strong> undo · <strong>⌘S</strong> save
          </p>
        </div>

        {/* Save button */}
        <div className="predict-btn-wrap">
          <button
            className={`predict-btn ${saveStatus === 'saved' ? 'predict-btn-success' : saveStatus === 'error' ? 'predict-btn-error' : ''}`}
            onClick={saveMask}
            disabled={saving || !selectedImage}
          >
            {saving ? (
              <><span className="btn-spinner" /> Saving…</>
            ) : saveStatus === 'saved' ? (
              '✓ Saved'
            ) : saveStatus === 'error' ? (
              '✕ Error'
            ) : (
              'Save Mask'
            )}
          </button>
          <button
            className="predict-btn"
            style={{ marginTop: 8, background: 'var(--accent2, #4d7fff)' }}
            onClick={() => { window.location.href = `${API}/labeling/download-zip` }}
            title="Download all images and masks as a zip"
          >
            ⬇ Download All (zip)
          </button>
        </div>
      </aside>

      {/* ── Canvas area ── */}
      <main className="labeler-main">
        {!selectedImage && !loadingImage && (
          <div className="placeholder">
            <div className="placeholder-icon">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M3 9h18M9 21V9" />
              </svg>
            </div>
            <p>Add images to <strong>Labeling/Images</strong> and they will appear in the list.</p>
          </div>
        )}

        {loadingImage && (
          <div className="loading-screen">
            <div className="loading-spinner" />
            <p>Loading image…</p>
          </div>
        )}

        {selectedImage && !loadingImage && (
          <div className="labeler-canvas-wrap">
            <div className="labeler-canvas-stack">
              {/* Layer 0 — original image */}
              <canvas ref={baseCanvasRef} className="labeler-canvas labeler-canvas-base" />
              {/* Layer 1 — coloured label overlay (user interaction target) */}
              <canvas
                ref={drawCanvasRef}
                className="labeler-canvas labeler-canvas-draw"
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerLeave={onPointerUp}
                style={{ cursor: tool === 'pan' ? 'grab' : tool === 'eraser' ? 'cell' : 'crosshair', touchAction: tool === 'pan' ? 'auto' : 'none' }}
              />
              {/* Layer 2 — hidden mask (grayscale, used for export) */}
              <canvas ref={maskCanvasRef} className="labeler-canvas labeler-canvas-mask" />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
