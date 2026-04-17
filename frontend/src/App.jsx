import { useRef, useState, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// Color scheme per bin type
const BIN_META = {
  battery:    { label: 'HAZARDOUS',     color: 'text-red-400',    bar: 'bg-red-400',    badge: 'bg-red-500/20 border-red-500/40' },
  biological: { label: 'COMPOST',       color: 'text-amber-400',  bar: 'bg-amber-400',  badge: 'bg-amber-500/20 border-amber-500/40' },
  cardboard:  { label: 'RECYCLING',     color: 'text-green-400',  bar: 'bg-green-400',  badge: 'bg-green-500/20 border-green-500/40' },
  glass:      { label: 'RECYCLING',     color: 'text-green-400',  bar: 'bg-green-400',  badge: 'bg-green-500/20 border-green-500/40' },
  metal:      { label: 'RECYCLING',     color: 'text-green-400',  bar: 'bg-green-400',  badge: 'bg-green-500/20 border-green-500/40' },
  paper:      { label: 'RECYCLING',     color: 'text-green-400',  bar: 'bg-green-400',  badge: 'bg-green-500/20 border-green-500/40' },
  plastic:    { label: 'RECYCLING',     color: 'text-green-400',  bar: 'bg-green-400',  badge: 'bg-green-500/20 border-green-500/40' },
  textiles:   { label: 'DONATE',        color: 'text-purple-400', bar: 'bg-purple-400', badge: 'bg-purple-500/20 border-purple-500/40' },
  trash:      { label: 'GENERAL WASTE', color: 'text-gray-400',   bar: 'bg-gray-400',   badge: 'bg-gray-500/20 border-gray-500/40' },
}

export default function App() {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const historyRef = useRef([])   // last 5 predicted class names for smoothing

  const [result,   setResult]   = useState(null)
  const [isReady,  setIsReady]  = useState(false)
  const [camError, setCamError] = useState(false)

  // ── Start camera ────────────────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } } })
      .then(stream => {
        videoRef.current.srcObject = stream
        setIsReady(true)
      })
      .catch(() => setCamError(true))

    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
    }
  }, [])

  // ── Inference loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isReady) return

    const interval = setInterval(async () => {
      const video  = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) return

      // Capture current frame at 224×224 (matches model input size)
      canvas.width  = 224
      canvas.height = 224
      canvas.getContext('2d').drawImage(video, 0, 0, 224, 224)

      canvas.toBlob(async (blob) => {
        if (!blob) return
        const form = new FormData()
        form.append('file', blob, 'frame.jpg')

        try {
          const res  = await fetch(`${API_URL}/predict`, { method: 'POST', body: form })
          const data = await res.json()

          // Temporal smoothing — only update display if the same class
          // appears 3 or more times in the last 5 frames. Prevents flickering.
          const h = historyRef.current
          h.push(data.class)
          if (h.length > 5) h.shift()

          const counts = {}
          h.forEach(c => { counts[c] = (counts[c] || 0) + 1 })
          const dominant = Object.entries(counts).find(([, n]) => n >= 3)
          if (dominant) setResult(data)
        } catch {
          // API unreachable — skip frame silently
        }
      }, 'image/jpeg', 0.85)
    }, 600)

    return () => clearInterval(interval)
  }, [isReady])

  const meta = result ? BIN_META[result.class] : null

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="relative h-full w-full bg-black overflow-hidden">

      {/* Live camera feed */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Hidden canvas for frame capture only */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Vignette — keeps text readable over any camera background */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/80 pointer-events-none" />

      {/* ── Header ── */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2">
          <span className="text-2xl">♻️</span>
          <span className="text-white font-bold text-xl tracking-wide">RecycleSmart</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-green-400 text-xs font-medium tracking-widest uppercase">Live</span>
        </div>
      </div>

      {/* ── Scanning reticle ── */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="relative w-56 h-56">
          <div className="absolute inset-0 rounded-2xl border border-green-400/20" />
          <div className="absolute inset-0 rounded-2xl border border-green-400/50" />
          {/* Corner accents */}
          <div className="absolute top-0 left-0 w-7 h-7 border-t-2 border-l-2 border-green-400 rounded-tl-xl" />
          <div className="absolute top-0 right-0 w-7 h-7 border-t-2 border-r-2 border-green-400 rounded-tr-xl" />
          <div className="absolute bottom-0 left-0 w-7 h-7 border-b-2 border-l-2 border-green-400 rounded-bl-xl" />
          <div className="absolute bottom-0 right-0 w-7 h-7 border-b-2 border-r-2 border-green-400 rounded-br-xl" />
        </div>
      </div>

      {/* ── Camera error ── */}
      {camError && (
        <div className="absolute inset-0 flex items-center justify-center px-8 text-center">
          <div>
            <p className="text-white text-lg font-medium mb-2">Camera access denied</p>
            <p className="text-gray-400 text-sm">Allow camera access in your browser settings and refresh.</p>
          </div>
        </div>
      )}

      {/* ── Result panel ── */}
      {result && meta ? (
        <div className="absolute bottom-0 left-0 right-0 px-5 pb-10 pt-8">

          {/* Low confidence warning */}
          {result.low_confidence && (
            <div className="mb-4 px-4 py-2.5 bg-yellow-500/15 border border-yellow-500/40 rounded-2xl text-center">
              <p className="text-yellow-400 text-sm">⚠️ Low confidence — check your local guidelines</p>
            </div>
          )}

          {/* Class name + bin label + confidence badge */}
          <div className="flex items-end justify-between mb-3">
            <div>
              <p className={`text-xs font-bold tracking-widest uppercase mb-1 ${meta.color}`}>
                {meta.label}
              </p>
              <h2 className="text-white text-5xl font-bold capitalize leading-none">
                {result.class}
              </h2>
            </div>
            <div className={`px-4 py-2 rounded-2xl border text-sm font-bold text-white ${meta.badge}`}>
              {(result.confidence * 100).toFixed(0)}%
            </div>
          </div>

          {/* Bin instruction */}
          <p className="text-gray-300 text-sm leading-relaxed mb-5">
            {result.bin_instruction}
          </p>

          {/* Top 3 confidence bars */}
          <div className="space-y-2">
            {Object.entries(result.all_scores)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([cls, score]) => (
                <div key={cls} className="flex items-center gap-3">
                  <span className="text-gray-400 text-xs w-20 capitalize">{cls}</span>
                  <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${BIN_META[cls]?.bar ?? 'bg-gray-400'}`}
                      style={{ width: `${(score * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="text-gray-400 text-xs w-9 text-right">
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>

      ) : !camError && (
        /* ── Idle — waiting for first stable result ── */
        <div className="absolute bottom-0 left-0 right-0 px-5 pb-10 text-center">
          <p className="text-green-400 text-sm tracking-wide animate-pulse">
            Point camera at a waste item…
          </p>
        </div>
      )}
    </div>
  )
}
