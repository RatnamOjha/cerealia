import { useEffect, useRef, useState } from 'react'
import { geoOrthographic, geoPath, geoGraticule10, geoCentroid, geoDistance } from 'd3-geo'
import { feature } from 'topojson-client'
import worldTopo from '../data/world-110m.json'

const world = feature(worldTopo, worldTopo.objects.countries)
const india = world.features.find((f) => f.properties.name === 'India')
const INDIA_CENTER = india ? geoCentroid(india) : [78.9, 22.6]

// Rotation is the negated centroid: spinning the sphere so India faces us.
const INDIA_ROTATION = [-INDIA_CENTER[0], -INDIA_CENTER[1], 0]

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

// Precomputed so the field stays put across frames instead of flickering.
const STARS = Array.from({ length: 260 }, () => ({
  x: Math.random(),
  y: Math.random(),
  r: Math.random() * 1.15 + 0.25,
  base: Math.random() * 0.4 + 0.22,
  // Each star twinkles on its own clock; a shared one makes the sky pulse.
  speed: Math.random() * 0.0012 + 0.0004,
  phase: Math.random() * Math.PI * 2,
}))

/**
 * Sub-solar direction, so the shadow sweeps as the globe turns. Not
 * astronomically exact — it only has to read as "this side is in daylight".
 */
function sunDirection(rotationLambda) {
  const rad = ((-rotationLambda + 35) * Math.PI) / 180
  return [Math.cos(rad), Math.sin(rad)]
}

export default function GlobeView({ onEnter }) {
  const canvasRef = useRef(null)
  const [zooming, setZooming] = useState(false)
  const stateRef = useRef({ rotation: [-32, -14, 0], scale: 1, spinning: true, opacity: 1 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    let frame

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const { width, height } = canvas.getBoundingClientRect()
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = (now) => {
      const { width, height } = canvas.getBoundingClientRect()
      const cx = width / 2
      const cy = height / 2
      const st = stateRef.current
      const radius = Math.min(width, height) * 0.36 * st.scale

      ctx.clearRect(0, 0, width, height)

      // --- starfield -------------------------------------------------------
      for (const s of STARS) {
        const twinkle = Math.sin(now * s.speed + s.phase) * 0.3
        ctx.globalAlpha = Math.max(0.04, (s.base + twinkle) * st.opacity)
        ctx.beginPath()
        ctx.arc(s.x * width, s.y * height, s.r, 0, Math.PI * 2)
        ctx.fillStyle = '#cfe9e4'
        ctx.fill()
      }
      ctx.globalAlpha = st.opacity

      const projection = geoOrthographic()
        .scale(radius)
        .translate([cx, cy])
        .rotate(st.rotation)
        .clipAngle(90)
      const path = geoPath(projection, ctx)

      // --- outer atmosphere ------------------------------------------------
      const halo = ctx.createRadialGradient(cx, cy, radius * 0.94, cx, cy, radius * 1.32)
      halo.addColorStop(0, 'rgba(110, 220, 190, 0.28)')
      halo.addColorStop(0.45, 'rgba(88, 170, 200, 0.09)')
      halo.addColorStop(1, 'rgba(88, 170, 200, 0)')
      ctx.beginPath()
      ctx.arc(cx, cy, radius * 1.32, 0, Math.PI * 2)
      ctx.fillStyle = halo
      ctx.fill()

      // --- ocean, lit from the sun side ------------------------------------
      const [sx, sy] = sunDirection(st.rotation[0])
      const ocean = ctx.createRadialGradient(
        cx + sx * radius * 0.42, cy - Math.abs(sy) * radius * 0.3, radius * 0.06,
        cx, cy, radius,
      )
      ocean.addColorStop(0, '#1d5570')
      ocean.addColorStop(0.55, '#123648')
      ocean.addColorStop(1, '#04121a')
      ctx.beginPath()
      path({ type: 'Sphere' })
      ctx.fillStyle = ocean
      ctx.fill()

      ctx.save()
      ctx.beginPath()
      path({ type: 'Sphere' })
      ctx.clip()

      ctx.beginPath()
      path(geoGraticule10())
      ctx.strokeStyle = 'rgba(130, 205, 210, 0.10)'
      ctx.lineWidth = 0.5
      ctx.stroke()

      for (const country of world.features) {
        const isIndia = country.properties.name === 'India'
        ctx.beginPath()
        path(country)
        ctx.fillStyle = isIndia ? '#54b96a' : '#215058'
        ctx.fill()
        ctx.strokeStyle = isIndia ? 'rgba(150, 240, 175, 0.9)' : 'rgba(8, 26, 32, 0.75)'
        ctx.lineWidth = isIndia ? 1.5 : 0.4
        ctx.stroke()
      }

      // --- night side ------------------------------------------------------
      // A soft gradient rather than a hard terminator: this is for depth, not
      // an accurate daylight map.
      const night = ctx.createRadialGradient(
        cx + sx * radius * 0.55, cy - sy * radius * 0.55, radius * 0.15,
        cx - sx * radius * 0.35, cy + sy * radius * 0.35, radius * 1.5,
      )
      night.addColorStop(0, 'rgba(0, 0, 0, 0)')
      night.addColorStop(0.5, 'rgba(2, 10, 16, 0.30)')
      night.addColorStop(1, 'rgba(1, 6, 10, 0.80)')
      ctx.beginPath()
      path({ type: 'Sphere' })
      ctx.fillStyle = night
      ctx.fill()
      ctx.restore()

      // --- rim light -------------------------------------------------------
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(140, 230, 205, 0.42)'
      ctx.lineWidth = 1.1
      ctx.stroke()

      // --- India marker ----------------------------------------------------
      if (st.spinning) {
        const centre = [-st.rotation[0], -st.rotation[1]]
        if (geoDistance(INDIA_CENTER, centre) < Math.PI / 2) {
          const p = projection(INDIA_CENTER)
          if (p) {
            const [px, py] = p
            const t = (now % 2200) / 2200
            // Two rings half a cycle apart, so one is always expanding.
            for (const offset of [0, 0.5]) {
              const k = (t + offset) % 1
              ctx.beginPath()
              ctx.arc(px, py, 5 + k * 22, 0, Math.PI * 2)
              ctx.strokeStyle = `rgba(160, 255, 185, ${(1 - k) * 0.55 * st.opacity})`
              ctx.lineWidth = 1.4
              ctx.stroke()
            }
            ctx.beginPath()
            ctx.arc(px, py, 3.2, 0, Math.PI * 2)
            ctx.fillStyle = '#b6ffc9'
            ctx.fill()
          }
        }
      }
      ctx.globalAlpha = 1
    }

    const tick = (now) => {
      const st = stateRef.current
      if (st.spinning) st.rotation = [st.rotation[0] + 0.12, st.rotation[1], 0]
      draw(now)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  const flyToIndia = () => {
    if (zooming) return
    setZooming(true)
    const st = stateRef.current
    st.spinning = false

    const from = [...st.rotation]
    const target = [...INDIA_ROTATION]
    // Take the short way round rather than unwinding a whole turn.
    const delta = ((target[0] - from[0]) % 360 + 540) % 360 - 180
    const duration = 1700
    const start = performance.now()

    const step = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const e = easeInOutCubic(t)
      st.rotation = [from[0] + delta * e, from[1] + (target[1] - from[1]) * e, 0]
      // Accelerate into the planet, then fade so the map can take over.
      st.scale = 1 + 2.4 * e * e
      st.opacity = t > 0.65 ? Math.max(0, 1 - (t - 0.65) / 0.35) : 1
      if (t < 1) requestAnimationFrame(step)
      else onEnter()
    }
    requestAnimationFrame(step)
  }

  return (
    <div className="globe-view">
      <canvas ref={canvasRef} className="globe-canvas" />
      <div className={`globe-overlay ${zooming ? 'fading' : ''}`}>
        <span className="globe-eyebrow">Global crop intelligence</span>
        <h1 className="globe-title">
          Cere<span>alia</span>
        </h1>
        <p className="globe-sub">
          AI crop intelligence for farmers — soil, climate, price and risk in one answer
        </p>
        <button className="enter-btn" onClick={flyToIndia} disabled={zooming}>
          <span>{zooming ? 'Approaching India' : 'Explore India'}</span>
          <i className="enter-arrow">→</i>
        </button>
        <p className="globe-hint">
          36 states &amp; UTs · 22 crops · ranked by risk-adjusted expected return
        </p>
      </div>
      <div className="globe-vignette" />
    </div>
  )
}
