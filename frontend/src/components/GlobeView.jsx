import { useEffect, useRef, useState } from 'react'
import { geoOrthographic, geoPath, geoGraticule10, geoCentroid, geoDistance } from 'd3-geo'
import { feature } from 'topojson-client'
import worldTopo from '../data/world-110m.json'

const world = feature(worldTopo, worldTopo.objects.countries)
const india = world.features.find((f) => f.properties.name === 'India')
const INDIA_CENTER = india ? geoCentroid(india) : [78.9, 22.6]

// Rotation is the negated centroid: spinning the sphere so India faces us.
const INDIA_ROTATION = [-INDIA_CENTER[0], -INDIA_CENTER[1], 0]

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export default function GlobeView({ onEnter }) {
  const canvasRef = useRef(null)
  const [zooming, setZooming] = useState(false)
  const stateRef = useRef({ rotation: [-20, -12, 0], scale: 1, spinning: true })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
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

    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect()
      const cx = width / 2
      const cy = height / 2
      const st = stateRef.current
      const radius = Math.min(width, height) * 0.38 * st.scale

      const projection = geoOrthographic()
        .scale(radius)
        .translate([cx, cy])
        .rotate(st.rotation)
        .clipAngle(90)
      const path = geoPath(projection, ctx)

      ctx.clearRect(0, 0, width, height)

      // Ocean sphere with a soft atmospheric rim
      ctx.beginPath()
      path({ type: 'Sphere' })
      const ocean = ctx.createRadialGradient(cx - radius * 0.3, cy - radius * 0.35, radius * 0.1, cx, cy, radius)
      ocean.addColorStop(0, '#12324a')
      ocean.addColorStop(1, '#071823')
      ctx.fillStyle = ocean
      ctx.fill()

      ctx.save()
      ctx.beginPath()
      path({ type: 'Sphere' })
      ctx.clip()

      ctx.beginPath()
      path(geoGraticule10())
      ctx.strokeStyle = 'rgba(120, 190, 200, 0.13)'
      ctx.lineWidth = 0.5
      ctx.stroke()

      for (const country of world.features) {
        const isIndia = country.properties.name === 'India'
        ctx.beginPath()
        path(country)
        ctx.fillStyle = isIndia ? '#3fa34d' : '#1f4a52'
        ctx.fill()
        ctx.strokeStyle = isIndia ? '#7fe08a' : 'rgba(10, 30, 36, 0.7)'
        ctx.lineWidth = isIndia ? 1.4 : 0.4
        ctx.stroke()
      }
      ctx.restore()

      // Atmosphere glow
      ctx.beginPath()
      ctx.arc(cx, cy, radius * 1.02, 0, Math.PI * 2)
      const glow = ctx.createRadialGradient(cx, cy, radius * 0.9, cx, cy, radius * 1.18)
      glow.addColorStop(0, 'rgba(110, 220, 190, 0.22)')
      glow.addColorStop(1, 'rgba(110, 220, 190, 0)')
      ctx.fillStyle = glow
      ctx.fill()

      // Marker pin on India while idle
      if (st.spinning) {
        // geoPath bound to a canvas context always returns undefined, so
        // visibility has to be tested against the hemisphere directly.
        const centre = [-st.rotation[0], -st.rotation[1]]
        const p = geoDistance(INDIA_CENTER, centre) < Math.PI / 2 ? projection(INDIA_CENTER) : null
        if (p) {
          const [px, py] = p
          const pulse = 6 + Math.sin(Date.now() / 350) * 2.5
          ctx.beginPath()
          ctx.arc(px, py, pulse, 0, Math.PI * 2)
          ctx.strokeStyle = 'rgba(160, 255, 180, 0.85)'
          ctx.lineWidth = 1.6
          ctx.stroke()
        }
      }
    }

    const tick = () => {
      const st = stateRef.current
      if (st.spinning) {
        st.rotation = [st.rotation[0] + 0.16, st.rotation[1], 0]
      }
      draw()
      frame = requestAnimationFrame(tick)
    }
    tick()

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

    // Take the short way round rather than unwinding a full turn.
    const from = [...st.rotation]
    const target = [...INDIA_ROTATION]
    let delta = ((target[0] - from[0]) % 360 + 540) % 360 - 180
    const duration = 1500
    const start = performance.now()

    const step = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const e = easeInOut(t)
      st.rotation = [from[0] + delta * e, from[1] + (target[1] - from[1]) * e, 0]
      st.scale = 1 + 1.6 * e
      if (t < 1) requestAnimationFrame(step)
      else onEnter()
    }
    requestAnimationFrame(step)
  }

  return (
    <div className="globe-view">
      <canvas ref={canvasRef} className="globe-canvas" />
      <div className={`globe-overlay ${zooming ? 'fading' : ''}`}>
        <h1 className="globe-title">
          Krishi<span>Mitra</span>
        </h1>
        <p className="globe-sub">
          AI crop recommendation and scheme advisory for Indian farmers
        </p>
        <button className="enter-btn" onClick={flyToIndia} disabled={zooming}>
          {zooming ? 'Locating India…' : 'Explore India →'}
        </button>
        <p className="globe-hint">
          36 states &amp; UTs · 22 crops · ranked by risk-adjusted expected return
        </p>
      </div>
    </div>
  )
}
