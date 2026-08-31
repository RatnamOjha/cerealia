import { useMemo, useState } from 'react'
import { geoMercator, geoPath } from 'd3-geo'
import statesGeo from '../data/india_states.geojson?url'
import { useGeoJson } from '../useGeoJson'

const WIDTH = 620
const HEIGHT = 700

// Colour by the category of each state's top recommendation, so the map itself
// carries information rather than being a click target with decorative fill.
export const CATEGORY_COLORS = {
  cereal: '#e0a458',
  pulse: '#b5651d',
  horticulture: '#4c9f70',
  plantation: '#2d6a4f',
  cash: '#8e6cae',
  fibre: '#5b8ab0',
  unknown: '#3a4a52',
}

export const CATEGORY_LABELS = {
  cereal: 'Cereal',
  pulse: 'Pulse',
  horticulture: 'Horticulture',
  plantation: 'Plantation',
  cash: 'Cash crop',
  fibre: 'Fibre',
}

export default function IndiaMap({ selectedId, onSelect, topCropByState, loadingStates }) {
  const geo = useGeoJson(statesGeo)
  const [hovered, setHovered] = useState(null)

  const { pathGen, features } = useMemo(() => {
    if (!geo) return { pathGen: null, features: [] }
    const projection = geoMercator().fitSize([WIDTH, HEIGHT], geo)
    return { pathGen: geoPath(projection), features: geo.features }
  }, [geo])

  if (!geo || !pathGen) {
    return <div className="map-loading">Loading map of India…</div>
  }

  const hoveredInfo = hovered ? topCropByState[hovered.properties.id] : null

  return (
    <div className="map-wrap">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="india-svg" role="img" aria-label="Map of India by state">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {features.map((f) => {
          const id = f.properties.id
          const top = topCropByState[id]
          const isSelected = id === selectedId
          const isHovered = hovered?.properties.id === id
          const fill = top
            ? CATEGORY_COLORS[top.category] || CATEGORY_COLORS.unknown
            : CATEGORY_COLORS.unknown

          return (
            <path
              key={id}
              d={pathGen(f)}
              fill={fill}
              fillOpacity={isSelected ? 1 : isHovered ? 0.92 : 0.75}
              stroke={isSelected ? '#f5f3ef' : '#0d1b1e'}
              strokeWidth={isSelected ? 2 : 0.6}
              filter={isSelected ? 'url(#glow)' : undefined}
              className="state-path"
              onMouseEnter={() => setHovered(f)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onSelect(id)}
            >
              <title>{f.properties.name}</title>
            </path>
          )
        })}
      </svg>

      <div className="map-legend">
        <span className="legend-title">Top crop type</span>
        {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
          <span key={key} className="legend-item">
            <i style={{ background: CATEGORY_COLORS[key] }} />
            {label}
          </span>
        ))}
      </div>

      {loadingStates && <div className="map-status">Scoring all 36 states…</div>}

      {hovered && (
        <div className="map-tooltip">
          <strong>{hovered.properties.name}</strong>
          {hoveredInfo ? (
            <span>
              {hoveredInfo.display} · ₹
              {Math.round(hoveredInfo.expected / 1000)}k/ha expected
            </span>
          ) : (
            <span>click to analyse</span>
          )}
        </div>
      )}
    </div>
  )
}
