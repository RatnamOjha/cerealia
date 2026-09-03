import { useCallback, useEffect, useState } from 'react'
import GlobeView from './components/GlobeView'
import IndiaMap from './components/IndiaMap'
import RecommendationPanel from './components/RecommendationPanel'
import ChatPanel from './components/ChatPanel'
import { getHealth, recommendForState } from './api'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8010'

export default function App() {
  const [view, setView] = useState('globe')
  const [selectedId, setSelectedId] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [landHa, setLandHa] = useState(1)
  const [topByState, setTopByState] = useState({})
  const [loadingOverview, setLoadingOverview] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  // Colour the map as soon as we switch to it.
  useEffect(() => {
    if (view !== 'map' || Object.keys(topByState).length) return
    setLoadingOverview(true)
    fetch(`${API}/api/overview`)
      .then((r) => r.json())
      .then((d) => setTopByState(d.top_by_state || {}))
      .catch((err) => console.error('overview failed', err))
      .finally(() => setLoadingOverview(false))
  }, [view, topByState])

  const loadState = useCallback(async (stateId, ha) => {
    setLoading(true)
    setError(null)
    try {
      const data = await recommendForState(stateId, { landHa: ha, topN: 6 })
      setRecommendation(data)
    } catch (err) {
      setError(err.message)
      setRecommendation(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSelect = (stateId) => {
    setSelectedId(stateId)
    loadState(stateId, landHa)
  }

  // Land size only rescales rupee figures, but the backend owns that maths so
  // the panel never has to guess how capex is amortised.
  const handleLandChange = (ha) => {
    setLandHa(ha)
    if (selectedId) loadState(selectedId, ha)
  }

  const contextNote = recommendation
    ? `Farmer is looking at ${recommendation.state.name} (${recommendation.state.zone}, ` +
      `${recommendation.state.soil} soil). Top recommended crop: ` +
      `${recommendation.headline.balanced_pick}. Farm size ${landHa} ha.`
    : null

  if (view === 'globe') {
    return <GlobeView onEnter={() => setView('map')} />
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand" onClick={() => setView('globe')}>
          Cere<span>alia</span>
        </div>
        <nav className="topbar-meta">
          {health?.metrics && (
            <span className="metric-chip">
              model {(health.metrics.cv_accuracy_mean * 100).toFixed(1)}% CV
            </span>
          )}
          <span className={`metric-chip ${health ? 'ok' : 'bad'}`}>
            {health ? 'API connected' : 'API offline'}
          </span>
        </nav>
      </header>

      <main className="layout">
        <section className="map-section">
          <IndiaMap
            selectedId={selectedId}
            onSelect={handleSelect}
            topCropByState={topByState}
            loadingStates={loadingOverview}
          />
        </section>

        <RecommendationPanel
          data={recommendation}
          loading={loading}
          error={error}
          landHa={landHa}
          onLandChange={handleLandChange}
          onBack={() => setView('globe')}
        />
      </main>

      <ChatPanel
        open={chatOpen}
        onToggle={() => setChatOpen((o) => !o)}
        contextNote={contextNote}
        sttServerSide={Boolean(health?.stt?.server_side)}
      />
    </div>
  )
}
