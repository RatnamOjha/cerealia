const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8010'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const getHealth = () => request('/api/health')
export const getStates = () => request('/api/states')

export const recommendForState = (stateId, { landHa = 1, topN = 6, overrides = null } = {}) =>
  request('/api/recommend/state', {
    method: 'POST',
    body: JSON.stringify({ state_id: stateId, land_ha: landHa, top_n: topN, overrides }),
  })

export const sendChat = (message, contextNote, history, lang = 'auto') =>
  request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, context_note: contextNote, history, lang }),
  })
