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

/** Upload a recorded clip for server-side transcription. */
export async function transcribe(blob, mimeType, lang = 'hi') {
  const form = new FormData()
  const ext = (mimeType || '').includes('mp4') ? 'm4a' : 'webm'
  form.append('audio', blob, `speech.${ext}`)
  form.append('language', lang)
  const res = await fetch(`${BASE}/api/stt`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`${res.status}: ${await res.text().catch(() => res.statusText)}`)
  return res.json()
}

export const sendChat = (message, contextNote, history, lang = 'auto') =>
  request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, context_note: contextNote, history, lang }),
  })
