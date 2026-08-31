import { useEffect, useState } from 'react'

/** Fetch a bundled .geojson asset once and cache it across mounts. */
const cache = new Map()

export function useGeoJson(url) {
  const [data, setData] = useState(() => cache.get(url) || null)

  useEffect(() => {
    if (cache.has(url)) {
      setData(cache.get(url))
      return
    }
    let cancelled = false
    fetch(url)
      .then((r) => r.json())
      .then((json) => {
        cache.set(url, json)
        if (!cancelled) setData(json)
      })
      .catch((err) => console.error('failed to load geojson', url, err))
    return () => {
      cancelled = true
    }
  }, [url])

  return data
}
