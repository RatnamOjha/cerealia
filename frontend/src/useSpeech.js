import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Speech input and output via the browser's Web Speech API.
 *
 * Why the platform API rather than a cloud STT service: it needs no key, no
 * per-request cost and no audio upload we have to explain to a farmer. The
 * trade-off is honest and worth stating — Chrome's recogniser streams audio to
 * Google's servers, so dictation needs a network connection even though the
 * rest of this application runs entirely offline. Speech *output* is fully
 * local, using the voices installed on the device.
 *
 * A production deployment would move dictation to Bhashini, the Government of
 * India's own speech stack, which covers 22 scheduled languages and keeps the
 * audio within Indian infrastructure.
 */

const SR =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null

const LOCALE = { hi: 'hi-IN', en: 'en-IN' }

/** Strip the markdown we render visually so it is not read out loud. */
export function speakableText(markdown) {
  return String(markdown || '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^[-•]\s*/gm, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function useSpeechInput(lang = 'hi') {
  const [listening, setListening] = useState(false)
  const [interim, setInterim] = useState('')
  const [error, setError] = useState(null)
  const recRef = useRef(null)
  const resolveRef = useRef(null)

  const supported = Boolean(SR)

  const stop = useCallback(() => {
    try {
      recRef.current?.stop()
    } catch {
      /* already stopped */
    }
    setListening(false)
  }, [])

  const start = useCallback(
    (onFinal) => {
      if (!SR) {
        setError('unsupported')
        return
      }
      setError(null)
      setInterim('')
      resolveRef.current = onFinal

      const rec = new SR()
      rec.lang = LOCALE[lang] || LOCALE.hi
      rec.interimResults = true
      rec.continuous = false
      rec.maxAlternatives = 1

      rec.onresult = (event) => {
        let finalText = ''
        let partial = ''
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const chunk = event.results[i][0].transcript
          if (event.results[i].isFinal) finalText += chunk
          else partial += chunk
        }
        setInterim(partial)
        if (finalText.trim()) {
          setInterim('')
          resolveRef.current?.(finalText.trim())
        }
      }

      rec.onerror = (event) => {
        // "no-speech" and "aborted" are the user saying nothing or cancelling;
        // neither deserves an error message.
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
          setError(event.error)
        }
        setListening(false)
      }

      rec.onend = () => {
        setListening(false)
        setInterim('')
      }

      recRef.current = rec
      try {
        rec.start()
        setListening(true)
      } catch {
        setError('start-failed')
      }
    },
    [lang],
  )

  useEffect(() => () => stop(), [stop])

  return { supported, listening, interim, error, start, stop }
}

export function useSpeechOutput() {
  const [speaking, setSpeaking] = useState(false)
  const [voices, setVoices] = useState([])

  const synth = typeof window !== 'undefined' ? window.speechSynthesis : null
  const supported = Boolean(synth)

  useEffect(() => {
    if (!synth) return undefined
    // Voices load asynchronously on most browsers and are empty on first call.
    const load = () => setVoices(synth.getVoices())
    load()
    synth.addEventListener('voiceschanged', load)
    return () => synth.removeEventListener('voiceschanged', load)
  }, [synth])

  const voiceFor = useCallback(
    (lang) => {
      const want = lang === 'hi' ? 'hi' : 'en'
      return (
        voices.find((v) => v.lang?.toLowerCase().startsWith(`${want}-in`)) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith(want)) ||
        null
      )
    },
    [voices],
  )

  const speak = useCallback(
    (text, lang = 'hi') => {
      if (!synth) return
      synth.cancel()
      const utterance = new SpeechSynthesisUtterance(speakableText(text))
      utterance.lang = LOCALE[lang] || LOCALE.hi
      const voice = voiceFor(lang)
      if (voice) utterance.voice = voice
      // Slightly slower than default: this is advice being read to someone who
      // may be acting on the numbers.
      utterance.rate = lang === 'hi' ? 0.92 : 0.98
      utterance.onstart = () => setSpeaking(true)
      utterance.onend = () => setSpeaking(false)
      utterance.onerror = () => setSpeaking(false)
      synth.speak(utterance)
    },
    [synth, voiceFor],
  )

  const cancel = useCallback(() => {
    synth?.cancel()
    setSpeaking(false)
  }, [synth])

  const hasVoice = useCallback((lang) => Boolean(voiceFor(lang)), [voiceFor])

  return { supported, speaking, speak, cancel, hasVoice }
}
