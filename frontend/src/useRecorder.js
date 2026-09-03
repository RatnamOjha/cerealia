import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Microphone capture with a live level meter and silence auto-stop.
 *
 * This feeds server-side speech-to-text, which is the better path than the
 * browser's own recogniser: the API key stays on the server, Hindi accuracy is
 * higher, and MediaRecorder works in every browser rather than Chrome only.
 *
 * The silence detection matters more than it looks. A farmer holding a phone at
 * arm's length will not reliably find and press "stop" — the recording has to
 * end itself when they finish speaking.
 */

// Ordered by preference: Opus is small and widely accepted; mp4/aac is the
// Safari path.
const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
]

const SILENCE_LEVEL = 0.035        // RMS below this counts as silence
const SILENCE_MS = 1800            // stop after this much quiet, once speech began
const MIN_SPEECH_MS = 400          // ignore a stray click as "speech"
const MAX_MS = 30000               // hard cap so nothing records forever

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null
  return MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || ''
}

export function useAudioRecorder() {
  const [state, setState] = useState('idle')   // idle | recording | processing
  const [level, setLevel] = useState(0)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState(null)

  const mediaRef = useRef(null)
  const streamRef = useRef(null)
  const audioCtxRef = useRef(null)
  const rafRef = useRef(null)
  const chunksRef = useRef([])
  const doneRef = useRef(null)
  const cancelledRef = useRef(false)

  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== 'undefined'

  const teardown = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (audioCtxRef.current?.state !== 'closed') {
      audioCtxRef.current?.close().catch(() => {})
    }
    audioCtxRef.current = null
    mediaRef.current = null
    setLevel(0)
  }, [])

  const stop = useCallback(() => {
    if (mediaRef.current?.state === 'recording') {
      mediaRef.current.stop()
    }
  }, [])

  const cancel = useCallback(() => {
    cancelledRef.current = true
    stop()
    teardown()
    setState('idle')
    setSeconds(0)
  }, [stop, teardown])

  const start = useCallback(
    async (onComplete) => {
      if (!supported) {
        setError('unsupported')
        return
      }
      setError(null)
      cancelledRef.current = false
      doneRef.current = onComplete
      chunksRef.current = []

      let stream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })
      } catch (err) {
        setError(err?.name === 'NotAllowedError' ? 'denied' : 'mic-failed')
        return
      }
      streamRef.current = stream

      const mimeType = pickMimeType()
      let recorder
      try {
        recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      } catch {
        setError('recorder-failed')
        teardown()
        return
      }
      mediaRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data?.size) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const type = recorder.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type })
        teardown()
        setSeconds(0)
        if (cancelledRef.current) {
          setState('idle')
          return
        }
        // Under ~4 KB is a button press, not a question.
        if (blob.size < 4000) {
          setState('idle')
          setError('too-short')
          return
        }
        setState('processing')
        doneRef.current?.(blob, type)
      }

      // --- level metering + silence detection ---
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const ctx = new AudioCtx()
      audioCtxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0.75
      source.connect(analyser)
      const buf = new Uint8Array(analyser.fftSize)

      const startedAt = performance.now()
      let speechStartedAt = 0
      let quietSince = 0

      const tick = () => {
        analyser.getByteTimeDomainData(buf)
        let sum = 0
        for (let i = 0; i < buf.length; i += 1) {
          const v = (buf[i] - 128) / 128
          sum += v * v
        }
        const rms = Math.sqrt(sum / buf.length)
        // Quiet, distant speech is the normal case for someone holding a
        // phone at arm's length, so the meter is deliberately sensitive.
        setLevel(Math.min(1, Math.sqrt(rms) * 2.1))

        const now = performance.now()
        setSeconds((now - startedAt) / 1000)

        if (rms > SILENCE_LEVEL) {
          if (!speechStartedAt) speechStartedAt = now
          quietSince = 0
        } else if (speechStartedAt && now - speechStartedAt > MIN_SPEECH_MS) {
          if (!quietSince) quietSince = now
          else if (now - quietSince > SILENCE_MS) {
            stop()
            return
          }
        }

        if (now - startedAt > MAX_MS) {
          stop()
          return
        }
        rafRef.current = requestAnimationFrame(tick)
      }

      recorder.start(100)
      setState('recording')
      rafRef.current = requestAnimationFrame(tick)
    },
    [supported, stop, teardown],
  )

  const finish = useCallback(() => setState('idle'), [])

  useEffect(() => () => { cancelledRef.current = true; teardown() }, [teardown])

  return { supported, state, level, seconds, error, start, stop, cancel, finish, setError }
}
