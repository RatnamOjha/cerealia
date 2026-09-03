import { useEffect, useRef, useState } from 'react'
import { sendChat, transcribe } from '../api'
import { useSpeechInput, useSpeechOutput } from '../useSpeech'
import { useAudioRecorder } from '../useRecorder'
import VoiceOrb from './VoiceOrb'
import MicIcon from './MicIcon'

const COPY = {
  en: {
    title: 'Scheme Advisor',
    greeting:
      'Namaste! Ask me what to grow on your land, or about government schemes — insurance, loans, irrigation subsidies, MSP. Tap the mic and just speak.',
    placeholder: 'Ask about a scheme…',
    send: 'Send',
    fab: 'Ask by voice',
    listening: 'Listening — speak now',
    transcribing: 'Understanding what you said…',
    thinking: 'Finding the right scheme…',
    micHint: 'Hold a moment and speak',
    micUnsupported: 'Microphone not available in this browser',
    indexed: '12 central schemes indexed',
    grok: 'Grok · grounded on official schemes',
    local: 'Local scheme database (no API key)',
    speak: 'Read aloud',
    stop: 'Stop',
    cancel: 'Cancel',
    youSaid: 'You said',
    sttServer: 'Grok speech recognition',
    sttBrowser: 'Browser speech recognition',
    suggestions: [
      'Which crop should I grow here?',
      'How do I insure my crop against drought?',
      'What subsidy can I get for drip irrigation?',
      'I need a loan to buy seeds. What are my options?',
    ],
  },
  hi: {
    title: 'योजना सलाहकार',
    greeting:
      'नमस्ते! पूछिए कि आपकी ज़मीन पर कौन सी फसल उगाएँ, या सरकारी योजनाओं के बारे में — बीमा, कर्ज़, सिंचाई सब्सिडी, एमएसपी। माइक दबाइए और बस बोलिए।',
    placeholder: 'योजना के बारे में पूछें…',
    send: 'भेजें',
    fab: 'बोलकर पूछें',
    listening: 'सुन रहा हूँ — अब बोलिए',
    transcribing: 'आपकी बात समझ रहा हूँ…',
    thinking: 'सही योजना ढूँढ रहा हूँ…',
    micHint: 'दबाइए और बोलिए',
    micUnsupported: 'इस ब्राउज़र में माइक उपलब्ध नहीं है',
    indexed: '12 केंद्रीय योजनाएँ उपलब्ध',
    grok: 'Grok · आधिकारिक योजनाओं पर आधारित',
    local: 'स्थानीय योजना डेटाबेस (बिना API key)',
    speak: 'सुनें',
    stop: 'रोकें',
    cancel: 'रद्द करें',
    youSaid: 'आपने कहा',
    sttServer: 'Grok आवाज़ पहचान',
    sttBrowser: 'ब्राउज़र आवाज़ पहचान',
    suggestions: [
      'मैं यहाँ कौन सी फसल उगाऊँ?',
      'सूखे से फसल बर्बाद हो जाए तो बीमा कैसे मिलेगा?',
      'ड्रिप सिंचाई पर कितनी सब्सिडी मिलती है?',
      'बीज खरीदने के लिए कर्ज़ कहाँ से लूँ?',
    ],
  },
}

const ERRORS = {
  denied: { en: 'Microphone permission was denied.', hi: 'माइक की अनुमति नहीं मिली।' },
  'too-short': { en: 'That was too short — please hold and speak.', hi: 'बहुत छोटा था — दबाकर बोलिए।' },
  unsupported: { en: 'This browser cannot record audio.', hi: 'यह ब्राउज़र आवाज़ रिकॉर्ड नहीं कर सकता।' },
  network: { en: 'Speech recognition needs an internet connection.', hi: 'आवाज़ पहचान के लिए इंटरनेट चाहिए।' },
  'mic-failed': { en: 'Could not open the microphone.', hi: 'माइक चालू नहीं हो सका।' },
}

export default function ChatPanel({ open, onToggle, contextNote, stateId, stateName, sttServerSide }) {
  const [lang, setLang] = useState('hi')
  // One transcript per language rather than one shared list. Switching the
  // toggle used to leave the previous language's greeting and answers on
  // screen under the new language's chrome; keeping the threads apart means a
  // Hindi answer is never presented as the English one, and switching back
  // still finds the conversation where it was left.
  const [threads, setThreads] = useState({ hi: [], en: [] })
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState(null)
  const [voiceError, setVoiceError] = useState(null)
  const [autoSpeak, setAutoSpeak] = useState(false)
  const endRef = useRef(null)

  const t = COPY[lang]
  const messages = threads[lang]

  // Writes land in the thread that was active when the call was made, so an
  // answer still arriving when the user flips the toggle lands beside its own
  // question rather than in the other language.
  const setMessages = (update) =>
    setThreads((prev) => ({
      ...prev,
      [lang]: typeof update === 'function' ? update(prev[lang]) : update,
    }))

  const recorder = useAudioRecorder()
  const browserSTT = useSpeechInput(lang)
  const voice = useSpeechOutput()

  // Server-side transcription is preferred whenever a key is configured; the
  // browser recogniser is the fallback, not the default.
  const useServerSTT = sttServerSide && recorder.supported

  // Seed each thread with its own greeting the first time it is opened.
  useEffect(() => {
    setThreads((prev) =>
      prev[lang].length === 0
        ? {
            ...prev,
            [lang]: [{ role: 'assistant', content: COPY[lang].greeting, greeting: true, lang }],
          }
        : prev,
    )
  }, [lang])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open, busy])

  useEffect(() => {
    if (!open) {
      recorder.cancel()
      browserSTT.stop()
      voice.cancel()
    }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  const ask = async (question, { spoken = false } = {}) => {
    const q = question.trim()
    if (!q || busy) return
    setInput('')
    setVoiceError(null)
    voice.cancel()
    const next = [...messages, { role: 'user', content: q, spoken }]
    setMessages(next)
    setBusy(true)
    try {
      const res = await sendChat(
        q,
        contextNote,
        next.slice(-6).map((m) => ({ role: m.role, content: m.content })),
        lang,
        stateId,
      )
      setMode(res.mode)
      const reply = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
        mode: res.mode,
        lang: res.lang || lang,
      }
      setMessages([...next, reply])
      // A farmer who asked by voice expects an answer by voice.
      if (spoken || autoSpeak) voice.speak(reply.content, reply.lang)
    } catch (err) {
      setMessages([
        ...next,
        {
          role: 'assistant',
          error: true,
          lang,
          content:
            lang === 'hi'
              ? `सलाह सेवा से संपर्क नहीं हो पाया। जाँचें कि बैकएंड पोर्ट 8010 पर चल रहा है।\n\n(${err.message})`
              : `I could not reach the advisory service. Check the backend is running on port 8010.\n\n(${err.message})`,
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  const handleRecorded = async (blob, mimeType) => {
    try {
      const res = await transcribe(blob, mimeType, lang)
      recorder.finish()
      if (res.ok && res.text) {
        setAutoSpeak(true)
        ask(res.text, { spoken: true })
      } else if (res.error === 'no_key') {
        // Key vanished between page load and now — drop to the browser path.
        browserSTT.start((txt) => ask(txt, { spoken: true }))
      } else {
        setVoiceError(res.detail || 'Could not recognise that.')
      }
    } catch (err) {
      recorder.finish()
      setVoiceError(err.message)
    }
  }

  const toggleMic = () => {
    setVoiceError(null)
    if (recorder.state === 'recording') return recorder.stop()
    if (browserSTT.listening) return browserSTT.stop()
    voice.cancel()
    if (useServerSTT) recorder.start(handleRecorded)
    else browserSTT.start((txt) => { setAutoSpeak(true); ask(txt, { spoken: true }) })
  }

  const listening = recorder.state === 'recording' || browserSTT.listening
  const processing = recorder.state === 'processing'
  const micBusy = listening || processing
  const micAvailable = useServerSTT || browserSTT.supported
  const errKey = recorder.error || browserSTT.error
  const errText = ERRORS[errKey]?.[lang] || voiceError

  return (
    <>
      <button className={`chat-fab ${open ? 'hidden' : ''}`} onClick={onToggle} aria-label={t.fab}>
        <span className="fab-icon"><MicIcon size={16} /></span>
        <span className="fab-text">{t.fab}</span>
      </button>

      <div className={`chat-drawer ${open ? 'open' : ''}`}>
        <header className="chat-head">
          <div className="chat-head-left">
            <h3>{t.title}</h3>
            <span className="chat-mode">
              {mode === 'grok' ? t.grok : mode ? t.local : t.indexed}
              {micAvailable && ` · ${useServerSTT ? t.sttServer : t.sttBrowser}`}
            </span>
          </div>
          <div className="chat-head-right">
            <div className="lang-toggle" role="group" aria-label="Language">
              <button className={lang === 'hi' ? 'on' : ''} onClick={() => setLang('hi')} lang="hi">
                हिंदी
              </button>
              <button className={lang === 'en' ? 'on' : ''} onClick={() => setLang('en')}>
                EN
              </button>
            </div>
            <button className="chat-close" onClick={onToggle} aria-label="Close">×</button>
          </div>
        </header>

        <div className="chat-body">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role} ${m.error ? 'error' : ''}`}>
              {m.spoken && (
                <span className="spoken-tag"><MicIcon size={10} /> {t.youSaid}</span>
              )}
              <div className="msg-text" lang={m.lang === 'hi' ? 'hi' : 'en'}>{m.content}</div>
              {m.role === 'assistant' && !m.error && voice.supported && (
                <button
                  className={`speak-btn ${voice.speaking ? 'on' : ''}`}
                  onClick={() => (voice.speaking ? voice.cancel() : voice.speak(m.content, m.lang || lang))}
                >
                  {voice.speaking ? `■ ${t.stop}` : `▶ ${t.speak}`}
                  {m.lang === 'hi' && !voice.hasVoice('hi') && (
                    <span className="voice-warn" title="No Hindi voice on this device"> ⚠</span>
                  )}
                </button>
              )}
              {m.sources?.length > 0 && (
                <div className="msg-sources">
                  {m.sources.map((src, si) =>
                    // Crop sources carry no portal, so they render as plain
                    // chips rather than links to nowhere. The index keeps keys
                    // unique where several sources share an empty portal.
                    src.portal ? (
                      <a
                        key={`${src.portal}-${si}`}
                        href={src.portal}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {src.name.split(' (')[0]}
                      </a>
                    ) : (
                      <span key={`${src.name}-${si}`} className="source-chip">
                        {src.name.split(' (')[0]}
                      </span>
                    ),
                  )}
                </div>
              )}
            </div>
          ))}
          {busy && (
            <div className="msg assistant">
              <div className="thinking-row">
                <div className="typing"><i /><i /><i /></div>
                <span>{t.thinking}</span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {messages.length <= 1 && !micBusy && (
          <div className="chat-suggestions">
            {t.suggestions.map((sug) => (
              <button key={sug} lang={lang} onClick={() => ask(sug)}>{sug}</button>
            ))}
          </div>
        )}

        {micBusy && (
          <div className="voice-stage">
            <VoiceOrb
              level={recorder.level}
              seconds={recorder.seconds}
              state={processing ? 'processing' : 'recording'}
              metered={useServerSTT}
            />
            <p className="voice-status">{processing ? t.transcribing : t.listening}</p>
            {browserSTT.interim && <p className="voice-interim">“{browserSTT.interim}”</p>}
            {listening && (
              <button className="voice-cancel" onClick={() => { recorder.cancel(); browserSTT.stop() }}>
                {t.cancel}
              </button>
            )}
          </div>
        )}

        {errText && !micBusy && <div className="voice-error">{errText}</div>}

        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); ask(input) }}>
          <button
            type="button"
            className={`mic-btn ${listening ? 'live' : ''} ${processing ? 'busy' : ''}`}
            onClick={toggleMic}
            disabled={!micAvailable || busy || processing}
            title={micAvailable ? t.micHint : t.micUnsupported}
            aria-label={t.micHint}
          >
            {listening ? <span className="stop-square" /> : <MicIcon />}
          </button>
          <input
            value={input}
            lang={lang}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t.placeholder}
            disabled={busy || micBusy}
          />
          <button type="submit" disabled={busy || !input.trim()}>{t.send}</button>
        </form>
      </div>
    </>
  )
}
