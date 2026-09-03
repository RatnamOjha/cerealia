import { useEffect, useRef, useState } from 'react'
import { sendChat } from '../api'
import { useSpeechInput, useSpeechOutput } from '../useSpeech'

const COPY = {
  en: {
    title: 'Scheme Advisor',
    greeting:
      'Namaste! Ask me about government schemes — crop insurance, loans, irrigation subsidies, soil testing, MSP. I answer from official scheme records.',
    placeholder: 'Ask about a scheme…',
    send: 'Send',
    fab: 'Scheme advisor',
    listening: 'Listening…',
    micHint: 'Tap to speak',
    micUnsupported: 'Voice input needs Chrome or Edge',
    indexed: '12 central schemes indexed',
    grok: 'Grok · grounded on official schemes',
    local: 'Local scheme database (no API key)',
    speak: 'Read aloud',
    stop: 'Stop',
    suggestions: [
      'What subsidy can I get for drip irrigation?',
      'How do I insure my crop against drought?',
      'I need a loan to buy seeds. What are my options?',
      'How do I get my soil tested for free?',
    ],
  },
  hi: {
    title: 'योजना सलाहकार',
    greeting:
      'नमस्ते! सरकारी योजनाओं के बारे में पूछिए — फसल बीमा, कर्ज़, सिंचाई सब्सिडी, मिट्टी जाँच, एमएसपी। मैं आधिकारिक योजना रिकॉर्ड से जवाब देता हूँ।',
    placeholder: 'योजना के बारे में पूछें…',
    send: 'भेजें',
    fab: 'योजना सलाहकार',
    listening: 'सुन रहा हूँ…',
    micHint: 'बोलने के लिए दबाएँ',
    micUnsupported: 'आवाज़ के लिए Chrome या Edge चाहिए',
    indexed: '12 केंद्रीय योजनाएँ उपलब्ध',
    grok: 'Grok · आधिकारिक योजनाओं पर आधारित',
    local: 'स्थानीय योजना डेटाबेस (बिना API key)',
    speak: 'सुनें',
    stop: 'रोकें',
    suggestions: [
      'ड्रिप सिंचाई पर कितनी सब्सिडी मिलती है?',
      'सूखे से फसल बर्बाद हो जाए तो बीमा कैसे मिलेगा?',
      'बीज खरीदने के लिए कर्ज़ कहाँ से लूँ?',
      'मिट्टी की जाँच मुफ़्त में कैसे कराएँ?',
    ],
  },
}

export default function ChatPanel({ open, onToggle, contextNote }) {
  const [lang, setLang] = useState('hi')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState(null)
  const endRef = useRef(null)

  const t = COPY[lang]
  const speech = useSpeechInput(lang)
  const voice = useSpeechOutput()

  // The greeting follows the language toggle until the farmer has actually
  // said something; after that, switching language must not wipe the thread.
  useEffect(() => {
    setMessages((prev) =>
      prev.length === 0 || (prev.length === 1 && prev[0].greeting)
        ? [{ role: 'assistant', content: COPY[lang].greeting, greeting: true, lang }]
        : prev,
    )
  }, [lang])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open])

  useEffect(() => {
    if (!open) {
      speech.stop()
      voice.cancel()
    }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  const submit = async (text) => {
    const question = (text ?? input).trim()
    if (!question || busy) return
    setInput('')
    voice.cancel()
    const next = [...messages, { role: 'user', content: question }]
    setMessages(next)
    setBusy(true)
    try {
      const res = await sendChat(
        question,
        contextNote,
        // Send only role and content. The message objects also carry UI state
        // (sources, mode, the greeting flag) which the API rightly rejects.
        next.slice(-6).map((m) => ({ role: m.role, content: m.content })),
        lang,
      )
      setMode(res.mode)
      setMessages([
        ...next,
        {
          role: 'assistant',
          content: res.answer,
          sources: res.sources,
          mode: res.mode,
          lang: res.lang || lang,
        },
      ])
    } catch (err) {
      setMessages([
        ...next,
        {
          role: 'assistant',
          content:
            lang === 'hi'
              ? `सलाह सेवा से संपर्क नहीं हो पाया। कृपया जाँचें कि बैकएंड पोर्ट 8010 पर चल रहा है।\n\n(${err.message})`
              : `I could not reach the advisory service. Make sure the backend is running on port 8010.\n\n(${err.message})`,
          error: true,
          lang,
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  const toggleMic = () => {
    if (speech.listening) {
      speech.stop()
      return
    }
    voice.cancel()
    // Send as soon as a final transcript arrives — a farmer using voice should
    // not have to find and press a second button to submit.
    speech.start((transcript) => submit(transcript))
  }

  return (
    <>
      <button className={`chat-fab ${open ? 'hidden' : ''}`} onClick={onToggle} aria-label={t.fab}>
        <span className="fab-icon">🎙</span>
        <span className="fab-text">{t.fab}</span>
      </button>

      <div className={`chat-drawer ${open ? 'open' : ''}`}>
        <header className="chat-head">
          <div className="chat-head-left">
            <h3>{t.title}</h3>
            <span className="chat-mode">
              {mode === 'grok' ? t.grok : mode ? t.local : t.indexed}
            </span>
          </div>
          <div className="chat-head-right">
            <div className="lang-toggle" role="group" aria-label="Language">
              <button
                className={lang === 'hi' ? 'on' : ''}
                onClick={() => setLang('hi')}
                lang="hi"
              >
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
              <div className="msg-text" lang={m.lang === 'hi' ? 'hi' : 'en'}>{m.content}</div>
              {m.role === 'assistant' && !m.error && voice.supported && (
                <button
                  className="speak-btn"
                  onClick={() =>
                    voice.speaking ? voice.cancel() : voice.speak(m.content, m.lang || lang)
                  }
                >
                  {voice.speaking ? `■ ${t.stop}` : `▶ ${t.speak}`}
                  {!voice.hasVoice(m.lang || lang) && m.lang === 'hi' && (
                    <span className="voice-warn" title="No Hindi voice installed on this device"> ⚠</span>
                  )}
                </button>
              )}
              {m.sources?.length > 0 && (
                <div className="msg-sources">
                  {m.sources.map((s) => (
                    <a key={s.portal} href={s.portal} target="_blank" rel="noreferrer">
                      {s.name.split(' (')[0]}
                    </a>
                  ))}
                </div>
              )}
            </div>
          ))}
          {busy && <div className="msg assistant"><div className="typing"><i /><i /><i /></div></div>}
          <div ref={endRef} />
        </div>

        {messages.length <= 1 && (
          <div className="chat-suggestions">
            {t.suggestions.map((sug) => (
              <button key={sug} lang={lang} onClick={() => submit(sug)}>{sug}</button>
            ))}
          </div>
        )}

        {speech.listening && (
          <div className="listening-bar">
            <span className="pulse-dot" />
            {t.listening}
            {speech.interim && <em>“{speech.interim}”</em>}
          </div>
        )}
        {speech.error === 'not-allowed' && (
          <div className="listening-bar err">
            {lang === 'hi' ? 'माइक की अनुमति नहीं मिली' : 'Microphone permission denied'}
          </div>
        )}
        {speech.error === 'network' && (
          <div className="listening-bar err">
            {lang === 'hi'
              ? 'आवाज़ पहचान के लिए इंटरनेट चाहिए'
              : 'Voice recognition needs an internet connection'}
          </div>
        )}

        <form
          className="chat-input"
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
        >
          <button
            type="button"
            className={`mic-btn ${speech.listening ? 'live' : ''}`}
            onClick={toggleMic}
            disabled={!speech.supported || busy}
            title={speech.supported ? t.micHint : t.micUnsupported}
            aria-label={t.micHint}
          >
            🎙
          </button>
          <input
            value={input}
            lang={lang}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t.placeholder}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>{t.send}</button>
        </form>
      </div>
    </>
  )
}
