import { useEffect, useRef, useState } from 'react'
import { sendChat } from '../api'

const SUGGESTIONS = [
  'What subsidy can I get for drip irrigation?',
  'How do I insure my crop against drought?',
  'I need a loan to buy seeds. What are my options?',
  'How do I get my soil tested for free?',
]

export default function ChatPanel({ open, onToggle, contextNote }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Namaste! Ask me about government schemes — crop insurance, loans, irrigation subsidies, soil testing, MSP. I answer from official scheme records.',
    },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, open])

  const submit = async (text) => {
    const question = (text ?? input).trim()
    if (!question || busy) return
    setInput('')
    const next = [...messages, { role: 'user', content: question }]
    setMessages(next)
    setBusy(true)
    try {
      const res = await sendChat(
        question,
        contextNote,
        next.filter((m) => m.role !== 'system').slice(-6),
      )
      setMode(res.mode)
      setMessages([
        ...next,
        { role: 'assistant', content: res.answer, sources: res.sources, mode: res.mode },
      ])
    } catch (err) {
      setMessages([
        ...next,
        {
          role: 'assistant',
          content: `I could not reach the advisory service. Make sure the backend is running on port 8000.\n\n(${err.message})`,
          error: true,
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button className={`chat-fab ${open ? 'hidden' : ''}`} onClick={onToggle} aria-label="Open scheme advisor">
        <span className="fab-icon">💬</span>
        <span className="fab-text">Scheme advisor</span>
      </button>

      <div className={`chat-drawer ${open ? 'open' : ''}`}>
        <header className="chat-head">
          <div>
            <h3>Scheme Advisor</h3>
            <span className="chat-mode">
              {mode === 'grok'
                ? 'Grok · grounded on official schemes'
                : mode
                  ? 'Local scheme database (no API key)'
                  : '12 central schemes indexed'}
            </span>
          </div>
          <button className="chat-close" onClick={onToggle} aria-label="Close">×</button>
        </header>

        <div className="chat-body">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role} ${m.error ? 'error' : ''}`}>
              <div className="msg-text">{m.content}</div>
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
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => submit(s)}>{s}</button>
            ))}
          </div>
        )}

        <form
          className="chat-input"
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a scheme…"
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>Send</button>
        </form>
      </div>
    </>
  )
}
