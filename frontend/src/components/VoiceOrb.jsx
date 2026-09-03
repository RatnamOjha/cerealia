/**
 * Live microphone feedback.
 *
 * A farmer speaking into a laptop needs to see that the machine is hearing
 * them — a static "recording…" label gives no such confidence.
 *
 * Two sources drive this. Server-side transcription records through an
 * analyser, so the bars follow real RMS level and silence genuinely looks
 * like silence. The browser recogniser exposes no level data at all, so the
 * bars self-animate instead: still honest about "listening", never a frozen
 * meter pretending to be live.
 */
export default function VoiceOrb({ level = 0, seconds = 0, state = 'recording', metered = true }) {
  // Fixed multipliers give the cluster a centre-weighted shape rather than
  // every bar moving as one block.
  const shape = [0.45, 0.7, 1, 0.82, 1, 0.62, 0.4]

  return (
    <div className={`voice-orb ${state} ${metered ? 'metered' : 'unmetered'}`}>
      <div
        className="orb-ring"
        style={metered ? { transform: `scale(${1 + level * 0.35})` } : undefined}
      />
      <div className="orb-core">
        {state === 'processing' ? (
          <div className="orb-spinner" />
        ) : (
          <div className="orb-bars">
            {shape.map((m, i) => (
              <i
                key={i}
                style={
                  metered
                    ? { height: `${Math.max(14, Math.min(100, level * 100 * m * 1.6 + 14))}%` }
                    : { animationDelay: `${i * 0.11}s` }
                }
              />
            ))}
          </div>
        )}
      </div>
      {state === 'recording' && metered && (
        <span className="orb-timer">{seconds.toFixed(1)}s</span>
      )}
    </div>
  )
}
