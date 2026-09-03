/**
 * Microphone glyph.
 *
 * An emoji was used here first and sat visibly off-centre in its circular
 * button: emoji carry their own side bearings and baseline, which no amount of
 * flex centring fixes. An inline SVG on a square viewBox centres exactly, and
 * inherits currentColor so it works on every button state.
 */
export default function MicIcon({ size = 17 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10v1a7 7 0 0 0 14 0v-1" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </svg>
  )
}
