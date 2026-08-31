import { useState } from 'react'

const rupees = (n) =>
  n >= 100000 ? `₹${(n / 100000).toFixed(2)} L` : `₹${Math.round(n).toLocaleString('en-IN')}`

const WATER_CLASS = {
  rainfed: 'ok',
  light: 'ok',
  moderate: 'warn',
  heavy: 'bad',
}

function CropCard({ rec, rank, landHa }) {
  const [open, setOpen] = useState(false)
  const e = rec.economics
  const mismatches = rec.why?.filter((w) => w.status !== 'match') ?? []

  return (
    <div className={`crop-card ${rank === 0 ? 'top' : ''}`}>
      <div className="crop-head" onClick={() => setOpen((o) => !o)}>
        <div className="crop-rank">{rank + 1}</div>
        <div className="crop-title">
          <h4>
            {rec.display}
            {rec.msp_backed && <span className="msp-tag" title="Covered by Minimum Support Price">MSP</span>}
          </h4>
          <span className="crop-meta">
            {rec.category} · {rec.season} · {rec.regional_tier} crop here
          </span>
        </div>
        <div className="crop-profit">
          <strong>{rupees(e.net_profit_per_ha_year * landHa)}</strong>
          <span>net / year</span>
        </div>
        <button className="expand" aria-label="Details">{open ? '−' : '+'}</button>
      </div>

      <div className="crop-bars">
        <div className="bar-row">
          <label>Climate fit</label>
          <div className="bar">
            <div className="fill fit" style={{ width: `${rec.agro_fit_pct}%` }} />
          </div>
          <span className={`conf ${rec.confidence}`}>{rec.agro_fit_pct}% {rec.confidence}</span>
        </div>
        <div className="bar-row">
          <label>Water need</label>
          <div className="bar">
            <div
              className={`fill water ${WATER_CLASS[rec.water.verdict]}`}
              style={{ width: `${Math.min(100, rec.water.dependence_ratio * 100)}%` }}
            />
          </div>
          <span className="water-note">{rec.water.label}</span>
        </div>
      </div>

      {open && (
        <div className="crop-detail">
          <div className="detail-grid">
            <div>
              <span className="k">Yield used</span>
              <span className="v">{e.yield_t_ha_used} t/ha</span>
            </div>
            <div>
              <span className="k">Price</span>
              <span className="v">₹{e.price_per_quintal.toLocaleString('en-IN')}/qtl</span>
            </div>
            <div>
              <span className="k">Gross revenue</span>
              <span className="v">{rupees(e.gross_revenue_per_ha_year * landHa)}</span>
            </div>
            <div>
              <span className="k">Operating cost</span>
              <span className="v">−{rupees(e.operating_cost_per_ha_year * landHa)}</span>
            </div>
            {e.amortised_capex_per_ha_year > 0 && (
              <div>
                <span className="k">Setup (amortised)</span>
                <span className="v">−{rupees(e.amortised_capex_per_ha_year * landHa)}</span>
              </div>
            )}
            <div>
              <span className="k">Risk-adjusted</span>
              <span className="v">{rupees(e.expected_profit_per_ha_year * landHa)}</span>
            </div>
            {rec.risk?.yield_cv != null && (
              <div>
                <span className="k">Yield volatility</span>
                <span className="v">CV {rec.risk.yield_cv}</span>
              </div>
            )}
            {rec.establishment_years > 0 && (
              <div className="full">
                <span className="k">First harvest</span>
                <span className="v">
                  {rec.establishment_years} year{rec.establishment_years > 1 ? 's' : ''} after planting
                </span>
              </div>
            )}
            {rec.water.irrigation_gap_mm > 0 && (
              <div className="full">
                <span className="k">Irrigation gap</span>
                <span className="v">
                  {rec.water.irrigation_gap_mm} mm beyond rainfall
                </span>
              </div>
            )}
          </div>

          <div className="provenance">
            <span className="prov-title">Data sources</span>
            <div className="prov-row">
              <span>Yield</span>
              <b className={e.yield_source === 'curated estimate' ? 'est' : 'real'}>
                {e.yield_source}
              </b>
            </div>
            <div className="prov-row">
              <span>Grown here</span>
              <b className="real">
                {rec.evidence}
                {rec.area_share != null && ` · ${(rec.area_share * 100).toFixed(1)}% of sown area`}
              </b>
            </div>
            <div className="prov-row">
              <span>Risk</span>
              <b className={rec.risk?.source === 'curated estimate' ? 'est' : 'real'}>
                {rec.risk?.source}
              </b>
            </div>
          </div>

          <h5>Why this crop</h5>
          <div className="why-list">
            {rec.why?.slice(0, 5).map((w) => (
              <div key={w.feature} className={`why-row ${w.status}`}>
                <span className="why-label">{w.label}</span>
                <span className="why-value">
                  {w.value}{w.unit && ` ${w.unit}`}
                </span>
                <span className="why-band">
                  ideal {w.ideal_low}–{w.ideal_high}
                </span>
                <span className={`why-tag ${w.status}`}>
                  {w.status === 'match' ? '✓ in range' : w.status === 'low' ? '↓ low' : '↑ high'}
                </span>
              </div>
            ))}
          </div>
          {mismatches.length === 0 && (
            <p className="all-match">Every measured condition sits inside this crop's ideal band.</p>
          )}

          {rec.fertiliser_plan?.items?.length > 0 && (
            <>
              <h5>Fertiliser to close the nutrient gap</h5>
              <table className="fert-table">
                <thead>
                  <tr><th>Product</th><th>Qty</th><th>50 kg bags</th><th>Cost</th></tr>
                </thead>
                <tbody>
                  {rec.fertiliser_plan.items.map((i) => (
                    <tr key={i.product}>
                      <td>{i.product}</td>
                      <td>{Math.round(i.quantity_kg_ha * landHa)} kg</td>
                      <td>{(i.bags_50kg_per_ha * landHa).toFixed(1)}</td>
                      <td>{rupees(i.estimated_cost_inr * landHa)}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <td colSpan="3">Total</td>
                    <td>{rupees(rec.fertiliser_plan.total_cost_inr_per_ha * landHa)}</td>
                  </tr>
                </tfoot>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function RecommendationPanel({ data, loading, error, landHa, onLandChange, onBack }) {
  if (loading) {
    return (
      <aside className="panel">
        <div className="panel-loading"><div className="spinner" />Analysing soil and climate…</div>
      </aside>
    )
  }
  if (error) {
    return (
      <aside className="panel">
        <div className="panel-error">
          <h3>Could not reach the API</h3>
          <p>{error}</p>
          <code>cd backend && ./.venv/bin/python -m uvicorn app.main:app --port 8010</code>
        </div>
      </aside>
    )
  }
  if (!data) {
    return (
      <aside className="panel panel-empty">
        <h3>Select a state</h3>
        <p>
          Click any state on the map to see which crops suit its soil and climate,
          what each is worth per hectare, and how much irrigation it needs.
        </p>
        <p className="empty-note">
          Colours show the crop type currently ranked first in each state.
        </p>
      </aside>
    )
  }

  const s = data.state
  const h = data.headline
  const g = data.gate

  return (
    <aside className="panel">
      <button className="back-btn" onClick={onBack}>← Globe</button>

      <header className="panel-head">
        <h2>{s.name}</h2>
        <p className="zone">{s.zone} · {s.soil} soil</p>
      </header>

      <div className="site-strip">
        <div><span>N</span>{data.site.N}</div>
        <div><span>P</span>{data.site.P}</div>
        <div><span>K</span>{data.site.K}</div>
        <div><span>pH</span>{data.site.ph}</div>
        <div><span>Temp</span>{data.site.temperature}°C</div>
        <div><span>Rain</span>{data.rainfall_annual_mm}mm</div>
      </div>

      <div className="headline-box">
        <div className="headline-main">
          <span className="hl-label">Best overall</span>
          <strong>{h.balanced_pick}</strong>
          <span className={`conf ${h.confidence}`}>{h.confidence} confidence</span>
        </div>
        {h.profit_vs_fitness_differ && (
          <p className="hl-note">
            Best climate match is <b>{h.best_agronomic_fit}</b>, highest raw profit is{' '}
            <b>{h.highest_profit}</b>. The ranking balances the two.
          </p>
        )}
        <p className="gate-note">
          {g.excluded_not_cultivated_in_region} crops excluded as not cultivated here ·{' '}
          {g.crops_considered} scored · {g.crops_with_measured_yield} using measured
          government yield data
        </p>
      </div>

      <div className="land-control">
        <label htmlFor="land">Farm size: <b>{landHa} ha</b> ({(landHa * 2.47).toFixed(1)} acres)</label>
        <input
          id="land"
          type="range"
          min="0.5"
          max="10"
          step="0.5"
          value={landHa}
          onChange={(e) => onLandChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="crop-list">
        {data.recommendations.map((rec, i) => (
          <CropCard key={rec.crop} rec={rec} rank={i} landHa={landHa} />
        ))}
      </div>

      <p className="disclaimer">
        Indicative figures from national averages. Verify against your Soil Health
        Card and local mandi prices before sowing.
      </p>
    </aside>
  )
}
