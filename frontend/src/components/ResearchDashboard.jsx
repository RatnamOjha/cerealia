import { useEffect, useState } from 'react'

const FIGURES = [
  ['Classifier robustness', 'classifier_noise_accuracy.png'],
  ['Repeated split variability', 'classifier_repeated_split_variability.png'],
  ['Clean confusion matrix', 'confusion_0pct.png'],
  ['Confusion matrix at 20% noise', 'confusion_20pct.png'],
  ['Confusion matrix at 50% noise', 'confusion_50pct.png'],
  ['Per-class recall', 'per_class_recall.png'],
  ['Ranking trade-offs', 'ranking_ablation_tradeoffs.png'],
  ['Weight sensitivity', 'weight_sensitivity_heatmap.png'],
  ['Input perturbation stability', 'ranking_input_perturbation_stability.png'],
  ['Regional case studies', 'regional_case_studies.png'],
]

const DOWNLOADS = [
  'classifier_summary.csv', 'classifier_runs.csv', 'predictions.csv',
  'classification_report.csv', 'confusion_summary.csv', 'paired_tests.csv',
  'ranking_ablation_by_region.csv', 'ranking_ablation_summary.csv',
  'weight_sensitivity.csv', 'ranking_stability.csv', 'manifest.json',
  'ranking_ablation_candidates.csv', 'legacy_predictions.csv', 'reproduction_metrics.json', 'manifest.sha256',
]

export default function ResearchDashboard({ onBack }) {
  const [manifest, setManifest] = useState(null)
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    Promise.all(['/research-artifacts/manifest.json', '/research-artifacts/dashboard_summary.json'].map((url) =>
      fetch(url).then((r) => { if (!r.ok) throw new Error('Precomputed evaluation artifacts are not available yet.'); return r.json() })))
      .then(([m, s]) => { setManifest(m); setSummary(s) }).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="research-page">
      <header className="research-header">
        <button className="research-back" onClick={onBack}>← Cerealia</button>
        <div><span className="eyebrow">RESEARCH &amp; EVALUATION</span><h1>Evidence behind the recommendations</h1></div>
        <p>Versioned experiments from the production classifier and ranking engine.</p>
      </header>
      {error && <p className="research-notice">{error}</p>}
      {manifest && <>
        <section className="research-meta">
          <div><span>Evaluation run</span><strong>{manifest.run_id}</strong></div>
          <div><span>Source commit</span><strong>{manifest.git_commit_sha?.slice(0, 12)}</strong></div>
          <div><span>Protocol</span><strong>{manifest.protocol_version}</strong></div>
          <div><span>Classifier runs</span><strong>{manifest.checks?.prediction_rows?.toLocaleString()} predictions</strong></div>
        </section>
        <section className="research-method">
          {summary && <>
            <h2>Benchmark overview</h2>
            <div className="research-metric-cards">
              <div><span>Legacy clean holdout</span><strong>{(summary.legacy.clean_holdout_accuracy * 100).toFixed(2)}%</strong><small>Repository benchmark</small></div>
              <div><span>Legacy holdout at 20% noise</span><strong>{(summary.legacy.noise20_holdout_accuracy * 100).toFixed(2)}%</strong><small>Repository benchmark</small></div>
              <div><span>Legacy five-fold at 20% noise</span><strong>{(summary.legacy.noise20_cv_accuracy * 100).toFixed(2)}%</strong><small>Model-selection benchmark</small></div>
              <div><span>Strict Extra Trees, clean mean</span><strong>{((summary.strict_classifier.find((x) => x.model === 'Extra Trees (augmentation)' && x.noise_pct === 0)?.mean_accuracy || 0) * 100).toFixed(2)}%</strong><small>Five matched splits</small></div>
            </div>
          </>}
          <h2>What these results measure</h2>
          <p><b>Production model:</b> the repository Extra Trees pipeline uses seven features, 150 trees and four jittered training copies. Legacy benchmark values and the stricter repeated-split evaluation are reported separately.</p>
          <p><b>Stricter evaluation:</b> five matched stratified splits, training-only noise scales, training-only augmentation and the same noisy holdout observations for all six classifiers. Noise is measured as a percentage of each feature’s training standard deviation.</p>
          <p><b>Ranking evidence:</b> the ablation calls the production recommendation engine. Profit, irrigation gap and risk are modeled estimates; they are not realized farmer income or measured water savings. Historical cultivation agreement is not independent validation because historical crop data also informs the regional prior.</p>
          <p><b>Limits:</b> the packaged crop classifier dataset has 2,200 samples across 22 classes and appears synthetic. No untouched external crop dataset was available for validation.</p>
        </section>
        <section className="research-figures">
          {FIGURES.map(([label, file]) => <figure key={file}>
            <img src={`/research-artifacts/figures/${file}`} alt={label} loading="lazy" />
            <figcaption>{label}</figcaption>
          </figure>)}
        </section>
        <section className="research-downloads"><h2>Download the underlying results</h2>
          <div>{DOWNLOADS.map((file) => <a key={file} href={`/research-artifacts/${file}`} download>{file}</a>)}</div>
        </section>
      </>}
    </div>
  )
}
