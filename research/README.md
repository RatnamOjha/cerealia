# Cerealia research evaluation

This suite evaluates the committed classifier and calls `backend/train.py` factories and `backend/app/recommender.py` directly. It does not retrain from a serving request and adds no production dependency. `research/requirements.txt` contains plotting and analysis dependencies only.

## Reproduce

From the repository root, install the exact research pins in `research/requirements.txt` into Python 3.12.2 and run:

```bash
python research/run_all.py --seeds 42 43 44 45 46
python -m unittest discover -s research/tests
python research/verify_reproducibility.py
```

The first run writes versioned results into `research/results/`. It records the source Git SHA, dataset hashes, library versions, seeds, run time and hardware. Main experiments use five fixed stratified folds; each fold has matched original train/test observations, training-only feature scales and training-only augmentation. Noise is Gaussian with SD equal to the requested percentage of each training feature's SD, shared across models.

Protocol v1 accepts exactly the five seeds `42 43 44 45 46`. Other seed lists fail fast rather than silently changing the documented experiment count.

## Legacy versus stricter protocol

`reproduction_metrics.json` preserves the committed metrics and re-runs the legacy seed-42 holdout and five-fold noise benchmark with `train.py` functions. Legacy augmentation SD is computed over the entire dataset before splitting, and the candidate benchmark selects on all rows before the final holdout is drawn. The legacy holdout is therefore not independent of model selection. Strict repeated-split results are separately stored in `classifier_runs.csv` and `classifier_summary.csv`.

## Ranking interpretation

The primary five-way ablation holds candidate eligibility, regional prior, fitness gate, source data and production normalization fixed. The return-only term includes agro-climatic fitness and regional evidence in its existing expected-return calculation. Additional raw-profit and classifier-probability rankings retain the gate and regional prior; the no-prior and no-gate comparisons are explicitly separate because they change candidate eligibility. Historical cultivation share is descriptive only: the same historical records inform the production prior. Profit and irrigation measures are modelled estimates, not observed income or water savings.

Ranking input perturbations are independent of coefficient preference sensitivity. Gaussian perturbations are applied independently to the seven state-profile averages using fixed seeds and percentages of the classifier dataset feature SD. This is a controlled sensitivity check, not a sensor error model: it does not represent correlated soil/weather changes, instrument calibration, time variation, or field measurement streams. Rankings continue through production code.

## Data and limits

The classifier dataset is the repository's 2,200-row, 22-class `Crop_recommendation.csv`; it appears synthetic and has no external untouched evaluation set in this repository. Region/yield/risk artifacts derive from the packaged Government of India crop statistics and curated crop economic data. Area-share missingness is retained as missing, not converted to zero: 10 of 36 Full Cerealia top recommendations have no matched state area-share value in this run. Sparse/missing regional yield and area values fall back through the existing production engine and are identified by source fields. No external dataset is available for validation.

To verify reproducibility under the pins, run `python research/verify_reproducibility.py`. It repeats protocol v1 into a temporary output directory and compares hashes for numeric artifacts with the committed run. Timestamped manifests and PDF metadata are excluded from byte-for-byte comparison.

## Artifacts

CSV, JSON and paired PDF/300-DPI PNG figures live under `research/results/`. Integrity tests verify result counts, confusion-matrix invariants, paired-test contingency totals and coverage of all 36 state/UT profiles.
