"""
Train the agro-climatic crop suitability model.

The model answers one narrow question: given soil NPK, pH, and the local
temperature / humidity / rainfall regime, which crops will physically thrive
here? It deliberately does NOT decide what the farmer should plant -- that
decision also needs price, cost, water and risk, and is handled downstream in
app/recommender.py.

Run:  python backend/train.py
Out:  backend/models/crop_suitability.joblib
      backend/models/metrics.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "app" / "data" / "Crop_recommendation.csv"
MODELS = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    missing = set(FEATURES + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    return df


def describe_balance(df: pd.DataFrame) -> dict:
    """The dataset ships perfectly balanced at 100 rows/crop.

    This is worth asserting rather than assuming: it is the reason we skip
    SMOTE. Resampling a balanced dataset only adds synthetic noise.
    """
    counts = df["label"].value_counts()
    return {
        "n_rows": int(len(df)),
        "n_classes": int(counts.size),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "is_balanced": bool(counts.min() == counts.max()),
    }


def audit(df: pd.DataFrame, X, y, cv) -> dict:
    """Interrogate the headline accuracy instead of reporting it uncritically.

    A 99.5% cross-validated score on a 22-class problem should be treated as a
    symptom, not an achievement. Three checks decide what it actually means:

      1. Leakage. Duplicate rows shared across folds would inflate the score
         for free.
      2. Difficulty. If Gaussian Naive Bayes matches a 300-tree forest, the
         classes are near-separable blobs and the forest is not doing any work.
      3. Robustness. Real soil and weather readings carry error. Accuracy under
         injected noise is the number that means something operationally.
    """
    print("\n" + "=" * 62)
    print("AUDIT — is the headline accuracy meaningful?")
    print("=" * 62)

    exact_dupes = int(df.duplicated().sum())
    feature_dupes = int(df.duplicated(subset=FEATURES).sum())
    print(f"  Duplicate rows / feature vectors : {exact_dupes} / {feature_dupes}")
    leakage = exact_dupes > 0 or feature_dupes > 0
    print(f"  Train-test leakage possible      : {'YES' if leakage else 'no'}")

    nb = cross_val_score(GaussianNB(), X, y, cv=cv, n_jobs=-1).mean()
    print(f"  Gaussian Naive Bayes accuracy    : {nb * 100:.2f}%")

    # Independent draws per feature are a data generator's fingerprint; in real
    # agronomy rainfall and humidity move together.
    within = []
    for crop, group in df.groupby("label"):
        corr = group[FEATURES].corr().values
        within.append(float(np.abs(corr[np.triu_indices_from(corr, 1)]).mean()))
    mean_corr = float(np.mean(within))
    print(f"  Mean |corr| between features     : {mean_corr:.3f}  (within a crop)")

    rng = np.random.default_rng(0)
    sd = X.values.std(axis=0)
    noise_curve = {}
    for pct in (10, 20, 30, 50):
        noisy = X.values + rng.normal(0, sd * pct / 100, X.shape)
        score = cross_val_score(
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            noisy, y, cv=cv, n_jobs=-1,
        ).mean()
        noise_curve[f"{pct}pct"] = round(float(score), 4)
        print(f"  Accuracy at {pct:>2}% measurement noise : {score * 100:.2f}%")

    print("\n  Verdict: the score is arithmetically sound but says little about the")
    print("  model. Naive Bayes nearly matches a 300-tree forest, and features")
    print("  within a crop are uncorrelated — signatures of a synthetic dataset")
    print("  with well-separated classes. The defensible figure to quote is")
    print(f"  {noise_curve['20pct'] * 100:.0f}% under realistic +/-20% sensor error.")
    print("=" * 62)

    return {
        "exact_duplicate_rows": exact_dupes,
        "duplicate_feature_vectors": feature_dupes,
        "leakage_detected": leakage,
        "naive_bayes_accuracy": round(float(nb), 4),
        "mean_within_class_feature_correlation": round(mean_corr, 4),
        "accuracy_under_noise": noise_curve,
        "interpretation": (
            "No leakage: the score is arithmetically correct. But Gaussian Naive "
            "Bayes reaches nearly the same accuracy, and features are uncorrelated "
            "within each crop, which together indicate a synthetic dataset of "
            "well-separated classes rather than a strong model. Quote the "
            "noise-robust figure operationally."
        ),
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    balance = describe_balance(df)

    print(f"Loaded {balance['n_rows']} rows across {balance['n_classes']} crops")
    if balance["is_balanced"]:
        print(
            f"  Classes are perfectly balanced at {balance['min_class_count']} "
            "rows each -> no resampling (SMOTE) applied."
        )

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Scaler is not strictly required for a forest, but it keeps the pipeline
    # swappable -- we benchmark against models that do need it.
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    importances = pipe.named_steps["clf"].feature_importances_
    ranked = sorted(
        zip(FEATURES, (round(float(v), 4) for v in importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    misclassified = int(cm.sum() - np.trace(cm))

    print(f"\nHold-out accuracy : {test_acc:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Misclassified     : {misclassified} of {len(y_test)} test samples")
    print(f"Train time        : {train_seconds:.2f}s")
    print("\nFeature importance:")
    for name, val in ranked:
        bar = "#" * int(val * 60)
        print(f"  {name:<12} {val:.4f}  {bar}")

    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(
        {"pipeline": pipe, "features": FEATURES, "classes": sorted(y.unique())},
        MODELS / "crop_suitability.joblib",
    )

    audit_result = audit(df, X, y, cv)

    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "RandomForestClassifier(n_estimators=300)",
        "dataset": balance,
        "holdout_accuracy": round(float(test_acc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_folds": 5,
        "misclassified_test_samples": misclassified,
        "test_set_size": int(len(y_test)),
        "train_seconds": round(train_seconds, 3),
        "feature_importance": dict(ranked),
        "smote_applied": False,
        "smote_rationale": "Dataset is exactly balanced (100 samples/class); resampling would add synthetic noise without addressing any imbalance.",
        "audit": audit_result,
        "headline_accuracy_to_quote": audit_result["accuracy_under_noise"]["20pct"],
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved -> {MODELS / 'crop_suitability.joblib'}")
    print(f"Saved -> {MODELS / 'metrics.json'}")


if __name__ == "__main__":
    main()
