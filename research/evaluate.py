"""Reproducible classifier, ranking and sensitivity experiments for Cerealia."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest, kendalltau
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OUT = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(BACKEND))
import train as production_train  # noqa: E402
from app import recommender  # noqa: E402

FEATURES = production_train.FEATURES
NOISES = [0, 10, 20, 30, 50]
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["Random Forest (no augmentation)", "Random Forest (augmentation)",
          "Extra Trees (no augmentation)", "Extra Trees (augmentation)",
          "HistGradientBoosting (augmentation)", "Gaussian Naive Bayes"]
PRIMARY = {
    "Fitness only": {"fitness": 1.0, "profit": 0.0, "water": 0.0, "risk": 0.0},
    "Return term only": {"fitness": 0.0, "profit": 1.0, "water": 0.0, "risk": 0.0},
    "Fitness + return": {"fitness": .45, "profit": .30, "water": 0.0, "risk": 0.0},
    "Fitness + return + water": {"fitness": .45, "profit": .30, "water": .15, "risk": 0.0},
    "Full Cerealia": dict(recommender.DEFAULT_WEIGHTS),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def legacy_reproduction(df: pd.DataFrame, original_sha: str) -> tuple[dict, Pipeline]:
    """Re-run the legacy holdout and selected benchmark as implemented in train.py."""
    X, y = df[FEATURES], df.label
    sd_global = X.to_numpy().std(axis=0)
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    x_aug, y_aug = production_train.augment(xtr.to_numpy(), ytr.to_numpy(), sd_global,
                                           np.random.default_rng(42))
    pipe = production_train.make_extra_trees()
    pipe.fit(pd.DataFrame(x_aug, columns=FEATURES), y_aug)
    pred = pipe.predict(xte)
    noisy = xte.to_numpy() + np.random.default_rng(7).normal(0, sd_global * .2, xte.shape)
    noisy_pred = pipe.predict(pd.DataFrame(noisy, columns=FEATURES))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    curve = production_train.noise_curve(production_train.make_extra_trees, X, y, cv, sd_global, True)
    # Original train.py's reported cv_accuracy_mean is clean, unaugmented CV on selected factory.
    from sklearn.model_selection import cross_val_score
    clean_cv = cross_val_score(production_train.make_extra_trees(), X, y, cv=cv,
                               scoring="accuracy", n_jobs=production_train.N_JOBS)
    labels = sorted(y.unique())
    predictions = pd.DataFrame({"sample_id": xte.index, "split_seed": 42,
                                "noise_pct": 0, "y_true": yte.to_numpy(), "y_pred": pred})
    predictions.to_csv(OUT / "legacy_holdout_predictions.csv", index=False)
    records = []
    legacy_predictions = []
    for pct in NOISES:
        if pct == 0:
            test_x = xte.to_numpy()
        else:
            noise_seed = 7 if pct == 20 else 700 + pct
            test_x = xte.to_numpy() + np.random.default_rng(noise_seed).normal(
                0, sd_global * pct / 100, xte.shape)
        yp = pipe.predict(pd.DataFrame(test_x, columns=FEATURES))
        for sample_id, truth, guess in zip(xte.index, yte, yp):
            legacy_predictions.append({"sample_id": sample_id, "split_seed": 42, "noise_pct": pct,
                            "y_true": truth, "y_pred": guess})
    pd.DataFrame(legacy_predictions).to_csv(OUT / "legacy_predictions.csv", index=False)
    return ({"protocol": "legacy_repository", "dataset_sha256": original_sha,
             "holdout_accuracy": float(accuracy_score(yte, pred)),
             "holdout_macro_f1": float(f1_score(yte, pred, average="macro")),
             "holdout_accuracy_20pct_noise": float(accuracy_score(yte, noisy_pred)),
             "five_fold_benchmark_accuracy_20pct_noise": float(curve[20]),
             "five_fold_clean_accuracy_mean": float(clean_cv.mean()),
             "five_fold_clean_accuracy_sd_population": float(clean_cv.std()),
             "global_sd_used_before_split": sd_global.tolist(),
             "augmentation": {"copies": production_train.NOISE_COPIES,
                              "pct_range": production_train.AUG_NOISE_RANGE,
                              "seed": 42},
             "holdout_seed": 42, "test_noise_seed": 7,
             "selection_benchmark_scope": "all rows before final holdout; not independent of model selection"}, pipe)


def factory(name: str, seed: int):
    if name.startswith("Random Forest"):
        return production_train.make_forest()
    if name.startswith("Extra Trees"):
        return production_train.make_extra_trees()
    if name.startswith("HistGradientBoosting"):
        return production_train.make_boosting()
    return Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())])


def classifier_experiments(df: pd.DataFrame):
    xall, yall = df[FEATURES].to_numpy(), df.label.to_numpy()
    labels = sorted(np.unique(yall))
    rows, pred_rows = [], []
    for seed in SEEDS:
        tr, te = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(xall, yall))
        xtr, xte, ytr, yte = xall[tr], xall[te], yall[tr], yall[te]
        # retain ids from the canonical source file; same observations for all models.
        ids = df.index.to_numpy()[te]
        sd = xtr.std(axis=0)
        for name in MODELS:
            aug = name.endswith("(augmentation)")
            model = factory(name, seed)
            if aug:
                xa, ya = production_train.augment(xtr, ytr, sd,
                                                  np.random.default_rng(seed))
            else:
                xa, ya = xtr, ytr
            model.fit(pd.DataFrame(xa, columns=FEATURES), ya)
            for pct in NOISES:
                # Same noise realization for every contender, each split and level.
                rng = np.random.default_rng(90000 + seed * 100 + pct)
                xt = xte.copy() if pct == 0 else xte + rng.normal(0, sd * pct / 100, xte.shape)
                yp = model.predict(pd.DataFrame(xt, columns=FEATURES))
                acc = accuracy_score(yte, yp)
                mf1 = f1_score(yte, yp, labels=labels, average="macro", zero_division=0)
                rows.append({"model": name, "seed": seed, "noise_pct": pct,
                             "accuracy": acc, "macro_f1": mf1,
                             "accuracy_wilson_low": wilson(int((yp == yte).sum()), len(yte))[0],
                             "accuracy_wilson_high": wilson(int((yp == yte).sum()), len(yte))[1]})
                for idx, (sid, yt, yh) in enumerate(zip(ids, yte, yp)):
                    entry = {"sample_id": int(sid), "seed": seed, "model": name,
                             "noise_pct": pct, "y_true": yt, "y_pred": yh}
                    entry.update({f"original_{feature}": float(xte[idx, j])
                                  for j, feature in enumerate(FEATURES)})
                    pred_rows.append(entry)
    runs = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    runs.to_csv(OUT / "classifier_runs.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    summary = runs.groupby(["model", "noise_pct"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), sample_sd_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"), sample_sd_macro_f1=("macro_f1", "std"))
    summary.to_csv(OUT / "classifier_summary.csv", index=False)
    report_rows, cm_paths = [], {}
    focus = preds[(preds.seed == 42) & (preds.model == MODELS[3])]
    cm_long = []
    for pct in [0, 20, 50]:
        sub = focus[focus.noise_pct == pct]
        cm = confusion_matrix(sub.y_true, sub.y_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).rename_axis("true_label").to_csv(
            OUT / f"confusion_matrix_{pct}pct.csv")
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(norm, index=labels, columns=labels).rename_axis("true_label").to_csv(
            OUT / f"confusion_matrix_{pct}pct_normalized.csv")
        cm_paths[pct] = cm
        yp, yt = sub.y_pred.to_numpy(), sub.y_true.to_numpy()
        rep = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
        for crop in labels:
            v = rep[crop]
            report_rows.append({"noise_pct": pct, "class": crop, "precision": v["precision"],
                                "recall": v["recall"], "f1": v["f1-score"], "support": v["support"]})
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                cm_long.append({"noise_pct": pct, "true_label": a, "predicted_label": b,
                                "count": int(cm[i, j]), "row_fraction": float(norm[i, j])})
    pd.DataFrame(report_rows).to_csv(OUT / "classification_report.csv", index=False)
    pd.DataFrame(cm_long).to_csv(OUT / "confusion_matrices_long.csv", index=False)
    confusion_summary = []
    pairs = {}
    for pct, cm in cm_paths.items():
        sub = focus[focus.noise_pct == pct]
        confusion_summary.append({"noise_pct": pct, "samples": int(cm.sum()),
                                  "correct": int(np.trace(cm)), "incorrect": int(cm.sum()-np.trace(cm)),
                                  "accuracy": float(np.trace(cm)/cm.sum()),
                                  "macro_precision": float(precision_score(sub.y_true, sub.y_pred, labels=labels, average="macro", zero_division=0)),
                                  "macro_recall": float(recall_score(sub.y_true, sub.y_pred, labels=labels, average="macro", zero_division=0)),
                                  "macro_f1": float(f1_score(sub.y_true, sub.y_pred, labels=labels, average="macro", zero_division=0)),
                                  "weighted_f1": float(f1_score(sub.y_true, sub.y_pred, labels=labels, average="weighted", zero_division=0))})
        pair_counts = Counter()
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                pair_counts[tuple(sorted((labels[i], labels[j])))] += int(cm[i,j] + cm[j,i])
        pairs[pct] = [{"crop_a": p[0], "crop_b": p[1], "count": n}
                      for p, n in pair_counts.most_common(5)]
    pd.DataFrame(confusion_summary).to_csv(OUT / "confusion_summary.csv", index=False)
    return runs, preds, summary, confusion_summary, pairs, labels


def wilson(k: int, n: int, z=1.959963984540054):
    p = k/n if n else 0
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h


def paired_tests(preds: pd.DataFrame):
    bname, aname = MODELS[2], MODELS[3]
    rows = []
    for pct in NOISES:
        sub = preds[(preds.seed == 42) & (preds.noise_pct == pct)]
        wide = sub.pivot(index="sample_id", columns="model", values=["y_true", "y_pred"])
        truth = wide[("y_true", bname)].to_numpy()
        base = wide[("y_pred", bname)].to_numpy() == truth
        aug = wide[("y_pred", aname)].to_numpy() == truth
        both = int((base & aug).sum()); onlybase = int((base & ~aug).sum())
        onlyaug = int((~base & aug).sum()); neither = int((~base & ~aug).sum())
        pval = float(binomtest(onlyaug, onlyaug+onlybase, .5, alternative="two-sided").pvalue) if onlyaug+onlybase else 1.0
        rows.append({"noise_pct": pct, "n": len(truth), "both_correct": both,
                     "baseline_only_correct": onlybase, "augmented_only_correct": onlyaug,
                     "both_incorrect": neither, "baseline_accuracy": float(base.mean()),
                     "augmented_accuracy": float(aug.mean()),
                     "accuracy_difference_pp": float((aug.mean()-base.mean())*100), "raw_p": pval})
    order = sorted(range(len(rows)), key=lambda i: rows[i]["raw_p"])
    running = 0.0
    adj = [1.0] * len(rows)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(rows)-rank)*rows[idx]["raw_p"]))
        adj[idx] = running
    for i, value in enumerate(adj): rows[i]["holm_adjusted_p"] = value
    pd.DataFrame(rows).to_csv(OUT / "paired_tests.csv", index=False)
    return rows


def rank_record(state, cfg, result, variant, multiplier=None):
    recs = result["recommendations"]
    top = recs[0] if recs else {}
    return {"state_id": state["id"], "state_name": state["name"], "configuration": cfg,
            "variant": variant, "multiplier": multiplier,
            "top1": top.get("crop"), "top3": "|".join(r["crop"] for r in recs[:3]),
            "top6": "|".join(r["crop"] for r in recs[:6]),
            "fitness": top.get("fitness"), "agro_fitness": top.get("agro_fit_pct"),
            "regional_prior": top.get("regional_prior"), "score": top.get("score"),
            "annual_net_profit_inr_ha": top.get("economics",{}).get("net_profit_per_ha_year"),
            "irrigation_gap_mm": top.get("water",{}).get("irrigation_gap_mm"),
            "irrigation_dependence": top.get("water",{}).get("dependence_ratio"),
            "production_risk": top.get("risk",{}).get("production_risk"),
            "market_risk": top.get("risk",{}).get("market_risk"),
            "combined_risk": top.get("risk",{}).get("risk"),
            "historical_area_share": top.get("area_share"),
            "yield_source": top.get("economics",{}).get("yield_source"),
            "risk_source": top.get("risk",{}).get("source"),
            "eligible_candidates": result["gate"]["crops_considered"],
            "fitness_gate_fallback": result["gate"]["excluded_poor_agro_fit"] > 0 and result["gate"]["crops_considered"] < recommender.MIN_RESULTS,
            "candidate_count": len(result["all_scores"])}


def ranking_experiments(pipe):
    recommender._bundle = lambda: {"pipeline": pipe}
    for fn in ["_feature_importance", "_crop_envelopes", "_economics", "_states", "_calendar", "_crop_stats"]:
        getattr(recommender, fn).cache_clear()
    states = recommender.list_states()
    assert len(states) == 36, f"expected 36 state/UT profiles, got {len(states)}"
    rows = []
    candidate_rows = []
    all_results = {}
    for state in states:
        state_id = state["id"]
        all_results[state_id] = {}
        for config, weights in PRIMARY.items():
            # Production function has default production weights: use actual configurable recommend.
            site = recommender.SiteConditions(**{f: float(state[f]) for f in FEATURES if f != "rainfall"},
                 rainfall=float(state["rainfall_annual_mm"])/12,
                 rainfall_annual_mm=float(state["rainfall_annual_mm"]))
            res = recommender.recommend(site, top_n=22, weights=weights, state_id=state_id)
            rows.append(rank_record(state, config, res, "primary"))
            for rank, crop in enumerate(res["recommendations"], 1):
                candidate_rows.append({"state_id":state_id,"state_name":state["name"],
                    "configuration":config,"rank":rank,"candidate_crop":crop["crop"],
                    "fitness":crop["fitness"],"agro_climatic_fitness":crop["agro_fit_pct"],
                    "regional_prior":crop["regional_prior"],"final_score":crop["score"],
                    "annual_net_profit_inr_ha":crop["economics"]["net_profit_per_ha_year"],
                    "estimated_irrigation_gap_mm":crop["water"]["irrigation_gap_mm"],
                    "irrigation_dependence":crop["water"]["dependence_ratio"],
                    "production_risk":crop["risk"]["production_risk"],
                    "market_risk":crop["risk"]["market_risk"],
                    "combined_risk":crop["risk"]["risk"],
                    "historical_area_share":crop["area_share"],
                    "yield_source":crop["economics"]["yield_source"],
                    "risk_source":crop["risk"]["source"],
                    "eligible_candidates":res["gate"]["crops_considered"],
                    "fitness_gate_fallback":res["gate"]["excluded_poor_agro_fit"] > 0 and res["gate"]["crops_considered"] < recommender.MIN_RESULTS})
            all_results[state_id][config] = res
        extras = [
            ("Raw estimated net-profit ranking", {}, True, True, "raw_profit"),
            ("Classifier probability only", {}, True, True, "classifier_probability"),
            ("Full without regional prior", recommender.DEFAULT_WEIGHTS, False, True, "weighted"),
            ("Full without fitness gate", recommender.DEFAULT_WEIGHTS, True, False, "weighted"),
        ]
        for cfg, weights, prior, gate, method in extras:
            res = recommender.recommend(site, top_n=22, weights=weights, state_id=state_id,
                                        use_regional_prior=prior, apply_fitness_gate=gate,
                                        ranking_method=method)
            rows.append(rank_record(state, cfg, res, "additional"))
            all_results[state_id][cfg] = res
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "ranking_ablation_by_region.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(OUT / "ranking_ablation_candidates.csv", index=False)
    summary = raw.groupby(["configuration", "variant"], as_index=False).agg(
        regions=("state_id", "nunique"), mean_fitness=("fitness", "mean"),
        mean_estimated_profit=("annual_net_profit_inr_ha", "mean"),
        mean_water_dependence=("irrigation_dependence", "mean"),
        mean_risk=("combined_risk", "mean"),
        mean_historical_area_share=("historical_area_share", "mean"),
        regions_missing_historical_area_share=("historical_area_share", lambda s: int(s.isna().sum())),
        regions_using_curated_yield=("yield_source", lambda s: int((s == "curated estimate").sum())),
        mean_eligible_candidates=("eligible_candidates", "mean"))
    full = summary[(summary.configuration == "Full Cerealia") & (summary.variant == "primary")].iloc[0]
    for metric in ["mean_fitness", "mean_estimated_profit", "mean_water_dependence", "mean_risk",
                   "mean_historical_area_share", "mean_eligible_candidates"]:
        summary[f"{metric}_delta_vs_full"] = summary[metric] - full[metric]
        summary[f"{metric}_relative_change_vs_full_pct"] = np.where(
            full[metric] != 0, (summary[metric] - full[metric]) / abs(full[metric]) * 100, np.nan)
    summary.to_csv(OUT / "ranking_ablation_summary.csv", index=False)

    weight_rows = []
    for weight in ["fitness", "profit", "water", "risk"]:
        for mult in [.5, .75, 1, 1.25, 1.5, 2]:
            changed = 0; overlaps3=[]; overlaps6=[]; taus=[]; profit=[]; water=[]; risk=[]
            for state in states:
                sid=state["id"]; base=all_results[sid]["Full Cerealia"]
                w=dict(recommender.DEFAULT_WEIGHTS); w[weight] *= mult
                site = recommender.SiteConditions(**{f: float(state[f]) for f in FEATURES if f != "rainfall"},
                     rainfall=float(state["rainfall_annual_mm"])/12,
                     rainfall_annual_mm=float(state["rainfall_annual_mm"]))
                cur=recommender.recommend(site, top_n=22, weights=w, state_id=sid)
                br=[x["crop"] for x in base["recommendations"]]; cr=[x["crop"] for x in cur["recommendations"]]
                changed += bool(br and cr and br[0]!=cr[0])
                overlaps3.append(len(set(br[:3])&set(cr[:3]))/max(1,min(3,len(br))))
                overlaps6.append(len(set(br[:6])&set(cr[:6]))/max(1,min(6,len(br))))
                btop=base["recommendations"][0]; ctop=cur["recommendations"][0]
                profit.append(ctop["economics"]["net_profit_per_ha_year"]-btop["economics"]["net_profit_per_ha_year"])
                water.append(ctop["water"]["dependence_ratio"]-btop["water"]["dependence_ratio"])
                risk.append(ctop["risk"]["risk"]-btop["risk"]["risk"])
                bmap={x["crop"]:i for i,x in enumerate(base["recommendations"])}
                cmap={x["crop"]:i for i,x in enumerate(cur["recommendations"])}
                common=sorted(set(bmap)&set(cmap))
                taus.append(float(kendalltau([bmap[x] for x in common],[cmap[x] for x in common]).statistic) if len(common)>1 else np.nan)
                region_row=rank_record(state,weight,cur,"weight_sensitivity",mult)
                region_row.update({"changed_top1":bool(br and cr and br[0]!=cr[0]),
                    "top3_overlap":overlaps3[-1],"top6_overlap":overlaps6[-1],
                    "kendall_tau":taus[-1],"profit_change":profit[-1],
                    "water_change":water[-1],"risk_change":risk[-1]})
                weight_rows.append(region_row)
            # aggregate row emitted below
            weight_rows.append({"state_id":"ALL","state_name":"All regions","configuration":weight,
                "variant":"weight_summary","multiplier":mult,"changed_top1":changed,
                "changed_top1_pct":changed/len(states)*100,"top3_overlap":np.nanmean(overlaps3),
                "top6_overlap":np.nanmean(overlaps6),"kendall_tau":np.nanmean(taus),
                "profit_change":np.nanmean(profit),"water_change":np.nanmean(water),
                "risk_change":np.nanmean(risk)})
    ws=pd.DataFrame(weight_rows); ws.to_csv(OUT / "weight_sensitivity.csv",index=False)

    # Perturb sensor/profile inputs using training population SD; regional/candidate rules fixed.
    df=pd.read_csv(BACKEND/"app/data/Crop_recommendation.csv")
    sd=df[FEATURES].std(ddof=0).to_numpy()
    stable=[]
    for pct in [5,10,20,30]:
        for state in states:
            sid=state["id"]
            vals=np.array([float(state[f]) for f in FEATURES if f!="rainfall"]+
                          [float(state["rainfall_annual_mm"])/12])
            base_site=recommender.SiteConditions(**dict(zip(FEATURES,vals)),
                         rainfall_annual_mm=float(state["rainfall_annual_mm"]))
            base=recommender.recommend(base_site,top_n=6,state_id=sid)
            for seed in SEEDS:
                rng=np.random.default_rng(300000+seed*100+pct)
                noisy=vals+rng.normal(0,sd*pct/100)
                site=recommender.SiteConditions(**dict(zip(FEATURES,noisy)),
                            rainfall_annual_mm=float(noisy[-1]*12))
                cur=recommender.recommend(site,top_n=6,state_id=sid)
                bt=[x["crop"] for x in base["recommendations"]]; ct=[x["crop"] for x in cur["recommendations"]]
                stable.append({"state_id":sid,"state_name":state["name"],"noise_pct_sd":pct,"seed":seed,
                    "base_top1":bt[0] if bt else None,"perturbed_top1":ct[0] if ct else None,
                    "top1_changed":bool(bt and ct and bt[0]!=ct[0]),
                    "top3_jaccard":len(set(bt[:3])&set(ct[:3]))/max(1,len(set(bt[:3])|set(ct[:3]))),
                    "top6_jaccard":len(set(bt)&set(ct))/max(1,len(set(bt)|set(ct)))})
    stable_df=pd.DataFrame(stable); stable_df.to_csv(OUT/"ranking_stability.csv",index=False)
    return raw, summary, ws[ws.variant=="weight_summary"], stable_df, all_results


def make_figures(summary, confusion_summary, labels, ranking_summary, weight_summary, stable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    figs=OUT/"figures"; figs.mkdir(parents=True,exist_ok=True)
    colors=["#176b87","#e07a35","#4f8a5b","#8b6bb1","#b04f63","#525f7f"]
    def save(fig,name):
        fig.tight_layout(); fig.savefig(figs/f"{name}.pdf",bbox_inches="tight"); fig.savefig(figs/f"{name}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for i,(name,g) in enumerate(summary.groupby("model")):
        ax.plot(g.noise_pct,g.mean_accuracy*100,marker="o",label=name,color=colors[i])
    ax.set(xlabel="Gaussian input noise (% of training feature SD)",ylabel="Accuracy (%)"); ax.set_xticks(NOISES); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2); save(fig,"classifier_noise_accuracy")
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for i,(name,g) in enumerate(summary.groupby("model")):
        means=g.mean_accuracy.to_numpy()*100; sds=g.sample_sd_accuracy.fillna(0).to_numpy()*100
        ax.errorbar(g.noise_pct,means,yerr=sds,marker="o",capsize=2,label=name,color=colors[i])
    ax.set(xlabel="Gaussian input noise (% of training feature SD)",ylabel="Accuracy (%)"); ax.set_xticks(NOISES); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2); save(fig,"classifier_repeated_split_variability")
    for pct in [0,20,50]:
        cm=pd.read_csv(OUT/f"confusion_matrix_{pct}pct.csv",index_col=0).to_numpy()
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=6); ax.set_yticks(range(len(labels)),labels,fontsize=6); ax.set_xlabel("Predicted crop"); ax.set_ylabel("True crop"); fig.colorbar(im,ax=ax,label="Samples"); save(fig,f"confusion_{pct}pct")
    report=pd.read_csv(OUT/"classification_report.csv")
    fig,ax=plt.subplots(figsize=(8,4.6))
    for pct in [0,20,50]:
        g=report[report.noise_pct==pct]; ax.plot(g["class"],g.recall*100,marker="o",label=f"{pct}%")
    ax.set(ylabel="Recall (%)",xlabel="Crop class"); ax.tick_params(axis="x",rotation=90,labelsize=7); ax.legend(title="Noise"); ax.grid(axis="y",alpha=.2); save(fig,"per_class_recall")
    r=ranking_summary[ranking_summary.variant=="primary"]
    fig,axs=plt.subplots(1,3,figsize=(10,3.4))
    for ax,col,label in zip(axs,["mean_estimated_profit","mean_water_dependence","mean_risk"],["Estimated net profit (INR/ha/year)","Irrigation dependence (ratio)","Combined risk (index)"]): ax.bar(range(len(r)),r[col],color=colors[:len(r)]); ax.set_xticks(range(len(r)),r.configuration,rotation=35,ha="right",fontsize=7); ax.set_ylabel(label)
    save(fig,"ranking_ablation_tradeoffs")
    pivot=weight_summary.pivot(index="configuration",columns="multiplier",values="changed_top1_pct")
    fig,ax=plt.subplots(figsize=(7,3.5)); im=ax.imshow(pivot,cmap=LinearSegmentedColormap.from_list("cerealia",["#edf4ef","#e07a35"]),aspect="auto",vmin=0,vmax=100); ax.set_xticks(range(len(pivot.columns)),[f"{x:g}×" for x in pivot.columns]); ax.set_yticks(range(len(pivot.index)),pivot.index); ax.set_xlabel("Coefficient multiplier"); ax.set_ylabel("Weight varied"); fig.colorbar(im,ax=ax,label="Regions with changed top crop (%)"); save(fig,"weight_sensitivity_heatmap")
    agg=stable.groupby("noise_pct_sd").top1_changed.mean()*100
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot(agg.index,agg.values,marker="o",color="#176b87"); ax.set(xlabel="Input perturbation (% of dataset feature SD)",ylabel="Regions with changed top crop (%)"); ax.set_xticks(agg.index); ax.set_ylim(0,100); ax.grid(alpha=.2); save(fig,"ranking_input_perturbation_stability")
    # 3 explicitly requested regions and 2 additional contextual cases.
    cases=pd.read_csv(OUT/"ranking_ablation_by_region.csv")
    wanted=["PB","MH","KL","RJ","LA"]
    fig,ax=plt.subplots(figsize=(7.2,4))
    c=cases[(cases.state_id.isin(wanted))&(cases.variant=="primary")]
    for i,sid in enumerate(wanted):
        g=c[c.state_id==sid]; ax.plot(g.configuration,g.annual_net_profit_inr_ha,marker="o",label=g.state_name.iloc[0] if len(g) else sid,color=colors[i])
    ax.set_ylabel("Top crop estimated net profit (INR/ha/year)"); ax.tick_params(axis="x",rotation=35,labelsize=7); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.2); save(fig,"regional_case_studies")


def main():
    global SEEDS
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    args=ap.parse_args()
    SEEDS=args.seeds
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    df=production_train.load_data(); data_path=production_train.DATA
    git_sha=__import__("subprocess").check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    legacy,pipe=legacy_reproduction(df,sha256(data_path))
    runs,preds,summary,cm_summary,pairs,labels=classifier_experiments(df)
    paired=paired_tests(preds)
    rank_raw,rank_summary,weight_summary,stable,all_res=ranking_experiments(pipe)
    make_figures(summary,cm_summary,labels,rank_summary,weight_summary,stable)
    legacy_result=json.loads((BACKEND/"models/metrics.json").read_text())
    ablation_focus=rank_summary[rank_summary.variant=="primary"].to_dict(orient="records")
    classifier_focus=summary.to_dict(orient="records")
    json_dump(OUT/"dashboard_summary.json",{"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+git_sha[:8],
      "legacy":{"clean_holdout_accuracy":legacy["holdout_accuracy"],
        "noise20_holdout_accuracy":legacy["holdout_accuracy_20pct_noise"],
        "noise20_cv_accuracy":legacy["five_fold_benchmark_accuracy_20pct_noise"]},
      "strict_classifier":classifier_focus,"ranking_ablation":ablation_focus})
    json_dump(OUT/"reproduction_metrics.json",{"legacy_reference_metrics":legacy_result,
        "legacy_reproduction":legacy,"strict_protocol":{
        "description":"5 stratified folds with fixed seeds; training-only feature SD; training-only augmentation; matched noisy test inputs across six models; original observations remain test units.",
        "seeds":SEEDS,"noise_pct":NOISES,"model_count":len(MODELS),"result_combinations":len(runs),
        "production_extra_trees_parameters":pipe.named_steps["clf"].get_params(),
        "class_order":labels}})
    case={}
    for name in ["Punjab","Maharashtra","Kerala","Rajasthan","Ladakh"]:
        state=next((s for s in recommender.list_states() if s["name"]==name),None)
        if state: case[name]={cfg:[{k:v for k,v in r.items() if k in ["crop","score","fitness","agro_fit_pct","regional_prior","economics","water","risk","area_share","evidence"]} for r in all_res[state["id"]][cfg]["recommendations"][:6]] for cfg in list(PRIMARY)+["Raw estimated net-profit ranking","Classifier probability only","Full without regional prior","Full without fitness gate"]}
    json_dump(OUT/"regional_case_studies.json",case)
    checks={"source_rows":len(df),"class_count":int(df.label.nunique()),
      "holdout_rows_each_seed":int(len(df)/5),"prediction_rows":len(preds),
      "expected_prediction_rows":len(SEEDS)*len(MODELS)*len(NOISES)*int(len(df)/5),
      "all_confusion_matrices_sum_440":all(x["samples"]==len(df)/5 for x in cm_summary),
      "regional_profiles":len(recommender.list_states()),"ranking_ablation_rows":len(rank_raw),
      "paired_tests":len(paired),"figures":len(list((OUT/"figures").glob("*.pdf")))}
    if checks["prediction_rows"]!=checks["expected_prediction_rows"]: raise AssertionError(checks)
    if not checks["all_confusion_matrices_sum_440"]: raise AssertionError(checks)
    pd.DataFrame([checks]).to_csv(OUT/"integrity_checks.csv",index=False)
    # Hash raw result artifacts for dashboard and subsequent verification.
    manifest={"run_id":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+git_sha[:8],
      "protocol_version":"cerealia-research-v1.0","git_commit_sha":git_sha,
      "working_tree_note":"SHA identifies the source commit used to run this evaluation.",
      "dataset_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in [data_path,*sorted((BACKEND/"app/data").glob("*"))] if p.is_file()},
      "model_hyperparameters":{"extra_trees":pipe.named_steps["clf"].get_params(),
          "random_forest":production_train.make_forest().named_steps["clf"].get_params(),
          "hist_gradient_boosting":production_train.make_boosting().named_steps["clf"].get_params(),
          "gaussian_nb":"sklearn GaussianNB defaults","augmentation":{"copies":production_train.NOISE_COPIES,"range_pct":production_train.AUG_NOISE_RANGE}},
      "dependencies":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"scipy":scipy.__version__},
      "feature_order":FEATURES,"preprocessing":"StandardScaler fitted within each model pipeline on training data only in the strict protocol; legacy run follows train.py full-dataset noise-scale behavior.",
      "dataset_provenance":"Packaged Crop_recommendation.csv; repository does not contain a verified primary provenance citation. Regional agriculture data is packaged with its source description in backend/build_crop_stats.py.",
      "seeds":SEEDS,"noise_levels_pct":NOISES,"models":MODELS,
      "execution":{"started_at_utc":datetime.fromtimestamp(t0,timezone.utc).isoformat(),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"runtime_seconds":round(time.time()-t0,2),"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),"cpu_count":__import__("os").cpu_count()},
      "checks":checks,"limitations":["No untouched external crop dataset available in repository.","Legacy candidate selection benchmark used all source rows before final holdout was defined.","Regional prior is derived from historical cultivation records and is not independent validation.","Profitability, irrigation gap and risk are modeled estimates, not realized outcomes or measured water savings."]}
    manifest["artifact_files_sha256"] = {
        str(p.relative_to(OUT)): sha256(p)
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "manifest.sha256"}
    }
    json_dump(OUT/"manifest.json",manifest)
    (OUT/"manifest.sha256").write_text(f"{sha256(OUT/'manifest.json')}  manifest.json\n")
    print(json.dumps({"manifest":str(OUT/"manifest.json"),"legacy":legacy,"checks":checks,"runtime":manifest["execution"]["runtime_seconds"]},indent=2))

if __name__=="__main__": main()
