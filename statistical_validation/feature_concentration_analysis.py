"""DT feature concentration analysis and ablation of PWY-7199.

Critic concern: PWY-7199 accounts for ~26-27% of total Gini importance in
the DT (≈6× the average feature weight), making the DT effectively a
thresholded single-feature classifier. This script:

  1. Visualises the Gini importance distribution (concentration curve) for
     the canonical DT run to quantify the dominance.
  2. Re-runs DT nested CV with PWY-7199 excluded from the candidate pool to
     measure how much accuracy depends on that one feature.
  3. Prints a comparison table of accuracy / AUC with vs. without PWY-7199.
"""

from __future__ import annotations

import ctypes
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_OMP_LIB = PROJECT_ROOT / "libomp_macos_arm64" / "libomp.dylib"
if LOCAL_OMP_LIB.exists():
    try:
        ctypes.CDLL(str(LOCAL_OMP_LIB))
    except OSError:
        pass

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

warnings.filterwarnings("ignore")

DATA_PATH = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv"
TARGET_COLUMN = "Case_status=PD"
CANONICAL_IMP_PATH = PROJECT_ROOT / "results" / "classification" / "feature_importance_dt.csv"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "novel_species"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_SPLITS = 5
INNER_SPLITS = 3
TOP_N = 20
SEED = 42
PWY7199_SUBSTRING = "PWY-7199"

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")


def load_data(exclude_pwy7199: bool = False):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    X = X.drop(columns=[c for c in X.columns if any(s in c.lower() for s in EXCLUDE_SUBSTRINGS)])
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns.tolist())
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
    X = X.fillna(X.median(numeric_only=True))
    if exclude_pwy7199:
        pwy_cols = [c for c in X.columns if PWY7199_SUBSTRING in c]
        print(f"  Dropping {len(pwy_cols)} PWY-7199 column(s): {pwy_cols}")
        X = X.drop(columns=pwy_cols)
    return X, y


def variance_filter(X_tr):
    vf = VarianceThreshold(threshold=0.0001)
    Xf = vf.fit_transform(X_tr)
    feats = np.array(X_tr.columns)[vf.get_support()]
    return Xf, feats


def select_top_dt(X_tr, y_tr, n=TOP_N):
    Xf, feats = variance_filter(X_tr)
    m = DecisionTreeClassifier(random_state=SEED, class_weight="balanced", criterion="gini")
    m.fit(Xf, y_tr)
    return feats[np.argsort(m.feature_importances_)[::-1][:n]].tolist()


def run_dt_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_dt(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    model = DecisionTreeClassifier(random_state=SEED, class_weight="balanced", criterion="gini")
    inner = cross_val_score(model, Xtr, y_tr, cv=inner_cv, scoring="accuracy", n_jobs=1)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "f1": f1_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "inner_cv_mean": float(inner.mean()),
        "top_features": feats,
    }


def run_nested_cv(exclude_pwy7199: bool):
    label = "no_PWY7199" if exclude_pwy7199 else "original"
    print(f"\n{'='*60}")
    print(f"DT Nested CV — exclude_PWY7199={exclude_pwy7199}  ({label})")
    print("=" * 60)
    X, y = load_data(exclude_pwy7199=exclude_pwy7199)
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold_idx, (tr, te) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold_idx)
        r = run_dt_fold(X_tr, X_te, y_tr, y_te, inner)
        r["fold"] = fold_idx
        rows.append(r)
        print(f"  Fold {fold_idx}: acc={r['accuracy']:.4f}  auc={r['roc_auc']:.4f}")

    accs = [r["accuracy"] for r in rows]
    aucs = [r["roc_auc"] for r in rows]
    print(f"  MEAN acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  AUC: {np.mean(aucs):.4f}")
    return {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "fold_accuracies": accs}


def concentration_analysis():
    """Quantify how concentrated Gini importance is around PWY-7199."""
    imp = pd.read_csv(CANONICAL_IMP_PATH)
    imp = imp[imp["importance"] > 0].sort_values("importance", ascending=False).reset_index(drop=True)
    n = len(imp)
    total = imp["importance"].sum()
    top1_pct = imp.loc[0, "importance"] / total * 100
    top5_pct = imp.head(5)["importance"].sum() / total * 100
    avg_imp = total / n
    top1_ratio = imp.loc[0, "importance"] / avg_imp

    print(f"\n{'='*60}")
    print("DT Gini Importance Concentration Analysis")
    print("=" * 60)
    print(f"  Features with non-zero importance : {n}")
    print(f"  Total Gini sum                    : {total:.4f}")
    print(f"  Top-1 feature (PWY-7199) share    : {top1_pct:.1f}%")
    print(f"  Top-5 features share              : {top5_pct:.1f}%")
    print(f"  Average per-feature importance    : {avg_imp:.4f}")
    print(f"  PWY-7199 / average ratio          : {top1_ratio:.1f}×")

    # Cumulative importance chart
    cumulative = np.cumsum(imp["importance"].values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, n + 1), cumulative / total * 100, linewidth=1.5)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax.axvline(1, color="red", linestyle=":", linewidth=0.8, label="Rank 1 (PWY-7199)")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Cumulative Gini importance (%)")
    ax.set_title("DT Gini importance concentration — PWY-7199 dominance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_fig = ARTIFACT_DIR / "dt_gini_concentration_curve.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n  Concentration curve saved: {out_fig}")

    return {"n_nonzero_features": n, "top1_pct": top1_pct, "top5_pct": top5_pct,
            "avg_importance": float(avg_imp), "top1_vs_avg_ratio": float(top1_ratio)}


def main():
    conc = concentration_analysis()
    orig = run_nested_cv(exclude_pwy7199=False)
    nopwy = run_nested_cv(exclude_pwy7199=True)

    delta = nopwy["accuracy_mean"] - orig["accuracy_mean"]
    print(f"\n{'='*60}")
    print("COMPARISON: DT with vs. without PWY-7199")
    print("=" * 60)
    print(f"  Original    — acc: {orig['accuracy_mean']:.4f} ± {orig['accuracy_std']:.4f}  "
          f"AUC: {orig['auc_mean']:.4f}")
    print(f"  No PWY-7199 — acc: {nopwy['accuracy_mean']:.4f} ± {nopwy['accuracy_std']:.4f}  "
          f"AUC: {nopwy['auc_mean']:.4f}")
    print(f"  Δ accuracy  : {delta:+.4f}")
    print(f"\n  PWY-7199 accounts for {conc['top1_pct']:.1f}% of DT Gini importance "
          f"({conc['top1_vs_avg_ratio']:.1f}× average feature weight).")
    if abs(delta) < 0.02:
        print("  NOTE: Accuracy drop is small — other features partially compensate.")
    else:
        print(f"  NOTE: Accuracy drops by {abs(delta)*100:.1f}pp without PWY-7199.")

    result = {
        "concentration": conc,
        "original_cv": orig,
        "no_pwy7199_cv": nopwy,
        "delta_accuracy": delta,
    }
    with open(ARTIFACT_DIR / "dt_pwy7199_dominance.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifacts written to {ARTIFACT_DIR}/")
    print("  - dt_pwy7199_dominance.json")
    print("  - dt_gini_concentration_curve.png")


if __name__ == "__main__":
    main()
