"""DT nested CV with kingdom-level taxonomic features excluded.

Critic concern: k__Bacteria (kingdom-level) captures library-size / total
bacterial load rather than disease signal. This script re-runs the DT nested
CV after dropping the two kingdom-only columns, and compares accuracy + top
features against the original run.

Kingdom-only definition: metaphlan columns whose taxonomic path has a k__
prefix but no phylum (p__) component — i.e. metaphlan_counts::k__Bacteria
and metaphlan_rel_ab::k__Bacteria.
"""

from __future__ import annotations

import ctypes
import json
import os
import warnings
from pathlib import Path

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
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "novel_species"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_SPLITS = 5
INNER_SPLITS = 3
TOP_N = 20
SEED = 42

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")


def is_kingdom_only(col: str) -> bool:
    """True if col is a metaphlan feature resolved only to kingdom level."""
    if not (col.startswith("metaphlan_counts::") or col.startswith("metaphlan_rel_ab::")):
        return False
    taxon = col.split("::")[-1]
    return "k__" in taxon and "p__" not in taxon


def load_data(drop_kingdom: bool = False):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    X = X.drop(columns=[c for c in X.columns if any(s in c.lower() for s in EXCLUDE_SUBSTRINGS)])
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns.tolist())
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
    X = X.fillna(X.median(numeric_only=True))
    if drop_kingdom:
        kingdom_cols = [c for c in X.columns if is_kingdom_only(c)]
        print(f"  Dropping {len(kingdom_cols)} kingdom-only column(s): {kingdom_cols}")
        X = X.drop(columns=kingdom_cols)
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
        "feature_importances": dict(zip(
            feats,
            model.feature_importances_.tolist(),
        )),
    }


METRIC_KEYS = ["accuracy", "roc_auc", "f1", "precision", "recall"]


def run_nested_cv(drop_kingdom: bool):
    label = "no_kingdom" if drop_kingdom else "original"
    print(f"\n{'='*60}")
    print(f"DT Nested CV — drop_kingdom={drop_kingdom}  ({label})")
    print("=" * 60)
    X, y = load_data(drop_kingdom=drop_kingdom)
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

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
    print(f"\n  MEAN accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  MEAN AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Aggregate feature importances across folds
    from collections import Counter, defaultdict
    imp_sum: dict[str, float] = defaultdict(float)
    count: Counter = Counter()
    for r in rows:
        for feat, imp in r["feature_importances"].items():
            imp_sum[feat] += imp
            count[feat] += 1
    # Mean importance across folds in which the feature appeared
    mean_imp = {f: imp_sum[f] / count[f] for f in imp_sum}
    imp_df = (
        pd.DataFrame({"feature": list(mean_imp.keys()), "mean_gini_across_folds": list(mean_imp.values())})
        .sort_values("mean_gini_across_folds", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["n_folds_selected"] = [count[f] for f in imp_df["feature"]]

    summary = {
        "config": {"drop_kingdom": drop_kingdom, "outer": OUTER_SPLITS, "inner": INNER_SPLITS, "top_n": TOP_N},
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "fold_accuracies": accs,
    }
    return summary, imp_df, rows


def main():
    orig_summary, orig_imp, _ = run_nested_cv(drop_kingdom=False)
    nokw_summary, nokw_imp, _ = run_nested_cv(drop_kingdom=True)

    print("\n" + "=" * 60)
    print("COMPARISON: DT with vs. without kingdom-level features")
    print("=" * 60)
    print(f"  Original  — acc: {orig_summary['accuracy_mean']:.4f} ± {orig_summary['accuracy_std']:.4f}  "
          f"AUC: {orig_summary['auc_mean']:.4f}")
    print(f"  No-kingdom— acc: {nokw_summary['accuracy_mean']:.4f} ± {nokw_summary['accuracy_std']:.4f}  "
          f"AUC: {nokw_summary['auc_mean']:.4f}")
    delta = nokw_summary["accuracy_mean"] - orig_summary["accuracy_mean"]
    print(f"  Δ accuracy (no-kingdom − original): {delta:+.4f}")

    orig_imp.to_csv(ARTIFACT_DIR / "dt_original_fold_importances.csv", index=False)
    nokw_imp.to_csv(ARTIFACT_DIR / "dt_no_kingdom_fold_importances.csv", index=False)

    comparison = {
        "original": orig_summary,
        "no_kingdom": nokw_summary,
        "delta_accuracy": delta,
    }
    with open(ARTIFACT_DIR / "dt_kingdom_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nTop-10 features without kingdom features:")
    for i, row in nokw_imp.head(10).iterrows():
        print(f"  {i+1:>2}. {row['feature'][:80]:<80}  {row['mean_gini_across_folds']:.4f}  "
              f"({int(row['n_folds_selected'])}/{OUTER_SPLITS} folds)")

    print(f"\nArtifacts written to {ARTIFACT_DIR}/")
    print("  - dt_original_fold_importances.csv")
    print("  - dt_no_kingdom_fold_importances.csv")
    print("  - dt_kingdom_comparison.json")


if __name__ == "__main__":
    main()
