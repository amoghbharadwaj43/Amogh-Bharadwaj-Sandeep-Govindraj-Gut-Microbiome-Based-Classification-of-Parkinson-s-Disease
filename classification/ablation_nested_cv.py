"""Nested-CV clinical-feature ablation for DT, RF, and XGBoost.

Mirrors grade_all_models_nested_cv.py exactly (same outer/inner seed scheme,
same Optuna search space for XGBoost, same per-fold top-N feature reselection
and class-weighting), but runs each model under three feature-pool conditions:

  Full           : microbiome + all clinical features
  No-Constipation: microbiome + clinical except any column containing 'Constipation'
  Microbiome-Only: only columns prefixed with metaphlan_ or humann_

The pool restriction is applied BEFORE per-fold variance filtering and top-N
selection, so each condition gets a fair top-20 budget drawn from its allowed
candidate pool. Outer fold splits are identical across all 9 (model, condition)
combinations because StratifiedKFold(seed=42) is fed the same y. This means
Full-condition results are directly comparable to the existing Table II numbers.
"""

from __future__ import annotations

import ctypes
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv"
TARGET_COLUMN = "Case_status=PD"
OUT_DIR = PROJECT_ROOT / "results" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_SPLITS = 5
INNER_SPLITS = 3
TOP_N = 20
OPTUNA_TRIALS = 20
SEED = 42

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")
MICRO_PREFIXES = ("metaphlan_counts::", "metaphlan_rel_ab::",
                  "humann_KO_group_counts::", "humann_pathway_counts::")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    X = X.drop(columns=[c for c in X.columns if any(s in c.lower() for s in EXCLUDE_SUBSTRINGS)])
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns.tolist())
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def restrict_pool(X: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition == "Full":
        return X
    if condition == "No-Constipation":
        drop = [c for c in X.columns if "constipation" in c.lower()]
        return X.drop(columns=drop)
    if condition == "Microbiome-Only":
        keep = [c for c in X.columns if c.startswith(MICRO_PREFIXES)]
        return X[keep]
    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Per-model selection + tuning (copied verbatim from grade_all_models_nested_cv.py)
# ---------------------------------------------------------------------------

def xgb_base_params(spw):
    return dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.6, reg_lambda=1.0,
        reg_alpha=0.0, gamma=0.0, min_child_weight=1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=-1, scale_pos_weight=spw,
    )


def _variance_filter(X_tr):
    vf = VarianceThreshold(threshold=0.0001)
    X_f = vf.fit_transform(X_tr)
    feats = np.array(X_tr.columns)[vf.get_support()]
    return X_f, feats


def select_top_xgb(X_tr, y_tr, n=TOP_N):
    X_f, feats = _variance_filter(X_tr)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = XGBClassifier(**xgb_base_params(spw))
    m.fit(X_f, y_tr)
    shap_vals = shap.TreeExplainer(m).shap_values(X_f)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    imp = np.abs(shap_vals).mean(axis=0)
    return feats[np.argsort(imp)[::-1][:n]].tolist()


def select_top_rf(X_tr, y_tr, n=TOP_N):
    X_f, feats = _variance_filter(X_tr)
    m = RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced",
    )
    m.fit(X_f, y_tr)
    return feats[np.argsort(m.feature_importances_)[::-1][:n]].tolist()


def select_top_dt(X_tr, y_tr, n=TOP_N):
    X_f, feats = _variance_filter(X_tr)
    m = DecisionTreeClassifier(random_state=SEED, class_weight="balanced", criterion="gini")
    m.fit(X_f, y_tr)
    return feats[np.argsort(m.feature_importances_)[::-1][:n]].tolist()


def tune_xgb(X_tr, y_tr, inner_cv, spw):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": SEED, "n_jobs": -1, "scale_pos_weight": spw,
        }
        scores = cross_val_score(
            XGBClassifier(**params), X_tr, y_tr,
            cv=inner_cv, scoring="accuracy", n_jobs=1,
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


def run_xgb_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_xgb(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    best_params = tune_xgb(Xtr, y_tr, inner_cv, spw)
    params = xgb_base_params(spw)
    params.update(best_params)
    model = XGBClassifier(**params)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return accuracy_score(y_te, pred), roc_auc_score(y_te, proba)


def run_rf_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_rf(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    model = RandomForestClassifier(
        n_estimators=400, random_state=SEED, n_jobs=-1,
        class_weight="balanced", max_depth=None,
    )
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return accuracy_score(y_te, pred), roc_auc_score(y_te, proba)


def run_dt_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_dt(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    model = DecisionTreeClassifier(
        random_state=SEED, class_weight="balanced", criterion="gini",
    )
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return accuracy_score(y_te, pred), roc_auc_score(y_te, proba)


RUNNERS = {
    "Decision Tree": run_dt_fold,
    "Random Forest": run_rf_fold,
    "XGBoost": run_xgb_fold,
}
CONDITIONS = ["Full", "No-Constipation", "Microbiome-Only"]


def main():
    X_full, y = load_data()
    print(f"Data: {X_full.shape[0]} samples, {X_full.shape[1]} features after load_data")
    print(f"Class balance: {y.value_counts().to_dict()}")

    pools = {cond: restrict_pool(X_full, cond) for cond in CONDITIONS}
    for cond, Xc in pools.items():
        print(f"  {cond:>16}: {Xc.shape[1]} features in candidate pool")

    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    splits = list(outer.split(X_full, y))

    rows = []
    t0 = time.time()
    for cond in CONDITIONS:
        Xc = pools[cond]
        print(f"\n========== Condition: {cond} ==========")
        for model_name, runner in RUNNERS.items():
            print(f"  --- {model_name} ---")
            for fold_idx, (tr, te) in enumerate(splits, 1):
                X_tr, X_te = Xc.iloc[tr], Xc.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]
                inner = StratifiedKFold(
                    n_splits=INNER_SPLITS, shuffle=True,
                    random_state=SEED + fold_idx,
                )
                acc, auc = runner(X_tr, X_te, y_tr, y_te, inner)
                rows.append({
                    "model": model_name,
                    "condition": cond,
                    "fold_idx": fold_idx,
                    "accuracy": acc,
                    "auc": auc,
                })
                print(f"    fold {fold_idx}: acc={acc:.4f}  auc={auc:.4f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ablation_nested_cv_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Build markdown summary
    summary = (df.groupby(["model", "condition"])
                 .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                      auc_mean=("auc", "mean"), auc_std=("auc", "std"))
                 .reset_index())

    def cell(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    print("\n\n=== Markdown table (paste into paper) ===\n")
    print("| Model | Full Acc | Full AUC | No-Const Acc | No-Const AUC | Micro-Only Acc | Micro-Only AUC |")
    print("|---|---|---|---|---|---|---|")
    for model_name in RUNNERS.keys():
        cells = [model_name]
        for cond in CONDITIONS:
            row = summary[(summary["model"] == model_name) & (summary["condition"] == cond)].iloc[0]
            cells.append(cell(row["acc_mean"], row["acc_std"]))
            cells.append(cell(row["auc_mean"], row["auc_std"]))
        print("| " + " | ".join(cells) + " |")

    # Sanity check vs. existing Table II numbers
    table_ii = {
        "Decision Tree": (0.6754501915708813, 0.6263625894391058),
        "Random Forest": (0.7527394636015325, 0.7889836508146274),
        "XGBoost":       (0.7251436781609195, 0.770588457399611),
    }
    print("\n=== Sanity check vs. Table II (Microbiome-Only column) ===")
    for model_name, (ref_acc, ref_auc) in table_ii.items():
        row = summary[(summary["model"] == model_name) & (summary["condition"] == "Microbiome-Only")].iloc[0]
        print(f"  {model_name:>14} | new: acc={row['acc_mean']:.4f}, auc={row['auc_mean']:.4f}  "
              f"| ref: acc={ref_acc:.4f}, auc={ref_auc:.4f}  "
              f"| Δacc={row['acc_mean']-ref_acc:+.4f}, Δauc={row['auc_mean']-ref_auc:+.4f}")

    print("\n=== Sanity check vs. Table II (Full column, expected to match the existing artifact) ===")
    for model_name, (ref_acc, ref_auc) in table_ii.items():
        row = summary[(summary["model"] == model_name) & (summary["condition"] == "Full")].iloc[0]
        print(f"  {model_name:>14} | new: acc={row['acc_mean']:.4f}, auc={row['auc_mean']:.4f}  "
              f"| ref: acc={ref_acc:.4f}, auc={ref_auc:.4f}  "
              f"| Δacc={row['acc_mean']-ref_acc:+.4f}, Δauc={row['auc_mean']-ref_auc:+.4f}")


if __name__ == "__main__":
    main()
