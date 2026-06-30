"""Nested CV grading for XGBoost, Random Forest, and Decision Tree.

Same 5 outer x 3 inner fold structure across all three models, with feature
re-selection inside each outer training fold using model-specific importance.
The shared outer fold seed guarantees identical splits across models so the
accuracy/AUC numbers are directly comparable.

XGBoost keeps Optuna tuning in the inner loop (the original justification for
nested CV). DT and RF have no tuning, but feature selection inside the loop
still requires nested CV to avoid selection-bias leakage.
"""

from __future__ import annotations

import ctypes
import json
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
ARTIFACT_DIR = PROJECT_ROOT / "results" / "classification"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_SPLITS = 5
INNER_SPLITS = 3
TOP_N = 20
OPTUNA_TRIALS = 20
SEED = 42

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")


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


# ---------------------------------------------------------------------------
# Per-model feature selection (run on outer-train only, inside the outer loop)
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


# ---------------------------------------------------------------------------
# XGBoost inner-CV Optuna tuning (DT/RF have no tuning step)
# ---------------------------------------------------------------------------

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
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Per-model outer-fold runners
# ---------------------------------------------------------------------------

def eval_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def run_xgb_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_xgb(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    best_params, inner_best = tune_xgb(Xtr, y_tr, inner_cv, spw)
    params = xgb_base_params(spw)
    params.update(best_params)
    model = XGBClassifier(**params)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        **eval_metrics(y_te, pred, proba),
        "inner_best_cv_acc": float(inner_best),
        "best_params": best_params,
        "top_features": feats,
        "_y_true": y_te.values.tolist(),
        "_y_proba": proba.tolist(),
        "_test_idx": list(map(int, y_te.index)),
    }


def run_rf_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_rf(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    model = RandomForestClassifier(
        n_estimators=400, random_state=SEED, n_jobs=-1,
        class_weight="balanced", max_depth=None,
    )
    inner = cross_val_score(model, Xtr, y_tr, cv=inner_cv, scoring="accuracy", n_jobs=1)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        **eval_metrics(y_te, pred, proba),
        "inner_cv_acc_mean": float(inner.mean()),
        "inner_cv_acc_std": float(inner.std()),
        "top_features": feats,
        "_y_true": y_te.values.tolist(),
        "_y_proba": proba.tolist(),
        "_test_idx": list(map(int, y_te.index)),
    }


def run_dt_fold(X_tr, X_te, y_tr, y_te, inner_cv):
    feats = select_top_dt(X_tr, y_tr)
    Xtr = X_tr[feats].values
    Xte = X_te[feats].values
    model = DecisionTreeClassifier(
        random_state=SEED, class_weight="balanced", criterion="gini",
    )
    inner = cross_val_score(model, Xtr, y_tr, cv=inner_cv, scoring="accuracy", n_jobs=1)
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        **eval_metrics(y_te, pred, proba),
        "inner_cv_acc_mean": float(inner.mean()),
        "inner_cv_acc_std": float(inner.std()),
        "top_features": feats,
        "_y_true": y_te.values.tolist(),
        "_y_proba": proba.tolist(),
        "_test_idx": list(map(int, y_te.index)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

METRIC_KEYS = ["accuracy", "roc_auc", "f1", "precision", "recall"]


def summarize(rows, model_name):
    df_ = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ("best_params", "top_features", "_y_true", "_y_proba", "_test_idx")}
        for r in rows
    ])
    print(f"\n--- {model_name} ({OUTER_SPLITS} outer folds) ---")
    for metric in METRIC_KEYS:
        v = df_[metric].values
        print(f"  {metric:>9}: {v.mean():.4f} ± {v.std():.4f}  "
              f"[{v.min():.4f}, {v.max():.4f}]")
    return df_


def feature_stability(rows):
    from collections import Counter
    c = Counter()
    for r in rows:
        for f in r["top_features"]:
            c[f] += 1
    return (pd.DataFrame(c.items(), columns=["feature", "n_folds_selected"])
            .sort_values("n_folds_selected", ascending=False))


def main():
    X, y = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance: {y.value_counts().to_dict()}")

    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    splits = list(outer.split(X, y))

    xgb_rows, rf_rows, dt_rows = [], [], []
    runners = [
        ("XGB", xgb_rows, run_xgb_fold),
        ("RF", rf_rows, run_rf_fold),
        ("DT", dt_rows, run_dt_fold),
    ]

    t0 = time.time()
    for fold_idx, (tr, te) in enumerate(splits, 1):
        print(f"\n=== Outer fold {fold_idx}/{OUTER_SPLITS} ===")
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        inner = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold_idx,
        )

        for name, rows, runner in runners:
            r = runner(X_tr, X_te, y_tr, y_te, inner)
            r["fold"] = fold_idx
            r["n_train"] = int(len(tr))
            r["n_test"] = int(len(te))
            rows.append(r)
            print(f"  {name:>3}: acc={r['accuracy']:.4f}  auc={r['roc_auc']:.4f}  "
                  f"f1={r['f1']:.4f}  prec={r['precision']:.4f}  rec={r['recall']:.4f}")

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"NESTED CV SUMMARY (elapsed {elapsed:.1f}s)")
    print(f"{OUTER_SPLITS} outer × {INNER_SPLITS} inner, top-{TOP_N} features re-selected per fold")
    print("Outer splits identical across models (seed=%d)" % SEED)
    print("=" * 60)

    xg_df = summarize(xgb_rows, "XGBoost")
    rf_df = summarize(rf_rows, "Random Forest")
    dt_df = summarize(dt_rows, "Decision Tree")

    xg_df.insert(0, "model", "XGBoost")
    rf_df.insert(0, "model", "RandomForest")
    dt_df.insert(0, "model", "DecisionTree")
    combined = pd.concat([xg_df, rf_df, dt_df], ignore_index=True)
    combined.to_csv(ARTIFACT_DIR / "nested_cv_all_models_fold_metrics.csv", index=False)

    head_summary = combined.groupby("model")[METRIC_KEYS].agg(["mean", "std"])
    head_summary.to_csv(ARTIFACT_DIR / "nested_cv_all_models_summary.csv")

    for name, rows in [("xgb", xgb_rows), ("rf", rf_rows), ("dt", dt_rows)]:
        feature_stability(rows).to_csv(
            ARTIFACT_DIR / f"nested_cv_all_models_{name}_feature_stability.csv",
            index=False,
        )
        oof_rows = []
        for r in rows:
            for idx, yt, yp in zip(r["_test_idx"], r["_y_true"], r["_y_proba"]):
                oof_rows.append({"sample_idx": idx, "fold": r["fold"],
                                 "y_true": yt, "y_proba": yp})
        pd.DataFrame(oof_rows).to_csv(
            ARTIFACT_DIR / f"nested_cv_all_models_{name}_oof_predictions.csv",
            index=False,
        )

    full = {
        "config": {
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "top_n": TOP_N,
            "optuna_trials_xgb": OPTUNA_TRIALS,
            "seed": SEED,
            "data_path": str(DATA_PATH),
            "target_column": TARGET_COLUMN,
        },
        "elapsed_seconds": elapsed,
        "xgb": xgb_rows,
        "rf": rf_rows,
        "dt": dt_rows,
    }
    with open(ARTIFACT_DIR / "nested_cv_all_models_full.json", "w") as f:
        json.dump(full, f, indent=2, default=str)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print("  - nested_cv_all_models_fold_metrics.csv")
    print("  - nested_cv_all_models_summary.csv")
    print("  - nested_cv_all_models_{xgb,rf,dt}_feature_stability.csv")
    print("  - nested_cv_all_models_full.json")


if __name__ == "__main__":
    main()
