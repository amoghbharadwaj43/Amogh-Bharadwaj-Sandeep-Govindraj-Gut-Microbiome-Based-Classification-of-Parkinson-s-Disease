#!/usr/bin/env python3
"""Gaussian noise stress test for the Decision Tree classifier.

Adds increasing levels of Gaussian noise to the top-20 feature set and
measures how accuracy, F1, precision, recall, and ROC-AUC degrade.
Produces a CSV table, a degradation curve PNG, and updates the findings file.
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path

# Matplotlib cache boilerplate
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "wallen_2022_primary_cohort" / "microbiome_V2.csv"
TARGET_COLUMN = "Case_status_PD"
FEATURES_FILE = BASE_DIR / "results" / "classification" / "top_20_features_dt.txt"
ARTIFACT_DIR = BASE_DIR / "results" / "classification"
RESULTS_DIR = BASE_DIR / "results" / "classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Noise levels: fraction of each feature's standard deviation
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]
N_TRIALS = 30  # repeated trials per noise level for stable estimates
RANDOM_SEED = 42


def load_data():
    """Load dataset and select top-20 DT features."""
    df = pd.read_csv(DATA_PATH)
    features = [f.strip() for f in FEATURES_FILE.read_text().splitlines() if f.strip()]

    y = df[TARGET_COLUMN]
    X = df[features].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

    return X, y, features


def train_clean_model(X_train, y_train):
    """Train a Decision Tree on clean data (mirrors model_pipeline_DT.py)."""
    model = DecisionTreeClassifier(
        random_state=RANDOM_SEED,
        class_weight="balanced",
        max_depth=10, min_samples_split=15, min_samples_leaf=3,
        criterion="log_loss", max_features=0.9155324799199657,
        ccp_alpha=0.018056399710624147,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y):
    """Return a dict of classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def add_gaussian_noise(X, noise_fraction, rng):
    """Add Gaussian noise scaled to noise_fraction * per-feature std."""
    stds = X.std(axis=0).values
    noise = rng.normal(0, 1, size=X.shape) * stds[np.newaxis, :] * noise_fraction
    return pd.DataFrame(X.values + noise, columns=X.columns, index=X.index)


def main():
    print("=" * 72)
    print("Gaussian Noise Stress Test — Decision Tree (Top 20 Features)")
    print("=" * 72)

    X, y, features = load_data()

    # Use same split as model_pipeline_DT.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y,
    )

    # Train on clean data
    model = train_clean_model(X_train, y_train)
    clean_metrics = evaluate(model, X_test, y_test)
    print(f"\nClean test accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"Clean test ROC AUC: {clean_metrics['roc_auc']:.4f}")

    # Also do CV on clean data for reference
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        DecisionTreeClassifier(
            random_state=RANDOM_SEED, class_weight="balanced",
            max_depth=10, min_samples_split=15, min_samples_leaf=3,
            criterion="log_loss", max_features=0.9155324799199657,
            ccp_alpha=0.018056399710624147,
        ),
        X, y, cv=cv, scoring="accuracy",
    )
    print(f"Clean CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Stress test: add noise to TEST set, measure degradation
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for noise_frac in NOISE_LEVELS:
        trial_metrics = {k: [] for k in clean_metrics}

        for trial in range(N_TRIALS):
            if noise_frac == 0.0:
                X_noisy = X_test.copy()
            else:
                X_noisy = add_gaussian_noise(X_test, noise_frac, rng)

            m = evaluate(model, X_noisy, y_test)
            for k in m:
                trial_metrics[k].append(m[k])

        row = {"noise_level": noise_frac}
        for k in trial_metrics:
            vals = np.array(trial_metrics[k])
            row[f"{k}_mean"] = vals.mean()
            row[f"{k}_std"] = vals.std()
        rows.append(row)

        print(
            f"  noise={noise_frac:.2f}x std  |  "
            f"acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}  |  "
            f"f1={row['f1_mean']:.4f}  |  "
            f"auc={row['roc_auc_mean']:.4f}"
        )

    results_df = pd.DataFrame(rows)

    # --- Save CSV ---
    csv_path = RESULTS_DIR / "stress_test_dt_gaussian.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # --- Degradation curve ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Decision Tree Stress Test: Gaussian Noise Degradation",
        fontsize=14, fontweight="bold",
    )

    metrics_to_plot = [
        ("accuracy_mean", "accuracy_std", "Accuracy", "#2563eb"),
        ("balanced_accuracy_mean", "balanced_accuracy_std", "Balanced Accuracy", "#7c3aed"),
        ("f1_mean", "f1_std", "F1 Score", "#059669"),
    ]

    for ax, (mean_col, std_col, label, color) in zip(axes[:3], metrics_to_plot):
        means = results_df[mean_col].values
        stds = results_df[std_col].values
        x = results_df["noise_level"].values

        ax.plot(x, means, "o-", color=color, linewidth=2, markersize=6, label=label)
        ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)
        ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.5, label="Clean baseline")
        ax.set_xlabel("Noise Level (fraction of feature std)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    png_path = RESULTS_DIR / "stress_test_dt_gaussian_curves.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")

    # --- ROC AUC degradation curve ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = results_df["noise_level"].values
    auc_means = results_df["roc_auc_mean"].values
    auc_stds = results_df["roc_auc_std"].values

    ax2.plot(x, auc_means, "o-", color="#dc2626", linewidth=2, markersize=6)
    ax2.fill_between(x, auc_means - auc_stds, auc_means + auc_stds, alpha=0.2, color="#dc2626")
    ax2.axhline(y=0.5, color="black", linestyle=":", alpha=0.5, label="Random (AUC=0.5)")
    ax2.axhline(y=auc_means[0], color="gray", linestyle="--", alpha=0.5, label="Clean baseline")
    ax2.set_xlabel("Noise Level (fraction of feature std)", fontsize=11)
    ax2.set_ylabel("ROC AUC", fontsize=11)
    ax2.set_title("Decision Tree ROC AUC Under Gaussian Noise", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)

    png_path2 = RESULTS_DIR / "stress_test_dt_gaussian_auc.png"
    fig2.savefig(png_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved AUC plot: {png_path2}")

    # --- Summary JSON ---
    # Find noise level where accuracy drops below 0.6 (degradation threshold)
    acc_means = results_df["accuracy_mean"].values
    threshold_60 = None
    for i, a in enumerate(acc_means):
        if a < 0.60:
            threshold_60 = float(results_df["noise_level"].iloc[i])
            break

    summary = {
        "model": "DecisionTree",
        "features": "top_20_dt",
        "n_noise_levels": len(NOISE_LEVELS),
        "n_trials_per_level": N_TRIALS,
        "clean_test_accuracy": clean_metrics["accuracy"],
        "clean_test_roc_auc": clean_metrics["roc_auc"],
        "clean_test_f1": clean_metrics["f1"],
        "clean_cv_accuracy_mean": float(cv_scores.mean()),
        "clean_cv_accuracy_std": float(cv_scores.std()),
        "noise_at_accuracy_below_60pct": threshold_60,
        "accuracy_at_50pct_noise": float(results_df.loc[results_df["noise_level"] == 0.50, "accuracy_mean"].iloc[0]),
        "accuracy_at_100pct_noise": float(results_df.loc[results_df["noise_level"] == 1.00, "accuracy_mean"].iloc[0]),
        "auc_at_50pct_noise": float(results_df.loc[results_df["noise_level"] == 0.50, "roc_auc_mean"].iloc[0]),
        "auc_at_100pct_noise": float(results_df.loc[results_df["noise_level"] == 1.00, "roc_auc_mean"].iloc[0]),
    }

    json_path = RESULTS_DIR / "stress_test_dt_gaussian_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {json_path}")

    print("\n" + "=" * 72)
    print("STRESS TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
