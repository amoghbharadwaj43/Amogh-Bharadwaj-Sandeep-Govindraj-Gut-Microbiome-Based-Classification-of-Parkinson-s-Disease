"""
RF permutation test.

Uses the RF-stable features (selected in >=3/5 nested CV outer folds) from
Source_Data_24Oct2022_prepped.csv, runs 5-fold stratified CV on real labels,
then 200 permutations of shuffled labels to build the null distribution.

Saves results to Repurposed/artifacts/permutation_test_rf_summary.json
and Repurposed/artifacts/permutation_test_rf_shuffled_accuracies.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "mpl"))

DATA_PATH   = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv"
TARGET      = "Case_status=PD"
OUT_DIR     = PROJECT_ROOT / "results" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 200
N_FOLDS        = 5
SEED           = 42

# Features stable in >=3 of 5 outer folds from nested CV
RF_STABLE_FEATURES = [
    "Sex=M",
    "Constipation=Y",
    "metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae",
    "metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae|g__Roseburia",
    "metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia",
    "metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Clostridiaceae|g__Clostridium",
    "metaphlan_counts::k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Micrococcales|f__Micrococcaceae|g__Rothia|s__Rothia_mucilaginosa",
    "humann_pathway_counts::PWY-6892: thiazole component of thiamine diphosphate biosynthesis I",
    "metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales",
    "humann_KO_group_counts::K10541: methyl-galactoside transport system permease protein",
    "metaphlan_rel_ab::k__Bacteria|p__Firmicutes|c__Bacilli|o__Lactobacillales|f__Streptococcaceae|g__Streptococcus|s__Streptococcus_australis",
]


def main():
    print("=" * 58)
    print("  RF Permutation Test")
    print("=" * 58)

    df = pd.read_csv(DATA_PATH)
    y  = df[TARGET].astype(int).values

    available = [f for f in RF_STABLE_FEATURES if f in df.columns]
    missing   = [f for f in RF_STABLE_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: {len(missing)} features not found in data, skipping: {missing}")

    X = df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    print(f"Using {len(available)} stable RF features on {len(y)} samples")
    print(f"Class distribution: PD={y.sum()}, Control={(1-y).sum()}")

    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"\nRunning {N_FOLDS}-fold CV on real labels...")
    real_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    real_accuracy = float(real_scores.mean())
    print(f"Real CV accuracy: {real_accuracy*100:.2f}% (+/- {real_scores.std()*100:.2f}%)")

    print(f"\nRunning {N_PERMUTATIONS} permutations...")
    rng = np.random.default_rng(SEED)
    shuffled_accs = []
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        perm_scores = cross_val_score(rf, X, y_perm, cv=cv, scoring="accuracy")
        shuffled_accs.append(float(perm_scores.mean()))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_PERMUTATIONS} permutations done")

    shuffled_accs = np.array(shuffled_accs)
    p_value = float((shuffled_accs >= real_accuracy).mean())
    p_corrected = (np.sum(shuffled_accs >= real_accuracy) + 1) / (N_PERMUTATIONS + 1)

    print(f"\nResults:")
    print(f"  Real CV accuracy:    {real_accuracy*100:.2f}%")
    print(f"  Shuffled mean:       {shuffled_accs.mean()*100:.2f}%")
    print(f"  Shuffled std:        {shuffled_accs.std()*100:.2f}%")
    print(f"  p-value:             {p_value:.4f}")
    print(f"  p-value (corrected): {p_corrected:.4f}")

    summary = {
        "data_path":            str(DATA_PATH),
        "n_samples":            int(len(y)),
        "n_features":           len(available),
        "features_used":        available,
        "n_folds":              N_FOLDS,
        "n_permutations":       N_PERMUTATIONS,
        "random_seed":          SEED,
        "real_cv_accuracy":     real_accuracy,
        "real_cv_folds":        real_scores.tolist(),
        "shuffled_mean":        float(shuffled_accs.mean()),
        "shuffled_std":         float(shuffled_accs.std()),
        "shuffled_best":        float(shuffled_accs.max()),
        "p_value":              p_value,
        "p_value_corrected":    p_corrected,
    }
    (OUT_DIR / "permutation_test_rf_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    pd.DataFrame({"shuffled_accuracy": shuffled_accs}).to_csv(
        OUT_DIR / "permutation_test_rf_shuffled_accuracies.csv", index=False
    )

    # Plot null distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(shuffled_accs * 100, bins=30, color="#2980b9", alpha=0.7,
            edgecolor="white", label="Shuffled labels")
    ax.axvline(real_accuracy * 100, color="#e74c3c", linewidth=2.5,
               label=f"Real accuracy: {real_accuracy*100:.1f}%")
    ax.set_xlabel("CV Accuracy (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"RF Permutation Test (n={N_PERMUTATIONS})\np={p_corrected:.4f}",
        fontsize=13
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "permutation_test_rf_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
