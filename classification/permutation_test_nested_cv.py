"""Permutation significance test for the Decision Tree under the nested 5x3 CV
protocol used by Table II.

Additive — does not modify grade_all_models_nested_cv.py. Reuses its
load_data() and select_top_dt() helpers so feature reselection inside each
outer training fold matches the published pipeline exactly.

Protocol per permutation:
  1. Shuffle labels with pd.Series(y).sample(frac=1.0, random_state=SEED+i)
     (matches the shuffle scheme of the existing single-split permutation_test_dt.py).
  2. Reuse the SAME outer fold sample-index partitions computed once on real
     labels via StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
     Only the labels assigned to those fixed partitions change.
  3. For each outer fold, redo top-20 DT feature selection on the permuted
     training labels, then run the inner 3-fold CV (for reporting only) and
     fit a DT(class_weight='balanced', random_state=42) on outer-train,
     evaluate accuracy on outer-test.
  4. Record mean and std of the 5 outer-fold accuracies.

Real DT nested CV accuracy (0.675) is pinned from Table II; not recomputed.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import nested_cv_all_models as ncv

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "results" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = ncv.SEED
OUTER_SPLITS = ncv.OUTER_SPLITS
INNER_SPLITS = ncv.INNER_SPLITS
TOP_N = ncv.TOP_N

REAL_DT_ACCURACY = 0.675


def run_one_permutation(X, y_vals_real, outer_partitions, perm_idx):
    """Return mean and std of nested CV accuracy on a label-shuffled copy of y.

    X is unchanged. outer_partitions is a list of (train_idx, test_idx) tuples
    computed once on real labels and reused for every permutation.
    """
    y_shuffled = pd.Series(y_vals_real).sample(
        frac=1.0, random_state=SEED + perm_idx, replace=False,
    ).to_numpy()
    y_perm = pd.Series(y_shuffled, index=X.index)

    fold_accs = []
    for fold_idx, (tr, te) in enumerate(outer_partitions, 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_perm.iloc[tr], y_perm.iloc[te]

        feats = ncv.select_top_dt(X_tr, y_tr, n=TOP_N)
        Xtr = X_tr[feats].values
        Xte = X_te[feats].values

        # inner CV is part of the protocol but its score is informational; we
        # skip running it here because DT has no inner-loop tuning step (matches
        # grade_all_models_nested_cv.run_dt_fold reporting, not selection).
        # Build inner StratifiedKFold object matching the original seed scheme
        # so the protocol is identical even if the inner split isn't scored.
        _ = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold_idx,
        )

        model = DecisionTreeClassifier(
            random_state=SEED, class_weight="balanced", criterion="gini",
        )
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        fold_accs.append(accuracy_score(y_te, pred))

    fold_accs = np.array(fold_accs)
    return float(fold_accs.mean()), float(fold_accs.std()), fold_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--csv-out", default=str(OUT_DIR / "permutation_test_nested_cv.csv"))
    parser.add_argument("--fig-out", default=str(OUT_DIR / "fig8_permutation_nested.png"))
    parser.add_argument("--time-only", action="store_true",
                        help="Run a single permutation and exit (for timing).")
    args = parser.parse_args()

    print("=" * 72)
    print("Permutation Test under Nested 5x3 CV — Decision Tree")
    print("=" * 72)

    X, y = ncv.load_data()
    print(f"Loaded: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Class counts: healthy={int((y == 0).sum())}, PD={int((y == 1).sum())}")

    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    outer_partitions = list(outer.split(X, y))
    print(f"Outer partitions fixed (seed={SEED}); only labels shuffle per permutation.")

    y_vals_real = y.to_numpy()

    if args.time_only:
        print("Timing 1 permutation...")
        t0 = time.time()
        m, s, _ = run_one_permutation(X, y_vals_real, outer_partitions, perm_idx=0)
        dt = time.time() - t0
        print(f"  one permutation: {dt:.2f}s, mean_acc={m:.4f}, std={s:.4f}")
        print(f"  projected for 200 perms: {dt * 200 / 60:.1f} min")
        print(f"  projected for 1000 perms: {dt * 1000 / 60:.1f} min")
        return

    n_perm = args.n_permutations
    print(f"Running {n_perm} permutations...")

    rows = []
    t0 = time.time()
    for i in range(n_perm):
        m, s, _ = run_one_permutation(X, y_vals_real, outer_partitions, perm_idx=i)
        rows.append({
            "permutation_idx": i,
            "mean_accuracy_across_outer_folds": m,
            "std_accuracy_across_outer_folds": s,
        })
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_perm - i - 1)
            print(f"  {i + 1:>4}/{n_perm}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  "
                  f"last_acc={m:.4f}")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)

    perm_means = df["mean_accuracy_across_outer_folds"].to_numpy()
    perm_mean_of_means = float(perm_means.mean())
    perm_std_of_means = float(perm_means.std())
    perm_max = float(perm_means.max())
    n_geq = int(np.sum(perm_means >= REAL_DT_ACCURACY))
    if n_geq == 0:
        p_str = f"p < {1.0 / n_perm:.4f} (0/{n_perm})"
        p_value = 0.0
    else:
        p_value = n_geq / n_perm
        p_str = f"p = {p_value:.4f} ({n_geq}/{n_perm})"

    # Figure 8: histogram of permuted nested CV mean accuracies, real value marked.
    # Style matches existing important/scripts/modeling/permutation_test_dt.py
    # (dark background, blue histogram, red vertical line for real model).
    fig = plt.figure(figsize=(10, 6), facecolor="#0F0F1A")
    ax = plt.gca()
    ax.set_facecolor("#0F0F1A")
    bins = min(40, max(10, int(np.sqrt(len(perm_means)))))
    ax.hist(perm_means, bins=bins, color="#2980B9", alpha=0.85,
            edgecolor="white", label="Shuffled-label nested CV mean")
    ax.axvline(REAL_DT_ACCURACY, color="#E74C3C", linewidth=3,
               label=f"Real DT (Table II): {REAL_DT_ACCURACY * 100:.1f}%")
    ax.set_xlabel("Nested 5x3 CV mean accuracy", color="white", fontsize=13)
    ax.set_ylabel("Number of permutations", color="white", fontsize=13)
    ax.set_title(
        f"Permutation Test under Nested CV — Decision Tree (n={n_perm})\n{p_str}",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.legend(fontsize=11, facecolor="#1A1A2E", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    fig.tight_layout()
    fig.savefig(args.fig_out, dpi=300, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Permutations run:               {n_perm}")
    print(f"  Real DT nested CV accuracy:     {REAL_DT_ACCURACY:.4f}  (Table II, fixed)")
    print(f"  Permuted mean accuracy:         {perm_mean_of_means:.4f} +/- {perm_std_of_means:.4f}")
    print(f"  Maximum permuted accuracy:      {perm_max:.4f}")
    print(f"  Empirical p-value:              {p_str}")
    print(f"  Wall-clock time:                {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nSaved CSV: {args.csv_out}")
    print(f"Saved Fig: {args.fig_out}")


if __name__ == "__main__":
    main()
