"""Effect sizes for the two novel species: log2 fold-change and prevalence.

Critic concern: Effect sizes (log2 fold-changes, prevalence in PD vs.
control) are missing for F. pleomorphus and Actinomyces odontolyticus.

Data note: metaphlan columns in Source_Data_24Oct2022_prepped.csv are
log2-transformed. Therefore:
  - log2FC = mean_PD_log2 - mean_ctrl_log2  (= log2 ratio of geometric means)
  - 'Direction' is inferred from the median difference (robust to right-skew
    outliers that can flip the mean-based direction for rare species)
  - 'Prevalence' = proportion with log2-abundance > 0 (i.e. above the 2^0 = 1
    unit threshold; consistent with the scale of these columns)

Outputs: novel_species_effect_sizes.csv for direct use in Table I-B text.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv"
TARGET_COLUMN = "Case_status=PD"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "results" / "novel_species"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

NOVEL_SPECIES = [
    "Faecalicoccus_pleomorphus",
    "Actinomyces_odontolyticus",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    return df, y


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else float("nan")


def effect_size_row(col: str, vals_pd: np.ndarray, vals_ctrl: np.ndarray) -> dict:
    mean_pd = float(np.mean(vals_pd))
    mean_ctrl = float(np.mean(vals_ctrl))
    med_pd = float(np.median(vals_pd))
    med_ctrl = float(np.median(vals_ctrl))

    # log2FC = difference of log2-space means = log2 ratio of geometric means
    log2fc = mean_pd - mean_ctrl

    # Median difference (robust direction indicator)
    median_diff = med_pd - med_ctrl

    # Prevalence: proportion with log2-abundance > 0 (above 1-count threshold)
    prev_pd = float((vals_pd > 0).mean() * 100)
    prev_ctrl = float((vals_ctrl > 0).mean() * 100)

    cd = cohens_d(vals_pd, vals_ctrl)
    _, pval = stats.ranksums(vals_pd, vals_ctrl)

    # Direction from median (robust to heavy right-skew outliers)
    direction = "enriched_PD" if median_diff > 0 else "depleted_PD"

    return {
        "feature": col,
        "mean_PD_log2": mean_pd,
        "mean_ctrl_log2": mean_ctrl,
        "median_PD_log2": med_pd,
        "median_ctrl_log2": med_ctrl,
        "log2FC_geom_means": log2fc,
        "median_diff_log2": median_diff,
        "prevalence_PD_pct": prev_pd,
        "prevalence_ctrl_pct": prev_ctrl,
        "cohens_d": cd,
        "wilcoxon_p": float(pval),
        "direction": direction,
    }


def main():
    print("Loading data...")
    df, y = load_data()
    pd_mask = y == 1
    ctrl_mask = y == 0
    n_pd = int(pd_mask.sum())
    n_ctrl = int(ctrl_mask.sum())
    print(f"  {len(y)} samples: {n_pd} PD, {n_ctrl} controls")
    print("  Note: metaphlan columns are log2-transformed; log2FC = mean_PD - mean_ctrl")

    rows = []
    print("\n" + "=" * 70)
    print("EFFECT SIZES FOR NOVEL SPECIES")
    print("=" * 70)

    for species in NOVEL_SPECIES:
        species_cols = [
            c for c in df.columns
            if species in c
            and (c.startswith("metaphlan_counts::") or c.startswith("metaphlan_rel_ab::"))
        ]
        print(f"\n{species}  ({len(species_cols)} columns)")
        for col in species_cols:
            vals_pd = df.loc[pd_mask, col].fillna(df[col].median()).values.astype(float)
            vals_ctrl = df.loc[ctrl_mask, col].fillna(df[col].median()).values.astype(float)
            row = effect_size_row(col, vals_pd, vals_ctrl)
            row["species"] = species
            rows.append(row)

            col_type = "counts" if "metaphlan_counts" in col else "rel_ab"
            print(f"  [{col_type}]")
            print(f"    Mean log2: PD {row['mean_PD_log2']:.3f}  vs  ctrl {row['mean_ctrl_log2']:.3f}  "
                  f"→ log2FC = {row['log2FC_geom_means']:+.3f}  ({row['direction']} by median)")
            print(f"    Median log2: PD {row['median_PD_log2']:.3f}  vs  ctrl {row['median_ctrl_log2']:.3f}  "
                  f"→ median diff = {row['median_diff_log2']:+.3f}")
            print(f"    Prevalence (log2 > 0): {row['prevalence_PD_pct']:.1f}% PD  vs  "
                  f"{row['prevalence_ctrl_pct']:.1f}% ctrl")
            print(f"    Cohen's d  : {row['cohens_d']:.3f}")
            print(f"    Wilcoxon p : {row['wilcoxon_p']:.4e}")

    result_df = pd.DataFrame(rows)[
        ["species", "feature", "log2FC_geom_means", "median_diff_log2", "direction",
         "mean_PD_log2", "mean_ctrl_log2", "median_PD_log2", "median_ctrl_log2",
         "prevalence_PD_pct", "prevalence_ctrl_pct",
         "cohens_d", "wilcoxon_p"]
    ]

    out_path = ARTIFACT_DIR / "novel_species_effect_sizes.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY FOR PAPER (rel_ab columns)")
    print("=" * 70)
    rel_ab = result_df[result_df["feature"].str.contains("metaphlan_rel_ab")]
    print(rel_ab[["species", "log2FC_geom_means", "median_diff_log2", "direction",
                   "prevalence_PD_pct", "prevalence_ctrl_pct",
                   "cohens_d", "wilcoxon_p"]].to_string(index=False))

    print(f"\nArtifact written to {out_path}")


if __name__ == "__main__":
    main()
