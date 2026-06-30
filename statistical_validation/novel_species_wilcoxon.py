"""Wilcoxon rank-sum tests with BH-FDR correction for the two novel species.

Critic concern: The paper states F. pleomorphus and Actinomyces odontolyticus
are 'FDR-significant' but never gives the exact q-values. This script runs
the formal Wilcoxon rank-sum test and BH-FDR correction and prints / saves
the explicit q-values for both species across all their count and rel_ab
columns.

FDR correction is performed across all taxonomic species features jointly
(same universe used for the original discovery claim), ensuring q-values
reflect the multiple-testing burden of the full feature set.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

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

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    return df, y


def get_species_cols(df: pd.DataFrame, species: str) -> list[str]:
    return [c for c in df.columns if species in c
            and (c.startswith("metaphlan_counts::") or c.startswith("metaphlan_rel_ab::"))]


def wilcoxon_all_taxa(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run Wilcoxon rank-sum on every taxonomic species column and BH-correct."""
    taxon_cols = [
        c for c in df.columns
        if (c.startswith("metaphlan_counts::") or c.startswith("metaphlan_rel_ab::"))
        and not any(s in c.lower() for s in EXCLUDE_SUBSTRINGS)
    ]
    # Only species-level (contain s__)
    taxon_cols = [c for c in taxon_cols if "|s__" in c]

    pd_mask = y == 1
    ctrl_mask = y == 0

    rows = []
    for col in taxon_cols:
        vals_pd = df.loc[pd_mask, col].dropna().values
        vals_ctrl = df.loc[ctrl_mask, col].dropna().values
        if len(vals_pd) < 5 or len(vals_ctrl) < 5:
            continue
        stat, pval = stats.ranksums(vals_pd, vals_ctrl)
        rows.append({"feature": col, "statistic": stat, "pvalue": pval})

    result = pd.DataFrame(rows)
    _, qvals, _, _ = multipletests(result["pvalue"].values, method="fdr_bh")
    result["qvalue_bh"] = qvals
    result = result.sort_values("qvalue_bh").reset_index(drop=True)
    return result


def main():
    print("Loading data...")
    df, y = load_data()
    print(f"  {df.shape[0]} samples, {(y == 1).sum()} PD, {(y == 0).sum()} controls")

    print("\nRunning Wilcoxon rank-sum on all species-level features (BH-FDR)...")
    wilcoxon_df = wilcoxon_all_taxa(df, y)
    print(f"  Tested {len(wilcoxon_df)} features")

    sig = wilcoxon_df[wilcoxon_df["qvalue_bh"] < 0.05]
    print(f"  FDR-significant (q < 0.05): {len(sig)}")

    print("\n" + "=" * 70)
    print("q-VALUES FOR NOVEL SPECIES")
    print("=" * 70)

    summary_rows = []
    for species in NOVEL_SPECIES:
        cols = get_species_cols(df, species)
        matches = wilcoxon_df[wilcoxon_df["feature"].isin(cols)].copy()
        print(f"\n{species}:")
        if matches.empty:
            print("  NOT FOUND in tested features")
            continue
        for _, row in matches.iterrows():
            rank = int(wilcoxon_df.index[wilcoxon_df["feature"] == row["feature"]].tolist()[0]) + 1
            direction = "enriched_PD" if row["statistic"] > 0 else "depleted_PD"
            print(f"  {row['feature']}")
            print(f"    p = {row['pvalue']:.4e}  |  q (BH) = {row['qvalue_bh']:.4e}  "
                  f"|  direction: {direction}  |  rank among all tested: {rank}")
            summary_rows.append({
                "species": species,
                "feature": row["feature"],
                "statistic": row["statistic"],
                "pvalue": row["pvalue"],
                "qvalue_bh": row["qvalue_bh"],
                "direction": direction,
                "rank_among_all_tested": rank,
                "fdr_significant": row["qvalue_bh"] < 0.05,
            })

    summary_df = pd.DataFrame(summary_rows)
    out_path = ARTIFACT_DIR / "novel_species_wilcoxon_qvalues.csv"
    summary_df.to_csv(out_path, index=False)

    all_out = ARTIFACT_DIR / "all_species_wilcoxon_results.csv"
    wilcoxon_df.to_csv(all_out, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df[["species", "feature", "pvalue", "qvalue_bh", "direction",
                       "fdr_significant"]].to_string(index=False))

    print(f"\nArtifacts written to {ARTIFACT_DIR}/")
    print("  - novel_species_wilcoxon_qvalues.csv   (novel species q-values)")
    print("  - all_species_wilcoxon_results.csv     (full ranked list)")


if __name__ == "__main__":
    main()
