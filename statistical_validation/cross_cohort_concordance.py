#!/usr/bin/env python3
"""
Cross-cohort biomarker concordance analysis.

Compares Wallen et al. (2022) XGBoost + RWBC results against
Metcalfe-Roach et al. (2024) supplementary data files.

Outputs
-------
artifacts/concordance_xgb_vs_mr_abundance.csv
artifacts/concordance_xgb_vs_mr_rf.csv
artifacts/concordance_rwbc_vs_mr_betweenness.csv
artifacts/concordance_summary.csv
artifacts/concordance_figure.png
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Matplotlib cache setup consistent with other scripts in this project
_TMP = Path(tempfile.gettempdir())
_MPL = _TMP / "mplconfig_concordance"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS    = PROJECT_ROOT / "results" / "cross_cohort"
MR_DIR       = (
    PROJECT_ROOT
    / "data"
    / "metcalfe_roach_2024_validation"
    / "Results"
    / "Supplementary Data"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HARDCODED WALLEN (2022) RESULTS
# ─────────────────────────────────────────────────────────────────────────────
#
# XGBoost features come from two files:
#   (a) artifacts/feature_importance_top10.csv          — raw gain, top species
#   (b) artifacts/xg_combined_pvalues_metrics_table.csv — features that are BOTH
#       in XGBoost AND in the RWBC co-abundance network (normalised gain scale)
#
# Direction "depleted_PD"  = species is LESS abundant in PD vs controls
#           "enriched_PD"  = species is MORE abundant in PD vs controls
#
# Sources for direction:
#   • (b) supplies direction directly via the "↑/↓ PD" column
#   • (a)-only species (not in (b)) use Wallen 2022 differential-abundance
#     results confirmed by published PD microbiome literature

WALLEN_XGB = pd.DataFrame([
    # Genuine XGBoost top-20 bacterial features from nested CV stable set.
    # Directions from Wallen 2022 differential abundance (PD vs controls).
    # These four are the only microbial taxa that appear in the actual XGBoost
    # top-20 features (artifacts/modeling/permutation_test_xg_summary.json).
    {
        "feature": "Blautia wexlerae",
        "importance": 11.508,
        "direction": "depleted_PD",
        "importance_source": "raw_gain",
    },
    {
        "feature": "Roseburia",
        "importance": 9.700,
        "direction": "depleted_PD",
        "importance_source": "raw_gain",
    },
    {
        "feature": "Streptococcus australis",
        "importance": 8.192,
        "direction": "enriched_PD",
        "importance_source": "raw_gain",
    },
    {
        "feature": "Actinomyces odontolyticus",
        "importance": 4.971,
        "direction": "enriched_PD",
        "importance_source": "raw_gain",
    },
])

# Top RWBC hub taxa from artifacts/rwbc_combined_top10_table.csv.
# rwbc_pd        = Random Walk Betweenness Centrality in the PD co-abundance network
# delta_rwbc     = rwbc_pd − rwbc_healthy  (positive = more central in PD)
# delta_direction = "higher_in_PD" / "higher_in_healthy" / "equal"

WALLEN_RWBC = pd.DataFrame([
    {
        "taxon": "Klebsiella michiganensis",
        "rwbc_pd": 2.7132,
        "delta_rwbc":  1.3970,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Clostridiales",          # order-level hub
        "rwbc_pd": 2.3695,
        "delta_rwbc":  1.2901,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Clostridia",             # class-level hub
        "rwbc_pd": 2.3612,
        "delta_rwbc":  1.3012,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Lachnospiraceae",        # family-level hub (counts feature)
        "rwbc_pd": 2.3468,
        "delta_rwbc":  1.1138,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Pseudoflavonifractor capillosus",
        "rwbc_pd": 1.8703,
        "delta_rwbc":  1.0342,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Lactobacillus salivarius",
        "rwbc_pd": 0.2374,
        "delta_rwbc": -0.9891,
        "delta_direction": "higher_in_healthy",
    },
    {
        "taxon": "Catenibacterium",
        "rwbc_pd": 0.6325,
        "delta_rwbc":  0.3054,
        "delta_direction": "higher_in_PD",
    },
    {
        "taxon": "Eubacterium limosum",
        "rwbc_pd": 0.2307,
        "delta_rwbc": -0.1074,
        "delta_direction": "higher_in_healthy",
    },
    {
        "taxon": "Ruthenibacterium lactatiformans",
        "rwbc_pd": 0.0000,
        "delta_rwbc":  0.0000,
        "delta_direction": "equal",
    },
])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_name(raw: str) -> str:
    """
    Return a canonical lowercase species-or-genus string for cross-dataset matching.

    Handles:
    • MetaPhlAn pipe-delimited lineages  k__Bacteria|...|s__Foo_bar  → "foo bar"
    • Metcalfe-Roach dash-delimited       k__Bacteria-...-s__Foo_bar  → "foo bar"
    • Rank-prefix only                    s__Foo_bar                  → "foo bar"
    • Plain name                          Foo bar / Foo_bar           → "foo bar"
    """
    name = str(raw).strip()
    if "|" in name:
        name = name.split("|")[-1]
    elif "-s__" in name:
        # take everything after the last species-rank separator
        name = "s__" + name.split("-s__")[-1]
    # strip any rank prefix (s__, g__, f__, o__, c__, p__)
    for prefix in ("s__", "g__", "f__", "o__", "c__", "p__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").lower().strip()


def names_match(a: str, b: str) -> bool:
    """
    Return True if two (raw or normalised) names refer to the same taxon.

    Supports exact match and genus-level partial match (one name is a single
    word that is the genus prefix of the other).
    """
    na, nb = normalize_name(a), normalize_name(b)
    if na == nb:
        return True
    # genus-level: one entry has only a genus name, the other includes genus + species
    parts_a, parts_b = na.split(), nb.split()
    if parts_a[0] == parts_b[0] and (len(parts_a) == 1 or len(parts_b) == 1):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LOAD METCALFE-ROACH DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_mr_abundance() -> pd.DataFrame:
    """
    Load Taxonomy_Abundance_and_Prevalence.xlsx, sheet "Species".
    Columns used: Species, Log 2 FC (PD/Ctrl), Rel_Ab_PD, Rel_Ab_Ctrl.
    Adds a "mr_direction" column: log2fc > 0 → enriched_PD, else depleted_PD.
    """
    path = MR_DIR / "Differential Abundance" / "Taxonomy_Abundance_and_Prevalence.xlsx"
    df = pd.read_excel(path, sheet_name="Species")
    df = df.rename(columns={
        "Species":          "species",
        "Log 2 FC (PD/Ctrl)": "log2fc",
        "Rel_Ab_PD":        "rel_ab_pd",
        "Rel_Ab_Ctrl":      "rel_ab_ctrl",
        "Prevalence_PD":    "prevalence_pd",
        "Prevalence_Ctrl":  "prevalence_ctrl",
    })
    df["mr_direction"] = df["log2fc"].apply(
        lambda x: "enriched_PD" if pd.notna(x) and x > 0 else "depleted_PD"
    )
    df["species_norm"] = df["species"].apply(normalize_name)
    return df


def load_mr_rf(top_n: int = 50) -> pd.DataFrame:
    """
    Load RF Importance Values - Status.xlsx, sheet "Taxonomy".
    Returns the top_n features sorted by importance (descending).
    """
    path = MR_DIR / "Random Forest" / "RF Importance Values - Status.xlsx"
    df = pd.read_excel(path, sheet_name="Taxonomy")
    # Column names may have a leading unnamed index column; keep Feature + Importance
    df.columns = [str(c).strip() for c in df.columns]
    if "Feature" not in df.columns:
        # First non-index column is Feature, second is Importance
        df = df.iloc[:, 1:3]
        df.columns = ["feature", "importance"]
    else:
        df = df.rename(columns={"Feature": "feature", "Importance": "importance"})
        df = df[["feature", "importance"]]
    df = df.dropna(subset=["feature"]).copy()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    df["feature_norm"] = df["feature"].apply(normalize_name)
    # MR RF rank is 1-indexed position in this sorted list
    df.insert(0, "mr_rf_rank", df.index + 1)
    return df


def load_mr_betweenness() -> pd.DataFrame:
    """
    Load Network Analysis Species-Level Data.xlsx, sheet "Betweenness Centrality".
    Columns: Status, Taxon, Betweenness, Kingdom … Species.
    Extracts a normalised species name from the Taxon lineage string.
    """
    path = MR_DIR / "Network Analysis" / "Network Analysis Species-Level Data.xlsx"
    df = pd.read_excel(path, sheet_name="Betweenness Centrality")
    df.columns = [str(c).strip() for c in df.columns]
    # Extract species from the Taxon lineage (last component after '-s__')
    def extract_species(taxon: str) -> str:
        taxon = str(taxon)
        if "-s__" in taxon:
            return taxon.split("-s__")[-1].replace("_", " ")
        # fall back to the last '-'-separated component
        return taxon.split("-")[-1].replace("_", " ")

    df["species_clean"] = df["Taxon"].apply(extract_species)
    df["species_norm"]  = df["Taxon"].apply(normalize_name)
    df["Betweenness"]   = pd.to_numeric(df["Betweenness"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

def compare_xgb_vs_abundance(
    wallen_xgb: pd.DataFrame,
    mr_abund: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each Wallen XGBoost feature look it up in the Metcalfe-Roach species
    abundance table and check whether the direction of PD change matches.

    Returns a row-per-Wallen-feature DataFrame with columns:
        wallen_feature, wallen_direction, wallen_importance,
        mr_species, mr_log2fc, mr_direction,
        mr_rel_ab_pd, mr_rel_ab_ctrl,
        found_in_MR, direction_concordant
    """
    rows = []
    for _, row in wallen_xgb.iterrows():
        # Boolean mask: any MR species that matches this Wallen feature name
        mask = mr_abund["species_norm"].apply(lambda x: names_match(row["feature"], x))
        matches = mr_abund[mask]

        if len(matches) > 0:
            best = matches.iloc[0]   # take the closest / first match
            dir_match = row["direction"] == best["mr_direction"]
            rows.append({
                "wallen_feature":       row["feature"],
                "wallen_direction":     row["direction"],
                "wallen_importance":    row["importance"],
                "mr_species":           best["species"],
                "mr_log2fc":            best["log2fc"],
                "mr_direction":         best["mr_direction"],
                "mr_rel_ab_pd":         best.get("rel_ab_pd"),
                "mr_rel_ab_ctrl":       best.get("rel_ab_ctrl"),
                "found_in_MR":          True,
                "direction_concordant": dir_match,
            })
        else:
            rows.append({
                "wallen_feature":       row["feature"],
                "wallen_direction":     row["direction"],
                "wallen_importance":    row["importance"],
                "mr_species":           None,
                "mr_log2fc":            None,
                "mr_direction":         None,
                "mr_rel_ab_pd":         None,
                "mr_rel_ab_ctrl":       None,
                "found_in_MR":          False,
                "direction_concordant": None,
            })
    return pd.DataFrame(rows)


def compare_xgb_vs_rf(
    wallen_xgb: pd.DataFrame,
    mr_rf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find Wallen XGBoost species that also appear among the top Metcalfe-Roach
    RF features.

    Returns a row-per-Wallen-feature DataFrame with columns:
        wallen_feature, wallen_importance,
        mr_feature, mr_rf_rank, mr_importance, overlap
    """
    rows = []
    for _, row in wallen_xgb.iterrows():
        mask = mr_rf["feature_norm"].apply(lambda x: names_match(row["feature"], x))
        matches = mr_rf[mask]

        if len(matches) > 0:
            best = matches.iloc[0]
            rows.append({
                "wallen_feature":    row["feature"],
                "wallen_importance": row["importance"],
                "mr_feature":        best["feature"],
                "mr_rf_rank":        int(best["mr_rf_rank"]),
                "mr_importance":     best["importance"],
                "overlap":           True,
            })
        else:
            rows.append({
                "wallen_feature":    row["feature"],
                "wallen_importance": row["importance"],
                "mr_feature":        None,
                "mr_rf_rank":        None,
                "mr_importance":     None,
                "overlap":           False,
            })
    return pd.DataFrame(rows)


def compare_rwbc_vs_betweenness(
    wallen_rwbc: pd.DataFrame,
    mr_bc: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each Wallen RWBC hub taxon, look it up in both the PD and control
    Metcalfe-Roach networks and check whether the relative centrality direction
    (higher in PD vs higher in healthy) is concordant.

    The Status column is searched case-insensitively; common values include
    "PD", "HC", "Control", "Healthy".

    Returns a row-per-Wallen-taxon DataFrame with columns:
        wallen_taxon, wallen_rwbc_pd, wallen_delta, wallen_direction,
        found_in_MR_PD, found_in_MR_ctrl,
        mr_bc_pd, mr_bc_ctrl, mr_delta_direction, direction_concordant
    """
    # Detect the two status values used in this file
    statuses = mr_bc["Status"].dropna().unique()
    pd_statuses   = [s for s in statuses if str(s).strip().upper() == "PD"]
    ctrl_statuses = [s for s in statuses if str(s).strip().upper() in
                     {"HC", "CTRL", "CONTROL", "HEALTHY", "CONTROL GROUP"}]

    mr_pd_net   = mr_bc[mr_bc["Status"].isin(pd_statuses)].copy()
    mr_ctrl_net = mr_bc[mr_bc["Status"].isin(ctrl_statuses)].copy()

    rows = []
    for _, row in wallen_rwbc.iterrows():
        match_pd   = mr_pd_net[mr_pd_net["species_norm"].apply(
                        lambda x: names_match(row["taxon"], x))]
        match_ctrl = mr_ctrl_net[mr_ctrl_net["species_norm"].apply(
                        lambda x: names_match(row["taxon"], x))]

        found_pd   = len(match_pd)   > 0
        found_ctrl = len(match_ctrl) > 0

        mr_bc_pd   = float(match_pd.iloc[0]["Betweenness"])   if found_pd   else None
        mr_bc_ctrl = float(match_ctrl.iloc[0]["Betweenness"]) if found_ctrl else None

        # Determine MR delta direction and concordance
        if mr_bc_pd is not None and mr_bc_ctrl is not None:
            if mr_bc_pd > mr_bc_ctrl:
                mr_dir = "higher_in_PD"
            elif mr_bc_pd < mr_bc_ctrl:
                mr_dir = "higher_in_healthy"
            else:
                mr_dir = "equal"
            concordant = (row["delta_direction"] == mr_dir)
        else:
            mr_dir     = None
            concordant = None

        rows.append({
            "wallen_taxon":        row["taxon"],
            "wallen_rwbc_pd":      row["rwbc_pd"],
            "wallen_delta":        row["delta_rwbc"],
            "wallen_direction":    row["delta_direction"],
            "found_in_MR_PD":      found_pd,
            "found_in_MR_ctrl":    found_ctrl,
            "mr_bc_pd":            mr_bc_pd,
            "mr_bc_ctrl":          mr_bc_ctrl,
            "mr_delta_direction":  mr_dir,
            "direction_concordant": concordant,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _bar_panel(
    ax: plt.Axes,
    counts: list[int],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str = "Number of features/taxa",
) -> None:
    """Draw a labelled bar chart on the given Axes."""
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.55)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(cnt),
                ha="center", va="bottom", fontweight="bold", fontsize=10,
            )
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, max(counts, default=1) + 2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def plot_concordance_summary(
    xgb_abund: pd.DataFrame,
    xgb_rf: pd.DataFrame,
    rwbc_bc: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Three-panel concordance figure.

    Panel A  XGBoost feature direction concordance vs MR abundance
    Panel B  XGBoost / MR RF feature overlap
    Panel C  RWBC hub centrality direction concordance vs MR betweenness
    """
    found_abund     = xgb_abund[xgb_abund["found_in_MR"]]
    conc_abund      = found_abund[found_abund["direction_concordant"] == True]
    disc_abund      = found_abund[found_abund["direction_concordant"] == False]
    notfound_abund  = xgb_abund[~xgb_abund["found_in_MR"]]
    conc_rate_abund = (
        len(conc_abund) / len(found_abund) * 100 if len(found_abund) > 0 else 0
    )

    rf_overlap   = xgb_rf["overlap"].sum()
    rf_only_wall = (~xgb_rf["overlap"]).sum()
    rf_rate      = rf_overlap / len(xgb_rf) * 100 if len(xgb_rf) > 0 else 0

    hub_found = rwbc_bc[rwbc_bc["found_in_MR_PD"] | rwbc_bc["found_in_MR_ctrl"]]
    hub_conc  = hub_found[hub_found["direction_concordant"] == True]
    hub_disc  = hub_found[hub_found["direction_concordant"] == False]
    hub_partial = hub_found[hub_found["direction_concordant"].isna()]
    hub_absent  = rwbc_bc[~(rwbc_bc["found_in_MR_PD"] | rwbc_bc["found_in_MR_ctrl"])]
    hub_rate    = (
        len(hub_conc) / len(hub_found) * 100 if len(hub_found) > 0 else 0
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Cross-Cohort Biomarker Concordance\n"
        "Wallen et al. 2022 (USA, n=724) vs Metcalfe-Roach et al. 2024 (Canada, n=276)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Panel A ──
    _bar_panel(
        axes[0],
        counts=[len(conc_abund), len(disc_abund), len(notfound_abund)],
        labels=["Concordant\ndirection", "Discordant\ndirection", "Not found\nin MR"],
        colors=["#2ecc71", "#e74c3c", "#95a5a6"],
        title=(
            f"A: XGBoost Features vs MR Abundance\n"
            f"Direction concordance: {len(conc_abund)}/{len(found_abund)}"
            f" ({conc_rate_abund:.0f}%)"
        ),
    )

    # ── Panel B ──
    _bar_panel(
        axes[1],
        counts=[rf_overlap, rf_only_wall],
        labels=["Overlap with\nMR RF", "Only in\nWallen XGB"],
        colors=["#3498db", "#bdc3c7"],
        title=(
            f"B: XGBoost vs MR Random Forest\n"
            f"Feature overlap: {rf_overlap}/{len(xgb_rf)} ({rf_rate:.0f}%)"
        ),
    )

    # ── Panel C ──
    _bar_panel(
        axes[2],
        counts=[len(hub_conc), len(hub_disc), len(hub_partial), len(hub_absent)],
        labels=["Concordant\nhub", "Discordant\nhub", "Found\n(1 net only)", "Not\nfound"],
        colors=["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"],
        title=(
            f"C: RWBC Hubs vs MR Betweenness\n"
            f"Hub overlap: {len(hub_found)}/{len(rwbc_bc)}, "
            f"concordant: {len(hub_conc)} ({hub_rate:.0f}%)"
        ),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN / REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f") -> str:
    """Format a possibly-None float for printing."""
    return f"{val:{fmt}}" if val is not None else "N/A"


def print_report(
    xgb_abund: pd.DataFrame,
    xgb_rf: pd.DataFrame,
    rwbc_bc: pd.DataFrame,
) -> None:
    """Print a structured concordance report to stdout."""
    sep = "─" * 70

    # ── Comparison 1 ──
    found       = xgb_abund[xgb_abund["found_in_MR"]]
    concordant  = found[found["direction_concordant"] == True]
    discordant  = found[found["direction_concordant"] == False]
    not_found   = xgb_abund[~xgb_abund["found_in_MR"]]
    conc_rate   = len(concordant) / len(found) * 100 if len(found) > 0 else 0.0

    print(f"\n{sep}")
    print("COMPARISON 1: Wallen XGBoost Features vs MR Species Abundance")
    print(sep)
    print(f"  XGBoost features tested : {len(WALLEN_XGB)}")
    print(f"  Found in MR data        : {len(found)}")
    print(f"  Direction concordant    : {len(concordant)} ({conc_rate:.1f}%)")
    if len(concordant) > 0:
        print("\n  Concordant taxa:")
        for _, r in concordant.iterrows():
            print(
                f"    + {r['wallen_feature']:<38}"
                f" log2fc={_fmt(r['mr_log2fc'], '+.3f')}"
                f"  [{r['mr_direction']}]"
            )
    if len(discordant) > 0:
        print("\n  Discordant taxa:")
        for _, r in discordant.iterrows():
            print(
                f"    ~ {r['wallen_feature']:<38}"
                f" Wallen={r['wallen_direction']}"
                f"  MR={r['mr_direction']}"
                f"  log2fc={_fmt(r['mr_log2fc'], '+.3f')}"
            )
    if len(not_found) > 0:
        print("\n  Not found in MR abundance table:")
        for _, r in not_found.iterrows():
            print(f"    - {r['wallen_feature']}")

    # ── Comparison 2 ──
    overlap    = xgb_rf[xgb_rf["overlap"]]
    rf_rate    = len(overlap) / len(xgb_rf) * 100 if len(xgb_rf) > 0 else 0.0

    print(f"\n{sep}")
    print("COMPARISON 2: Wallen XGBoost Features vs MR Random Forest")
    print(sep)
    print(f"  XGBoost features tested : {len(WALLEN_XGB)}")
    print(f"  Overlapping with MR RF  : {len(overlap)} ({rf_rate:.1f}%)")
    for _, r in overlap.iterrows():
        print(
            f"    ✓ {r['wallen_feature']:<38}"
            f" MR RF rank={r['mr_rf_rank']}"
            f"  MR importance={_fmt(r['mr_importance'], '.5f')}"
        )

    # ── Comparison 3 ──
    hub_found   = rwbc_bc[rwbc_bc["found_in_MR_PD"] | rwbc_bc["found_in_MR_ctrl"]]
    hub_conc    = hub_found[hub_found["direction_concordant"] == True]
    hub_rate    = len(hub_conc) / len(hub_found) * 100 if len(hub_found) > 0 else 0.0

    print(f"\n{sep}")
    print("COMPARISON 3: Wallen RWBC Hubs vs MR Betweenness Centrality")
    print(sep)
    print(f"  RWBC hubs tested        : {len(WALLEN_RWBC)}")
    print(f"  Found in MR network     : {len(hub_found)}")
    print(f"  Direction concordant    : {len(hub_conc)} ({hub_rate:.1f}%)")
    for _, r in hub_found.iterrows():
        if r["direction_concordant"] is True:
            tag = "CONCORDANT"
        elif r["direction_concordant"] is False:
            tag = "DISCORDANT"
        else:
            tag = "PARTIAL   "
        print(
            f"    [{tag}] {r['wallen_taxon']:<36}"
            f" MR-PD={_fmt(r['mr_bc_pd'])}"
            f"  MR-ctrl={_fmt(r['mr_bc_ctrl'])}"
        )

    # ── Convergently validated taxa ──
    xgb_conc_set  = set(
        concordant["wallen_feature"].apply(normalize_name)
    )
    rf_overlap_set = set(
        overlap["wallen_feature"].apply(normalize_name)
    )
    rwbc_conc_set  = set(
        hub_conc["wallen_taxon"].apply(normalize_name)
    )
    convergent = xgb_conc_set & (rf_overlap_set | rwbc_conc_set)

    print(f"\n{'=' * 70}")
    print("CONVERGENTLY VALIDATED TAXA  (concordant in ≥2 comparison types)")
    print("=" * 70)
    if convergent:
        for name in sorted(convergent):
            sources = []
            if name in xgb_conc_set:   sources.append("XGB-direction")
            if name in rf_overlap_set:  sources.append("RF-overlap")
            if name in rwbc_conc_set:   sources.append("RWBC-direction")
            print(f"  *** {name.title():<42}  [{', '.join(sources)}]")
    else:
        print("  (none found — see detailed results above)")


def main() -> None:
    print("=" * 70)
    print("Cross-Cohort Biomarker Concordance Analysis")
    print("Wallen et al. 2022 (USA, n=724)  vs  Metcalfe-Roach et al. 2024 (Canada, n=276)")
    print("=" * 70)

    # ── Load Metcalfe-Roach data ──
    print("\n[1/3] Loading Metcalfe-Roach supplementary data...")
    mr_abund = load_mr_abundance()
    mr_rf    = load_mr_rf(top_n=50)
    mr_bc    = load_mr_betweenness()
    print(f"      Abundance table  : {len(mr_abund)} species rows")
    print(f"      RF top-50 feats  : {len(mr_rf)} features")
    statuses = mr_bc["Status"].dropna().unique().tolist()
    print(f"      Betweenness rows : {len(mr_bc)}  (Status values: {statuses})")

    # ── Run comparisons ──
    print("\n[2/3] Running comparisons...")
    xgb_abund_comp = compare_xgb_vs_abundance(WALLEN_XGB, mr_abund)
    xgb_rf_comp    = compare_xgb_vs_rf(WALLEN_XGB, mr_rf)
    rwbc_bc_comp   = compare_rwbc_vs_betweenness(WALLEN_RWBC, mr_bc)

    # ── Print report ──
    print_report(xgb_abund_comp, xgb_rf_comp, rwbc_bc_comp)

    # ── Save CSVs ──
    print(f"\n[3/3] Saving results to {ARTIFACTS}/...")
    ARTIFACTS.mkdir(exist_ok=True)

    xgb_abund_comp.to_csv(ARTIFACTS / "concordance_xgb_vs_mr_abundance.csv",   index=False)
    xgb_rf_comp.to_csv(   ARTIFACTS / "concordance_xgb_vs_mr_rf.csv",          index=False)
    rwbc_bc_comp.to_csv(  ARTIFACTS / "concordance_rwbc_vs_mr_betweenness.csv", index=False)

    # Aggregate summary table
    found_a   = xgb_abund_comp[xgb_abund_comp["found_in_MR"]]
    conc_a    = found_a[found_a["direction_concordant"] == True]
    ov_b      = xgb_rf_comp[xgb_rf_comp["overlap"]]
    hub_f     = rwbc_bc_comp[rwbc_bc_comp["found_in_MR_PD"] | rwbc_bc_comp["found_in_MR_ctrl"]]
    hub_c     = hub_f[hub_f["direction_concordant"] == True]

    summary = pd.DataFrame([
        {
            "comparison":          "XGB vs MR Abundance (direction)",
            "n_wallen_features":   len(WALLEN_XGB),
            "n_found_in_MR":       len(found_a),
            "n_concordant":        len(conc_a),
            "concordance_rate_pct": round(
                len(conc_a) / len(found_a) * 100 if len(found_a) > 0 else 0, 1
            ),
        },
        {
            "comparison":          "XGB vs MR RF (feature overlap)",
            "n_wallen_features":   len(WALLEN_XGB),
            "n_found_in_MR":       len(ov_b),
            "n_concordant":        len(ov_b),
            "concordance_rate_pct": round(
                len(ov_b) / len(WALLEN_XGB) * 100, 1
            ),
        },
        {
            "comparison":          "RWBC vs MR Betweenness (hub direction)",
            "n_wallen_features":   len(WALLEN_RWBC),
            "n_found_in_MR":       len(hub_f),
            "n_concordant":        len(hub_c),
            "concordance_rate_pct": round(
                len(hub_c) / len(hub_f) * 100 if len(hub_f) > 0 else 0, 1
            ),
        },
    ])
    summary.to_csv(ARTIFACTS / "concordance_summary.csv", index=False)

    for fname in [
        "concordance_xgb_vs_mr_abundance.csv",
        "concordance_xgb_vs_mr_rf.csv",
        "concordance_rwbc_vs_mr_betweenness.csv",
        "concordance_summary.csv",
    ]:
        print(f"  [saved] {fname}")

    # ── Save figure ──
    plot_concordance_summary(
        xgb_abund_comp,
        xgb_rf_comp,
        rwbc_bc_comp,
        ARTIFACTS / "concordance_figure.png",
    )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
