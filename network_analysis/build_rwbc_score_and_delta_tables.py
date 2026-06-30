#!/usr/bin/env python3
"""Build RWBC score and delta-RWBC tables from PD/healthy centrality outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PD_RWBC_CSV = PROJECT_ROOT / "results" / "network" / "coabundance_rwbc_pd_centrality.csv"
DEFAULT_HEALTHY_RWBC_CSV = PROJECT_ROOT / "results" / "network" / "coabundance_rwbc_healthy_centrality.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "statistics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create top-10 RWBC score table and delta-RWBC table "
            "(delta = PD RWBC - Healthy RWBC)."
        )
    )
    parser.add_argument(
        "--pd-rwbc-csv",
        default=str(DEFAULT_PD_RWBC_CSV),
        help=f"PD RWBC CSV (default: {DEFAULT_PD_RWBC_CSV}).",
    )
    parser.add_argument(
        "--healthy-rwbc-csv",
        default=str(DEFAULT_HEALTHY_RWBC_CSV),
        help=f"Healthy RWBC CSV (default: {DEFAULT_HEALTHY_RWBC_CSV}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of rows for top tables (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def strip_sheet_prefix(name: str) -> str:
    text = str(name)
    _, sep, remainder = text.partition("::")
    return remainder if sep else text


def load_rwbc(path: Path, group_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{group_name} RWBC CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"feature", "random_walk_betweenness"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{group_name} RWBC CSV missing columns: {sorted(missing)}")
    if "degree" not in df.columns:
        df["degree"] = np.nan
    df = df.copy()
    df["feature"] = df["feature"].astype(str)
    df["random_walk_betweenness"] = pd.to_numeric(
        df["random_walk_betweenness"], errors="coerce"
    ).fillna(0.0)
    df["degree"] = pd.to_numeric(df["degree"], errors="coerce")
    df = df.sort_values(
        ["random_walk_betweenness", "degree", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1.")

    pd_path = Path(args.pd_rwbc_csv).expanduser().resolve()
    healthy_path = Path(args.healthy_rwbc_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pd_df = load_rwbc(pd_path, "PD")
    healthy_df = load_rwbc(healthy_path, "Healthy")

    # Table A: top-N RWBC scores (PD ranking).
    score_top = pd_df.head(args.top_n).copy()
    score_top["feature_clean"] = score_top["feature"].map(strip_sheet_prefix)
    score_top = score_top.rename(columns={"rank": "rwbc_rank", "random_walk_betweenness": "rwbc_score"})

    score_full_csv = output_dir / "rwbc_top20_feature_scores.csv"
    score_simple_csv = output_dir / "rwbc_top20_feature_scores_simple.csv"
    score_top[
        ["rwbc_rank", "feature", "feature_clean", "rwbc_score", "degree"]
    ].to_csv(score_full_csv, index=False)
    score_top[["rwbc_rank", "feature_clean", "rwbc_score"]].to_csv(score_simple_csv, index=False)

    # Table B: delta RWBC table across all shared features.
    pd_small = pd_df[["feature", "rank", "random_walk_betweenness", "degree"]].rename(
        columns={
            "rank": "pd_rank",
            "random_walk_betweenness": "rwbc_pd",
            "degree": "degree_pd",
        }
    )
    healthy_small = healthy_df[["feature", "rank", "random_walk_betweenness", "degree"]].rename(
        columns={
            "rank": "healthy_rank",
            "random_walk_betweenness": "rwbc_healthy",
            "degree": "degree_healthy",
        }
    )
    delta = pd.merge(pd_small, healthy_small, on="feature", how="outer")
    delta["feature_clean"] = delta["feature"].map(strip_sheet_prefix)
    delta["rwbc_pd"] = pd.to_numeric(delta["rwbc_pd"], errors="coerce").fillna(0.0)
    delta["rwbc_healthy"] = pd.to_numeric(delta["rwbc_healthy"], errors="coerce").fillna(0.0)
    delta["delta_rwbc"] = delta["rwbc_pd"] - delta["rwbc_healthy"]
    delta["abs_delta_rwbc"] = delta["delta_rwbc"].abs()
    delta["delta_direction"] = np.where(
        delta["delta_rwbc"] > 0,
        "higher_in_PD",
        np.where(delta["delta_rwbc"] < 0, "higher_in_healthy", "equal"),
    )
    delta = delta.sort_values(
        ["abs_delta_rwbc", "delta_rwbc", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    delta["delta_rank"] = np.arange(1, len(delta) + 1)

    delta_full_csv = output_dir / "rwbc_delta_table.csv"
    delta_simple_csv = output_dir / "rwbc_delta_top20_simple.csv"
    delta[
        [
            "delta_rank",
            "feature",
            "feature_clean",
            "pd_rank",
            "healthy_rank",
            "rwbc_pd",
            "rwbc_healthy",
            "delta_rwbc",
            "abs_delta_rwbc",
            "delta_direction",
            "degree_pd",
            "degree_healthy",
        ]
    ].to_csv(delta_full_csv, index=False)
    delta.head(args.top_n)[["delta_rank", "feature_clean", "delta_rwbc"]].to_csv(
        delta_simple_csv, index=False
    )

    print("=" * 72)
    print("RWBC Score and Delta-RWBC Tables")
    print("=" * 72)
    print(f"PD RWBC CSV: {pd_path}")
    print(f"Healthy RWBC CSV: {healthy_path}")
    print(f"Saved: {score_full_csv}")
    print(f"Saved: {score_simple_csv}")
    print(f"Saved: {delta_full_csv}")
    print(f"Saved: {delta_simple_csv}")


if __name__ == "__main__":
    main()
