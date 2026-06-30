#!/usr/bin/env python3
"""Build a normal-data feature graph and rank features by random-walk BC."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "results" / "network"

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv"
    if (PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "Source_Data_24Oct2022_prepped.csv").exists()
    else PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "microbiome_V2.csv"
)
DEFAULT_CENTRALITY_OUTPUT = ARTIFACT_DIR / "random_walk_betweenness_centrality.csv"
DEFAULT_EDGES_OUTPUT = ARTIFACT_DIR / "random_walk_network_edges.csv"
DEFAULT_METADATA_OUTPUT = ARTIFACT_DIR / "random_walk_betweenness_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a feature-correlation graph on normal data and compute "
            "approximate random-walk betweenness centrality."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV}).",
    )
    parser.add_argument(
        "--feature-domain",
        choices=("taxa", "protein", "pathway", "all"),
        default="all",
        help=(
            "Feature domain to analyze: taxa (k__ columns), protein (Kxxxxx...), "
            "pathway (...PWY...), or all (default: all)."
        ),
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of samples with value > 0 for a feature to be kept "
            "for graph modeling (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=300,
        help=(
            "Keep top-N remaining features by variance for graph modeling. "
            "Use 0 for no cap (default: 300)."
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.35,
        help=(
            "Absolute Pearson correlation threshold for adding graph edges "
            "(default: 0.35)."
        ),
    )
    parser.add_argument(
        "--rw-pairs",
        type=int,
        default=2000,
        help="Number of random source-target pairs to sample (default: 2000).",
    )
    parser.add_argument(
        "--walks-per-pair",
        type=int,
        default=8,
        help="Random walks simulated per source-target pair (default: 8).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps allowed per random walk (default: 200).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for pair sampling and walk transitions (default: 42).",
    )
    parser.add_argument(
        "--output-centrality-csv",
        default=str(DEFAULT_CENTRALITY_OUTPUT),
        help=f"Output random-walk centrality CSV (default: {DEFAULT_CENTRALITY_OUTPUT}).",
    )
    parser.add_argument(
        "--output-edges-csv",
        default=str(DEFAULT_EDGES_OUTPUT),
        help=f"Output edge list CSV (default: {DEFAULT_EDGES_OUTPUT}).",
    )
    parser.add_argument(
        "--output-metadata-json",
        default=str(DEFAULT_METADATA_OUTPUT),
        help=f"Output metadata JSON (default: {DEFAULT_METADATA_OUTPUT}).",
    )
    return parser.parse_args()


def strip_domain_prefix(column: str) -> str:
    _, sep, remainder = column.partition("::")
    return remainder if sep else column


def choose_taxa_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if strip_domain_prefix(c).startswith("k__")]


def choose_protein_columns(columns: list[str]) -> list[str]:
    kegg_pattern = re.compile(r"^K\d{5}(?:\b|:)")
    return [c for c in columns if kegg_pattern.match(strip_domain_prefix(c))]


def choose_pathway_columns(columns: list[str]) -> list[str]:
    pathway_pattern = re.compile(r"^(?:[A-Z0-9-]*PWY[A-Z0-9-]*)(?:\b|:)")
    return [c for c in columns if pathway_pattern.match(strip_domain_prefix(c))]


def choose_feature_columns(columns: list[str], domain: str) -> list[str]:
    taxa_cols = choose_taxa_columns(columns)
    protein_cols = choose_protein_columns(columns)
    pathway_cols = choose_pathway_columns(columns)

    if domain == "taxa":
        return taxa_cols
    if domain == "protein":
        return protein_cols
    if domain == "pathway":
        return pathway_cols

    selected = set(taxa_cols) | set(protein_cols) | set(pathway_cols)
    return [c for c in columns if c in selected]


def coerce_numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        df.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


@dataclass
class NetworkFeatureSelection:
    values: np.ndarray
    columns: list[str]
    dropped_by_prevalence: int
    dropped_by_zero_variance: int
    dropped_by_max_features: int


def select_features_for_network(
    values: np.ndarray,
    columns: list[str],
    min_prevalence: float,
    max_features: int,
) -> NetworkFeatureSelection:
    if not 0 <= min_prevalence <= 1:
        raise ValueError("--min-prevalence must be in [0, 1].")
    if max_features < 0:
        raise ValueError("--max-features must be >= 0.")

    prevalence = np.mean(values > 0, axis=0)
    keep_prev = prevalence >= min_prevalence
    filtered_values = values[:, keep_prev]
    filtered_cols = [c for c, keep in zip(columns, keep_prev) if keep]

    if filtered_values.shape[1] == 0:
        raise ValueError(
            "No features remain after prevalence filtering. Lower --min-prevalence."
        )

    variances = np.var(filtered_values, axis=0)
    keep_var = variances > 1e-12
    filtered_values = filtered_values[:, keep_var]
    filtered_cols = [c for c, keep in zip(filtered_cols, keep_var) if keep]

    if filtered_values.shape[1] < 2:
        raise ValueError(
            "Need at least 2 non-constant features for graph modeling. "
            "Lower --min-prevalence or broaden --feature-domain."
        )

    dropped_by_max = 0
    if max_features > 0 and filtered_values.shape[1] > max_features:
        variances = np.var(filtered_values, axis=0)
        top_idx = np.argsort(variances)[::-1][:max_features]
        top_idx.sort()
        dropped_by_max = int(filtered_values.shape[1] - max_features)
        filtered_values = filtered_values[:, top_idx]
        filtered_cols = [filtered_cols[i] for i in top_idx]

    return NetworkFeatureSelection(
        values=filtered_values,
        columns=filtered_cols,
        dropped_by_prevalence=int(np.sum(~keep_prev)),
        dropped_by_zero_variance=int(np.sum(~keep_var)),
        dropped_by_max_features=dropped_by_max,
    )


@dataclass
class NetworkEdges:
    src_idx: np.ndarray
    dst_idx: np.ndarray
    corr: np.ndarray
    abs_corr: np.ndarray


def build_edges_from_correlation(
    values: np.ndarray,
    correlation_threshold: float,
) -> NetworkEdges:
    if not 0 <= correlation_threshold <= 1:
        raise ValueError("--correlation-threshold must be in [0, 1].")

    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    triu_i, triu_j = np.triu_indices(corr.shape[0], k=1)
    corr_vals = corr[triu_i, triu_j]
    abs_corr_vals = np.abs(corr_vals)
    keep = abs_corr_vals >= correlation_threshold

    return NetworkEdges(
        src_idx=triu_i[keep],
        dst_idx=triu_j[keep],
        corr=corr_vals[keep],
        abs_corr=abs_corr_vals[keep],
    )


def build_adjacency_probabilities(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    abs_corr: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    weights: list[list[float]] = [[] for _ in range(n_nodes)]

    degree = np.zeros(n_nodes, dtype=int)
    strength = np.zeros(n_nodes, dtype=float)

    for u, v, w in zip(src_idx, dst_idx, abs_corr):
        u_int = int(u)
        v_int = int(v)
        w_float = float(w)
        neighbors[u_int].append(v_int)
        neighbors[v_int].append(u_int)
        weights[u_int].append(w_float)
        weights[v_int].append(w_float)
        degree[u_int] += 1
        degree[v_int] += 1
        strength[u_int] += w_float
        strength[v_int] += w_float

    neigh_arr: list[np.ndarray] = []
    prob_arr: list[np.ndarray] = []
    for n_list, w_list in zip(neighbors, weights):
        if not n_list:
            neigh_arr.append(np.array([], dtype=int))
            prob_arr.append(np.array([], dtype=float))
            continue
        n_np = np.array(n_list, dtype=int)
        w_np = np.array(w_list, dtype=float)
        w_sum = float(w_np.sum())
        if w_sum <= 0:
            p_np = np.full_like(w_np, 1.0 / len(w_np), dtype=float)
        else:
            p_np = w_np / w_sum
        neigh_arr.append(n_np)
        prob_arr.append(p_np)

    return neigh_arr, prob_arr, degree, strength


def connected_components_from_adjacency(neighbors: list[np.ndarray]) -> list[np.ndarray]:
    n_nodes = len(neighbors)
    seen = np.zeros(n_nodes, dtype=bool)
    components: list[np.ndarray] = []

    for start in range(n_nodes):
        if seen[start]:
            continue
        if neighbors[start].size == 0:
            seen[start] = True
            continue

        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                v_int = int(v)
                if not seen[v_int]:
                    seen[v_int] = True
                    stack.append(v_int)
        if len(comp) >= 2:
            components.append(np.array(comp, dtype=int))

    return components


def sample_source_target_pairs(
    components: list[np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_pairs <= 0:
        raise ValueError("--rw-pairs must be > 0.")
    if not components:
        raise ValueError("No connected component with at least 2 nodes.")

    # Weight components by number of ordered source-target pairs.
    comp_weights = np.array(
        [len(comp) * (len(comp) - 1) for comp in components], dtype=float
    )
    comp_weights /= comp_weights.sum()

    pairs = np.zeros((n_pairs, 2), dtype=int)
    for i in range(n_pairs):
        comp = components[int(rng.choice(len(components), p=comp_weights))]
        s_idx = int(rng.integers(0, len(comp)))
        t_idx = int(rng.integers(0, len(comp) - 1))
        if t_idx >= s_idx:
            t_idx += 1
        pairs[i, 0] = int(comp[s_idx])
        pairs[i, 1] = int(comp[t_idx])
    return pairs


def random_walk_betweenness_approx(
    n_nodes: int,
    neighbors: list[np.ndarray],
    probs: list[np.ndarray],
    source_target_pairs: np.ndarray,
    walks_per_pair: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    if walks_per_pair <= 0:
        raise ValueError("--walks-per-pair must be > 0.")
    if max_steps <= 0:
        raise ValueError("--max-steps must be > 0.")

    visits = np.zeros(n_nodes, dtype=float)
    successful_walks = 0
    attempted_walks = 0

    for source, target in source_target_pairs:
        s = int(source)
        t = int(target)
        for _ in range(walks_per_pair):
            attempted_walks += 1
            cur = s
            reached = False
            for _step in range(max_steps):
                neigh = neighbors[cur]
                if neigh.size == 0:
                    break
                p = probs[cur]
                next_idx = int(rng.choice(neigh.size, p=p))
                nxt = int(neigh[next_idx])
                if nxt == t:
                    reached = True
                    break
                if nxt != s:
                    visits[nxt] += 1.0
                cur = nxt
            if reached:
                successful_walks += 1

    denom = float(successful_walks) if successful_walks > 0 else float(attempted_walks)
    if denom <= 0:
        denom = 1.0
    centrality = visits / denom
    return centrality, attempted_walks, successful_walks


def main() -> None:
    args = parse_args()
    t0 = perf_counter()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_centrality_csv = Path(args.output_centrality_csv).expanduser().resolve()
    output_edges_csv = Path(args.output_edges_csv).expanduser().resolve()
    output_metadata_json = Path(args.output_metadata_json).expanduser().resolve()

    output_centrality_csv.parent.mkdir(parents=True, exist_ok=True)
    output_edges_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded shape: {df.shape}")

    feature_columns = choose_feature_columns(df.columns.tolist(), args.feature_domain)
    if not feature_columns:
        raise ValueError(f"No features found for --feature-domain={args.feature_domain!r}.")
    print(
        f"Selected {len(feature_columns)} '{args.feature_domain}' features "
        "for normal-data graph modeling."
    )

    values = coerce_numeric_matrix(df, feature_columns)

    selection = select_features_for_network(
        values=values,
        columns=feature_columns,
        min_prevalence=args.min_prevalence,
        max_features=args.max_features,
    )
    print(
        "Network feature filtering:"
        f" dropped_prevalence={selection.dropped_by_prevalence},"
        f" dropped_zero_variance={selection.dropped_by_zero_variance},"
        f" dropped_max_features={selection.dropped_by_max_features},"
        f" kept={len(selection.columns)}"
    )

    edges = build_edges_from_correlation(
        values=selection.values,
        correlation_threshold=args.correlation_threshold,
    )
    if edges.src_idx.size == 0:
        raise ValueError(
            "No edges passed --correlation-threshold. Lower the threshold "
            "(e.g., 0.2 to 0.3)."
        )
    print(f"Network edges kept: {edges.src_idx.size}")

    neighbors, probs, degree, strength = build_adjacency_probabilities(
        n_nodes=len(selection.columns),
        src_idx=edges.src_idx,
        dst_idx=edges.dst_idx,
        abs_corr=edges.abs_corr,
    )

    components = connected_components_from_adjacency(neighbors)
    print(f"Connected components (size>=2): {len(components)}")

    rng = np.random.default_rng(args.random_seed)
    pairs = sample_source_target_pairs(
        components=components,
        n_pairs=args.rw_pairs,
        rng=rng,
    )

    rw_centrality, attempted_walks, successful_walks = random_walk_betweenness_approx(
        n_nodes=len(selection.columns),
        neighbors=neighbors,
        probs=probs,
        source_target_pairs=pairs,
        walks_per_pair=args.walks_per_pair,
        max_steps=args.max_steps,
        rng=rng,
    )
    print(
        "Random-walk simulation:"
        f" pairs={len(pairs)}, attempted_walks={attempted_walks},"
        f" successful_walks={successful_walks}"
    )

    centrality_df = pd.DataFrame(
        {
            "feature": selection.columns,
            "degree": degree,
            "strength_abs_corr_sum": strength,
            "random_walk_betweenness": rw_centrality,
        }
    ).sort_values(
        by=["random_walk_betweenness", "degree", "strength_abs_corr_sum"],
        ascending=False,
    )
    centrality_df.insert(0, "rank", np.arange(1, len(centrality_df) + 1))
    centrality_df.to_csv(output_centrality_csv, index=False)

    edges_df = pd.DataFrame(
        {
            "feature_a": [selection.columns[i] for i in edges.src_idx],
            "feature_b": [selection.columns[j] for j in edges.dst_idx],
            "correlation": edges.corr,
            "abs_correlation": edges.abs_corr,
        }
    ).sort_values(by="abs_correlation", ascending=False)
    edges_df.to_csv(output_edges_csv, index=False)

    runtime_s = perf_counter() - t0
    metadata = {
        "input_csv": str(input_csv),
        "output_centrality_csv": str(output_centrality_csv),
        "output_edges_csv": str(output_edges_csv),
        "n_rows": int(df.shape[0]),
        "n_total_columns": int(df.shape[1]),
        "feature_domain": args.feature_domain,
        "n_features_initial": int(len(feature_columns)),
        "n_features_used_for_network": int(len(selection.columns)),
        "dropped_by_prevalence": int(selection.dropped_by_prevalence),
        "dropped_by_zero_variance": int(selection.dropped_by_zero_variance),
        "dropped_by_max_features": int(selection.dropped_by_max_features),
        "correlation_threshold": float(args.correlation_threshold),
        "edge_count": int(edges.src_idx.size),
        "rw_pairs": int(args.rw_pairs),
        "walks_per_pair": int(args.walks_per_pair),
        "max_steps": int(args.max_steps),
        "attempted_walks": int(attempted_walks),
        "successful_walks": int(successful_walks),
        "random_seed": int(args.random_seed),
        "runtime_seconds": float(runtime_s),
    }
    output_metadata_json.write_text(json.dumps(metadata, indent=2))

    print(f"Saved random-walk centrality CSV: {output_centrality_csv}")
    print(f"Saved edges CSV: {output_edges_csv}")
    print(f"Saved metadata JSON: {output_metadata_json}")
    print(f"Runtime: {runtime_s:.2f}s")
    print("\nTop 20 features by random-walk betweenness:")
    for row in centrality_df.head(20).itertuples(index=False):
        print(
            f"{row.rank:>2}. {row.feature} | "
            f"rw_bc={row.random_walk_betweenness:.6f} | degree={row.degree}"
        )


if __name__ == "__main__":
    main()
