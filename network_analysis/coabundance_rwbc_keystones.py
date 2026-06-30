#!/usr/bin/env python3
"""Compute random-walk betweenness keystones on co-abundance networks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# Keep Matplotlib cache writable in restricted environments.
_TMP_DIR = Path(tempfile.gettempdir())
_MPL_CACHE = _TMP_DIR / "mplconfig_science_fair"
_FONT_CACHE = _TMP_DIR / "xdg_cache_science_fair"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_FONT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_FONT_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from random_walk_betweenness import (
    build_adjacency_probabilities,
    connected_components_from_adjacency,
    random_walk_betweenness_approx,
    sample_source_target_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "results" / "network"
DEFAULT_TOP_FEATURES_CSV = DEFAULT_ARTIFACT_DIR / "coabundance_top_features_full_xg.csv"
DEFAULT_PD_EDGES_CSV = DEFAULT_ARTIFACT_DIR / "coabundance_pd_edges_full_xg.csv"
DEFAULT_HEALTHY_EDGES_CSV = DEFAULT_ARTIFACT_DIR / "coabundance_healthy_edges_full_xg.csv"
DEFAULT_NETWORK_FIGURE = DEFAULT_ARTIFACT_DIR / "coabundance_network_rwbc_all_features.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run random-walk betweenness centrality on co-abundance networks and "
            "extract top keystone features."
        )
    )
    parser.add_argument(
        "--top-features-csv",
        default=str(DEFAULT_TOP_FEATURES_CSV),
        help=f"CSV with all network nodes (default: {DEFAULT_TOP_FEATURES_CSV}).",
    )
    parser.add_argument(
        "--pd-edges-csv",
        default=str(DEFAULT_PD_EDGES_CSV),
        help=f"PD edge list CSV (default: {DEFAULT_PD_EDGES_CSV}).",
    )
    parser.add_argument(
        "--healthy-edges-csv",
        default=str(DEFAULT_HEALTHY_EDGES_CSV),
        help=f"Healthy edge list CSV (default: {DEFAULT_HEALTHY_EDGES_CSV}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top keystones to report per network (default: 20).",
    )
    parser.add_argument(
        "--rw-pairs",
        type=int,
        default=3000,
        help="Random source-target pairs sampled for random walks (default: 3000).",
    )
    parser.add_argument(
        "--walks-per-pair",
        type=int,
        default=12,
        help="Random walks per source-target pair (default: 12).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per random walk (default: 200).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help=f"Directory for output artifacts (default: {DEFAULT_ARTIFACT_DIR}).",
    )
    parser.add_argument(
        "--output-figure",
        default="",
        help=(
            "Optional output PNG for the all-features RWBC network figure. "
            f"If empty, defaults to {DEFAULT_NETWORK_FIGURE}."
        ),
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=80,
        help=(
            "Max labels to draw by RWBC rank per network when node count is large "
            "(default: 80)."
        ),
    )
    return parser.parse_args()


def load_nodes(top_features_csv: Path) -> list[str]:
    df = pd.read_csv(top_features_csv)
    if "feature" not in df.columns:
        raise ValueError(f"Expected 'feature' column in {top_features_csv}.")
    features = df["feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"No features found in {top_features_csv}.")
    return features


def load_edges(edges_csv: Path) -> pd.DataFrame:
    edges = pd.read_csv(edges_csv)
    required = {"feature_a", "feature_b", "abs_correlation"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing columns in {edges_csv}: {sorted(missing)}")
    edges = edges.copy()
    edges["feature_a"] = edges["feature_a"].astype(str)
    edges["feature_b"] = edges["feature_b"].astype(str)
    edges["abs_correlation"] = pd.to_numeric(edges["abs_correlation"], errors="coerce").fillna(0.0)
    edges = edges[edges["abs_correlation"] > 0].reset_index(drop=True)
    return edges


def shorten_label(name: str, max_len: int = 36) -> str:
    text = str(name)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_graph(nodes: list[str], edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        a = str(row.feature_a)
        b = str(row.feature_b)
        if a == b:
            continue
        corr = float(row.abs_correlation)
        sign = "positive"
        if hasattr(row, "sign"):
            sign = str(row.sign)
        graph.add_edge(a, b, weight=abs(corr), sign=sign)
    return graph


def draw_rwbc_network(
    ax: plt.Axes,
    graph: nx.Graph,
    centrality_df: pd.DataFrame,
    title: str,
    color_norm_max: float,
    label_top_k: int,
    seed: int = 42,
) -> None:
    nodes = list(graph.nodes())
    if not nodes:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    c_map = (
        centrality_df.set_index("feature")["random_walk_betweenness"].astype(float).to_dict()
        if not centrality_df.empty
        else {}
    )
    centrality_vals = np.array([float(c_map.get(node, 0.0)) for node in nodes], dtype=float)
    c_max = float(max(color_norm_max, 1e-12))
    scaled = np.clip(centrality_vals / c_max, 0.0, 1.0)
    node_sizes = (250.0 + 1700.0 * np.sqrt(scaled)).tolist()

    n_nodes = max(len(nodes), 2)
    k_value = max(0.75, 3.1 / np.sqrt(n_nodes))
    pos = nx.spring_layout(
        graph,
        seed=seed,
        k=k_value,
        weight="weight",
        iterations=420,
    )

    edges = list(graph.edges(data=True))
    edge_widths = [max(0.4, float(d.get("weight", 0.0)) * 3.2) for _, _, d in edges]
    edge_colors = [
        "#2AA876" if str(d.get("sign", "positive")) == "positive" else "#D48A41"
        for _, _, d in edges
    ]

    edge_collection = nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.35,
        ax=ax,
    )
    if edge_collection is not None:
        edge_collection.set_zorder(1)

    node_collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=centrality_vals,
        cmap=plt.cm.viridis,
        vmin=0.0,
        vmax=c_max,
        node_size=node_sizes,
        linewidths=0.6,
        edgecolors="white",
        alpha=0.95,
        ax=ax,
    )
    node_collection.set_zorder(2)

    labels_to_draw: list[str]
    if len(nodes) <= label_top_k:
        labels_to_draw = nodes
    else:
        ordered = centrality_df["feature"].astype(str).tolist()
        labels_to_draw = ordered[:label_top_k]
    label_dict = {n: shorten_label(n) for n in labels_to_draw if n in pos}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=label_dict,
        font_size=6.0,
        font_weight="bold",
        font_color="#111827",
        ax=ax,
    )

    ax.set_title(
        f"{title}\nNodes={graph.number_of_nodes()} | Edges={graph.number_of_edges()}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.axis("off")


def save_rwbc_network_figure(
    nodes: list[str],
    pd_edges: pd.DataFrame,
    healthy_edges: pd.DataFrame,
    pd_df: pd.DataFrame,
    healthy_df: pd.DataFrame,
    output_figure: Path,
    label_top_k: int,
    seed: int = 42,
) -> None:
    pd_graph = build_graph(nodes, pd_edges)
    healthy_graph = build_graph(nodes, healthy_edges)

    pd_vals = pd_df["random_walk_betweenness"].astype(float).to_numpy()
    healthy_vals = healthy_df["random_walk_betweenness"].astype(float).to_numpy()
    color_norm_max = float(max(pd_vals.max(initial=0.0), healthy_vals.max(initial=0.0)))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 10),
        facecolor="white",
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_facecolor("white")

    draw_rwbc_network(
        ax=axes[0],
        graph=pd_graph,
        centrality_df=pd_df,
        title="PD Network (RWBC)",
        color_norm_max=color_norm_max,
        label_top_k=label_top_k,
        seed=seed,
    )
    draw_rwbc_network(
        ax=axes[1],
        graph=healthy_graph,
        centrality_df=healthy_df,
        title="Healthy Network (RWBC)",
        color_norm_max=color_norm_max,
        label_top_k=label_top_k,
        seed=seed + 11,
    )

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_clim(0.0, max(color_norm_max, 1e-12))
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        pad=0.015,
        fraction=0.03,
    )
    cbar.set_label("Random-walk betweenness", fontsize=11)

    fig.suptitle(
        "Co-Abundance Networks With RWBC (All Features in Network)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_rwbc_for_network(
    nodes: list[str],
    edges: pd.DataFrame,
    rw_pairs: int,
    walks_per_pair: int,
    max_steps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)

    src_idx_list: list[int] = []
    dst_idx_list: list[int] = []
    abs_corr_list: list[float] = []
    for row in edges.itertuples(index=False):
        a = row.feature_a
        b = row.feature_b
        if a not in node_to_idx or b not in node_to_idx:
            continue
        if a == b:
            continue
        src_idx_list.append(node_to_idx[a])
        dst_idx_list.append(node_to_idx[b])
        abs_corr_list.append(float(row.abs_correlation))

    if src_idx_list:
        src_idx = np.asarray(src_idx_list, dtype=int)
        dst_idx = np.asarray(dst_idx_list, dtype=int)
        abs_corr = np.asarray(abs_corr_list, dtype=float)
    else:
        src_idx = np.array([], dtype=int)
        dst_idx = np.array([], dtype=int)
        abs_corr = np.array([], dtype=float)

    neighbors, probs, degree, strength = build_adjacency_probabilities(
        n_nodes=n_nodes,
        src_idx=src_idx,
        dst_idx=dst_idx,
        abs_corr=abs_corr,
    )
    components = connected_components_from_adjacency(neighbors)

    attempted_walks = 0
    successful_walks = 0
    if components:
        rng = np.random.default_rng(random_seed)
        pairs = sample_source_target_pairs(components=components, n_pairs=rw_pairs, rng=rng)
        rw_centrality, attempted_walks, successful_walks = random_walk_betweenness_approx(
            n_nodes=n_nodes,
            neighbors=neighbors,
            probs=probs,
            source_target_pairs=pairs,
            walks_per_pair=walks_per_pair,
            max_steps=max_steps,
            rng=rng,
        )
    else:
        rw_centrality = np.zeros(n_nodes, dtype=float)

    centrality_df = pd.DataFrame(
        {
            "feature": nodes,
            "degree": degree,
            "strength_abs_corr_sum": strength,
            "random_walk_betweenness": rw_centrality,
        }
    ).sort_values(
        by=["random_walk_betweenness", "degree", "strength_abs_corr_sum", "feature"],
        ascending=[False, False, False, True],
    )
    centrality_df.insert(0, "rank", np.arange(1, len(centrality_df) + 1))

    metadata = {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(src_idx)),
        "n_components_size_ge_2": int(len(components)),
        "attempted_walks": int(attempted_walks),
        "successful_walks": int(successful_walks),
        "rw_pairs": int(rw_pairs),
        "walks_per_pair": int(walks_per_pair),
        "max_steps": int(max_steps),
        "random_seed": int(random_seed),
    }
    return centrality_df, metadata


def save_outputs(
    output_dir: Path,
    pd_df: pd.DataFrame,
    healthy_df: pd.DataFrame,
    top_k: int,
    metadata: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd_csv = output_dir / "coabundance_rwbc_pd_centrality.csv"
    healthy_csv = output_dir / "coabundance_rwbc_healthy_centrality.csv"
    top20_csv = output_dir / "coabundance_rwbc_top20_keystones.csv"
    summary_json = output_dir / "coabundance_rwbc_summary.json"

    pd_df.to_csv(pd_csv, index=False)
    healthy_df.to_csv(healthy_csv, index=False)

    top_pd = pd_df.head(top_k).copy()
    top_pd["network"] = "PD"
    top_healthy = healthy_df.head(top_k).copy()
    top_healthy["network"] = "Healthy"
    out_cols = [
        "network",
        "rank",
        "feature",
        "random_walk_betweenness",
        "degree",
        "strength_abs_corr_sum",
    ]
    pd.concat([top_pd, top_healthy], ignore_index=True)[out_cols].to_csv(top20_csv, index=False)
    summary_json.write_text(json.dumps(metadata, indent=2))

    return pd_csv, healthy_csv, top20_csv, summary_json


def print_top_keystones(title: str, df: pd.DataFrame, top_k: int) -> None:
    print(f"\nTop {top_k} keystones ({title} network):")
    for row in df.head(top_k).itertuples(index=False):
        print(
            f"{int(row.rank):>2}. {row.feature} | "
            f"rw_bc={float(row.random_walk_betweenness):.6f} | "
            f"degree={int(row.degree)}"
        )


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")
    if args.rw_pairs < 1:
        raise ValueError("--rw-pairs must be >= 1.")
    if args.walks_per_pair < 1:
        raise ValueError("--walks-per-pair must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")

    top_features_csv = Path(args.top_features_csv).expanduser().resolve()
    pd_edges_csv = Path(args.pd_edges_csv).expanduser().resolve()
    healthy_edges_csv = Path(args.healthy_edges_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_figure = (
        Path(args.output_figure).expanduser().resolve()
        if str(args.output_figure).strip()
        else (output_dir / DEFAULT_NETWORK_FIGURE.name)
    )

    nodes = load_nodes(top_features_csv)
    pd_edges = load_edges(pd_edges_csv)
    healthy_edges = load_edges(healthy_edges_csv)

    print("=" * 72)
    print("Random-Walk Betweenness on Co-Abundance Networks")
    print("=" * 72)
    print(f"Nodes from: {top_features_csv}")
    print(f"PD edges from: {pd_edges_csv}")
    print(f"Healthy edges from: {healthy_edges_csv}")
    print(f"Nodes: {len(nodes)}, PD edges: {len(pd_edges)}, Healthy edges: {len(healthy_edges)}")

    pd_df, pd_meta = run_rwbc_for_network(
        nodes=nodes,
        edges=pd_edges,
        rw_pairs=args.rw_pairs,
        walks_per_pair=args.walks_per_pair,
        max_steps=args.max_steps,
        random_seed=args.random_seed,
    )
    healthy_df, healthy_meta = run_rwbc_for_network(
        nodes=nodes,
        edges=healthy_edges,
        rw_pairs=args.rw_pairs,
        walks_per_pair=args.walks_per_pair,
        max_steps=args.max_steps,
        random_seed=args.random_seed,
    )

    metadata = {
        "top_features_csv": str(top_features_csv),
        "pd_edges_csv": str(pd_edges_csv),
        "healthy_edges_csv": str(healthy_edges_csv),
        "top_k": int(args.top_k),
        "label_top_k": int(args.label_top_k),
        "rwbc_network_figure_png": str(output_figure),
        "pd_network": pd_meta,
        "healthy_network": healthy_meta,
    }
    pd_csv, healthy_csv, top20_csv, summary_json = save_outputs(
        output_dir=output_dir,
        pd_df=pd_df,
        healthy_df=healthy_df,
        top_k=args.top_k,
        metadata=metadata,
    )
    save_rwbc_network_figure(
        nodes=nodes,
        pd_edges=pd_edges,
        healthy_edges=healthy_edges,
        pd_df=pd_df,
        healthy_df=healthy_df,
        output_figure=output_figure,
        label_top_k=max(1, int(args.label_top_k)),
        seed=int(args.random_seed),
    )

    print_top_keystones("PD", pd_df, args.top_k)
    print_top_keystones("Healthy", healthy_df, args.top_k)

    print("\nSaved:")
    print(f"  - {pd_csv}")
    print(f"  - {healthy_csv}")
    print(f"  - {top20_csv}")
    print(f"  - {output_figure}")
    print(f"  - {summary_json}")


if __name__ == "__main__":
    main()
