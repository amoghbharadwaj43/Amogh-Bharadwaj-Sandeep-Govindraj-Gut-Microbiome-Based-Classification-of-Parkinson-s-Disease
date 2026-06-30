#!/usr/bin/env python3
"""Build PD vs healthy co-abundance networks from all CLR-transformed taxonomic features."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Avoid Matplotlib/fontconfig cache warnings in restricted environments.
_TMP_DIR = Path(tempfile.gettempdir())
_MPL_CACHE = _TMP_DIR / "mplconfig_science_fair"
_FONT_CACHE = _TMP_DIR / "xdg_cache_science_fair"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_FONT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_FONT_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "microbiome_V2_clr.csv"
DEFAULT_TARGET_COLUMN = "Case_status_PD"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "network"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build separate PD vs healthy Spearman co-abundance networks "
            "using all CLR-transformed taxonomic features."
        )
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help=f"Input CLR CSV path (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help=f"Binary target column (default: {DEFAULT_TARGET_COLUMN}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--feature-domain",
        choices=("taxa", "all"),
        default="taxa",
        help="Feature domain for networking (default: taxa).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.55,
        help="Minimum absolute Spearman r to draw an edge (default: 0.55).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for layouts (default: 42).",
    )
    return parser.parse_args()


def strip_sheet_prefix(name: str) -> str:
    text = str(name)
    _, sep, remainder = text.partition("::")
    return remainder if sep else text


def source_tag(name: str) -> str:
    prefix = str(name).partition("::")[0]
    mapping = {
        "metaphlan_counts": "counts",
        "metaphlan_rel_ab": "rel_ab",
        "humann_KO_group_counts": "ko",
        "humann_pathway_counts": "pathway",
    }
    return mapping.get(prefix, prefix)


def is_taxonomic_feature(name: str) -> bool:
    return strip_sheet_prefix(name).startswith("k__")


def build_display_name_map(features: list[str]) -> dict[str, str]:
    base_names = {f: strip_sheet_prefix(f) for f in features}
    base_counts = pd.Series(list(base_names.values())).value_counts().to_dict()

    output: dict[str, str] = {}
    used: set[str] = set()
    for feature in features:
        base = base_names[feature]
        if base_counts.get(base, 0) > 1:
            candidate = f"{base} [{source_tag(feature)}]"
        else:
            candidate = base

        if candidate in used:
            cursor = 2
            while f"{candidate} ({cursor})" in used:
                cursor += 1
            candidate = f"{candidate} ({cursor})"
        used.add(candidate)
        output[feature] = candidate
    return output


def load_and_prepare_data(
    data_path: Path,
    target_column: str,
    feature_domain: str,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {data_path}.")

    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Target column '{target_column}' contains non-numeric values.")
    y = y.astype(int)
    unique_vals = set(y.unique().tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must be binary 0/1. Found values: {sorted(unique_vals)}"
        )

    X = df.drop(columns=[target_column])
    exclude_substrings = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")
    exclude_cols = [
        col for col in X.columns if any(substr in col.lower() for substr in exclude_substrings)
    ]
    if exclude_cols:
        X = X.drop(columns=exclude_cols)

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)

    X = X.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = [col for col in X.columns if X[col].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    X = X.fillna(X.median(numeric_only=True))

    if feature_domain == "taxa":
        taxa_cols = [col for col in X.columns if is_taxonomic_feature(col)]
        if not taxa_cols:
            raise ValueError("No taxonomic features found after preprocessing.")
        X = X.loc[:, taxa_cols].copy()

    if X.shape[1] < 2:
        raise ValueError("Need at least 2 numeric features after preprocessing.")

    return X, y


def build_correlation_matrix(df_subset: pd.DataFrame) -> pd.DataFrame:
    if df_subset.shape[1] == 1:
        only_col = df_subset.columns[0]
        return pd.DataFrame([[1.0]], index=[only_col], columns=[only_col])

    corr_matrix, _ = spearmanr(df_subset.to_numpy(dtype=float), axis=0)
    corr_df = pd.DataFrame(corr_matrix, index=df_subset.columns, columns=df_subset.columns)
    corr_df = corr_df.fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)
    return corr_df


def build_network(corr_df: pd.DataFrame, threshold: float) -> nx.Graph:
    graph = nx.Graph()
    cols = corr_df.columns.tolist()
    graph.add_nodes_from(cols)

    for i, feature_a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            feature_b = cols[j]
            corr = float(corr_df.iat[i, j])
            if abs(corr) >= threshold:
                graph.add_edge(
                    feature_a,
                    feature_b,
                    weight=abs(corr),
                    correlation=corr,
                    sign="positive" if corr >= 0 else "negative",
                )
    return graph


def shorten_label(name: str, max_len: int = 28) -> str:
    text = str(name)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def get_node_color_map(features: list[str], X_pd: pd.DataFrame, X_healthy: pd.DataFrame) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for feature in features:
        pd_mean = float(X_pd[feature].mean())
        healthy_mean = float(X_healthy[feature].mean())
        color_map[feature] = "#E74C3C" if pd_mean > healthy_mean else "#2980B9"
    return color_map


def draw_network(
    graph: nx.Graph,
    color_map: dict[str, str],
    title: str,
    ax: plt.Axes,
    importance_series: pd.Series | None = None,
    seed: int = 42,
) -> None:
    if graph.number_of_nodes() == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
        ax.axis("off")
        return

    nodes = list(graph.nodes())
    if importance_series is not None:
        node_sizes = [max(220.0, float(importance_series.get(n, 0.0)) * 7000.0) for n in nodes]
    else:
        node_sizes = [360.0] * len(nodes)

    edge_data = list(graph.edges(data=True))
    edge_widths = [max(0.7, float(data["weight"]) * 3.8) for _, _, data in edge_data]
    edge_colors = [
        "#27AE60" if data.get("sign", "positive") == "positive" else "#E67E22"
        for _, _, data in edge_data
    ]
    node_colors = [color_map.get(n, "#6B7280") for n in nodes]

    # Use an aggressive spring layout first, then a small repulsion pass to
    # enforce readable spacing between nearby points.
    n_nodes = max(len(nodes), 2)
    k_value = max(0.95, 3.4 / np.sqrt(n_nodes))
    pos = nx.spring_layout(
        graph,
        seed=seed,
        k=k_value,
        iterations=500,
        weight="weight",
    )

    xy = np.array([pos[n] for n in nodes], dtype=float)
    if xy.size > 0 and len(nodes) > 2:
        min_dist = 0.24 if len(nodes) <= 30 else 0.16
        for _ in range(140):
            moved = False
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    delta = xy[j] - xy[i]
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-8:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = delta / dist
                    if dist < min_dist:
                        step = (min_dist - dist) * 0.42
                        xy[i] -= direction * step
                        xy[j] += direction * step
                        moved = True
            # Mild contraction keeps points from drifting too far.
            xy *= 0.998
            if not moved:
                break

        xy -= xy.mean(axis=0, keepdims=True)
        max_abs = float(np.max(np.abs(xy)))
        if max_abs > 0:
            xy = (xy / max_abs) * 0.97
        pos = {node: xy[idx] for idx, node in enumerate(nodes)}

    labels = {n: shorten_label(n) for n in nodes}

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        ax=ax,
    )
    text_items = nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=6.0,
        font_weight="bold",
        ax=ax,
        clip_on=True,
    )
    for text in text_items.values():
        text.set_clip_on(True)
        text.set_clip_path(ax.patch)
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.10, 1.10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")


def summarize_network(graph: nx.Graph) -> dict[str, int | float | str]:
    summary: dict[str, int | float | str] = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "components": 0,
        "hub_feature": "",
        "hub_degree": 0,
    }
    if graph.number_of_nodes() == 0:
        return summary

    degrees = dict(graph.degree())
    hub_feature = max(degrees, key=degrees.get)
    summary["hub_feature"] = str(hub_feature)
    summary["hub_degree"] = int(degrees[hub_feature])
    summary["components"] = int(nx.number_connected_components(graph))
    if graph.number_of_edges() > 0:
        strengths = [float(d.get("weight", 0.0)) for _, _, d in graph.edges(data=True)]
        summary["mean_abs_corr"] = float(np.mean(strengths))
    else:
        summary["mean_abs_corr"] = 0.0
    return summary


def normalized_edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {tuple(sorted((str(u), str(v)))) for u, v in graph.edges()}


def edge_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, data in graph.edges(data=True):
        a, b = sorted((str(u), str(v)))
        corr = float(data.get("correlation", 0.0))
        rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "sign": data.get("sign", "positive"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "correlation", "abs_correlation", "sign"]
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False, ignore_index=True)


def differential_edge_frame(
    pd_graph: nx.Graph, healthy_graph: nx.Graph
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pd_only = normalized_edge_set(pd_graph) - normalized_edge_set(healthy_graph)
    healthy_only = normalized_edge_set(healthy_graph) - normalized_edge_set(pd_graph)

    pd_rows = []
    for a, b in sorted(pd_only):
        edge_data = pd_graph.get_edge_data(a, b) or pd_graph.get_edge_data(b, a) or {}
        corr = float(edge_data.get("correlation", 0.0))
        pd_rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "sign": edge_data.get("sign", "positive"),
                "group": "PD_only",
            }
        )

    healthy_rows = []
    for a, b in sorted(healthy_only):
        edge_data = healthy_graph.get_edge_data(a, b) or healthy_graph.get_edge_data(b, a) or {}
        corr = float(edge_data.get("correlation", 0.0))
        healthy_rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "sign": edge_data.get("sign", "positive"),
                "group": "Healthy_only",
            }
        )

    pd_df = pd.DataFrame(pd_rows)
    if not pd_df.empty:
        pd_df = pd_df.sort_values("abs_correlation", ascending=False, ignore_index=True)
    else:
        pd_df = pd.DataFrame(
            columns=["feature_a", "feature_b", "correlation", "abs_correlation", "sign", "group"]
        )

    healthy_df = pd.DataFrame(healthy_rows)
    if not healthy_df.empty:
        healthy_df = healthy_df.sort_values("abs_correlation", ascending=False, ignore_index=True)
    else:
        healthy_df = pd.DataFrame(
            columns=["feature_a", "feature_b", "correlation", "abs_correlation", "sign", "group"]
        )
    return pd_df, healthy_df


def save_main_figure(
    pd_graph: nx.Graph,
    healthy_graph: nx.Graph,
    color_map: dict[str, str],
    importances: pd.Series | None,
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor="white")
    fig.suptitle(
        "Gut Feature Co-Abundance Networks: PD vs Healthy (All CLR Features)",
        fontsize=15,
        fontweight="bold",
        color="black",
        y=1.01,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.title.set_color("black")

    draw_network(
        pd_graph,
        color_map=color_map,
        title="PD Patients\nFeature Co-Abundance Network",
        ax=ax1,
        importance_series=importances,
    )
    draw_network(
        healthy_graph,
        color_map=color_map,
        title="Healthy Controls\nFeature Co-Abundance Network",
        ax=ax2,
        importance_series=importances,
    )

    legend_elements = [
        mpatches.Patch(facecolor="#E74C3C", label="Enriched in PD"),
        mpatches.Patch(facecolor="#2980B9", label="Enriched in Healthy"),
        mpatches.Patch(facecolor="#27AE60", label="Positive co-abundance"),
        mpatches.Patch(facecolor="#E67E22", label="Negative co-abundance"),
    ]
    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        facecolor="#F7F7F7",
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.06),
    )
    for text in legend.get_texts():
        text.set_color("black")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= args.corr_threshold <= 1.0:
        raise ValueError("--corr-threshold must be between 0 and 1.")

    print("=" * 72)
    print("Co-Abundance Network Build: all CLR features")
    print("=" * 72)
    print(f"Data path: {data_path}")
    print(f"Target column: {args.target_column}")
    print(f"Feature domain: {args.feature_domain}")
    print(f"Output dir: {output_dir}")

    X, y = load_and_prepare_data(data_path, args.target_column, args.feature_domain)
    print(f"Loaded matrix: {X.shape[0]} samples x {X.shape[1]} numeric features")
    print(f"Class counts: healthy={int((y == 0).sum())}, PD={int((y == 1).sum())}")

    all_features = X.columns.tolist()
    display_name_map = build_display_name_map(all_features)
    all_features_display = [display_name_map[f] for f in all_features]
    print(f"Using all {len(all_features)} features for co-abundance networks")

    X_display = X.copy()
    X_display.columns = all_features_display
    X_pd = X_display.loc[y == 1]
    X_healthy = X_display.loc[y == 0]

    corr_pd = build_correlation_matrix(X_pd)
    corr_healthy = build_correlation_matrix(X_healthy)

    pd_graph = build_network(corr_pd, threshold=args.corr_threshold)
    healthy_graph = build_network(corr_healthy, threshold=args.corr_threshold)
    color_map = get_node_color_map(all_features_display, X_pd, X_healthy)

    pd_summary = summarize_network(pd_graph)
    healthy_summary = summarize_network(healthy_graph)
    pd_only_df, healthy_only_df = differential_edge_frame(pd_graph, healthy_graph)

    pd_edges_df = edge_frame(pd_graph)
    healthy_edges_df = edge_frame(healthy_graph)

    feature_df = pd.DataFrame({"feature": all_features_display})

    features_csv = output_dir / "coabundance_top_features_full_xg.csv"
    features_raw_csv = output_dir / "coabundance_top_features_full_xg_raw.csv"
    pd_edges_csv = output_dir / "coabundance_pd_edges_full_xg.csv"
    healthy_edges_csv = output_dir / "coabundance_healthy_edges_full_xg.csv"
    diff_edges_csv = output_dir / "coabundance_differential_edges_full_xg.csv"
    network_png = output_dir / "coabundance_network_full_xg.png"
    summary_json = output_dir / "coabundance_summary_full_xg.json"

    feature_df.to_csv(features_csv, index=False)
    pd.DataFrame({"feature": all_features}).to_csv(features_raw_csv, index=False)
    pd_edges_df.to_csv(pd_edges_csv, index=False)
    healthy_edges_df.to_csv(healthy_edges_csv, index=False)
    pd.concat([pd_only_df, healthy_only_df], ignore_index=True).to_csv(diff_edges_csv, index=False)

    save_main_figure(
        pd_graph=pd_graph,
        healthy_graph=healthy_graph,
        color_map=color_map,
        importances=None,
        output_path=network_png,
    )

    summary = {
        "data_path": str(data_path),
        "target_column": args.target_column,
        "n_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "feature_domain": args.feature_domain,
        "corr_threshold": float(args.corr_threshold),
        "pd_summary": pd_summary,
        "healthy_summary": healthy_summary,
        "n_pd_only_edges": int(len(pd_only_df)),
        "n_healthy_only_edges": int(len(healthy_only_df)),
        "output_files": {
            "features_csv": str(features_csv),
            "features_raw_csv": str(features_raw_csv),
            "pd_edges_csv": str(pd_edges_csv),
            "healthy_edges_csv": str(healthy_edges_csv),
            "differential_edges_csv": str(diff_edges_csv),
            "network_figure_png": str(network_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print("\nNetwork stats:")
    print(f"  PD: nodes={pd_summary['nodes']}, edges={pd_summary['edges']}, hub={pd_summary['hub_feature']}")
    print(
        "  Healthy: "
        f"nodes={healthy_summary['nodes']}, edges={healthy_summary['edges']}, hub={healthy_summary['hub_feature']}"
    )
    print(f"  PD-only edges: {len(pd_only_df)}")
    print(f"  Healthy-only edges: {len(healthy_only_df)}")

    print("\nSaved outputs:")
    print(f"  - {features_csv}")
    print(f"  - {features_raw_csv}")
    print(f"  - {pd_edges_csv}")
    print(f"  - {healthy_edges_csv}")
    print(f"  - {diff_edges_csv}")
    print(f"  - {network_png}")
    print(f"  - {summary_json}")


if __name__ == "__main__":
    main()
