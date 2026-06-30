"""Improved co-abundance network analysis.

Improvements over original approach:
1. Species-level only (|s__) — fixes taxonomic mismatch with Metcalfe-Roach validation
2. Exact NetworkX betweenness centrality (replaces RWBC approximation)
3. Louvain community detection on each network
4. Global topology comparison: modularity, avg clustering, characteristic path length
5. Data is already CLR-transformed — compositionality is handled upstream

All outputs go to improvements/results/network/.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import community as community_louvain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

DATA_PATH = PROJECT_ROOT / "data" / "wallen_2022_primary_cohort" / "microbiome_V2_clr.csv"
TARGET = "Case_status_PD"
OUT_DIR = PROJECT_ROOT / "results" / "network"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORR_THRESHOLD = 0.45   # slightly tighter than original 0.55 on Spearman → fewer spurious edges
MIN_PREVALENCE = 0.10   # feature must be non-zero in >=10% of cohort samples
TOP_N_FEATURES = 150    # cap for tractable exact betweenness
TOP_K_HUBS = 15         # top keystones to report
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    # Keep only species-level taxonomic features (fixes taxonomic mismatch)
    species_cols = [c for c in X.columns if "|s__" in c]
    X = X[species_cols]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y


def filter_features(X: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    Xs = X.loc[mask]
    # prevalence filter: non-zero in >=MIN_PREVALENCE of cohort samples
    prev = (Xs != 0).mean()
    Xs = Xs.loc[:, prev >= MIN_PREVALENCE]
    # variance filter: keep top TOP_N by variance
    top = Xs.var().nlargest(TOP_N_FEATURES).index
    return Xs[top]


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_network(X: pd.DataFrame) -> nx.Graph:
    corr, _ = spearmanr(X.values, axis=0)
    if X.shape[1] == 1:
        return nx.Graph()
    corr = np.array(corr)
    np.fill_diagonal(corr, 0)
    G = nx.Graph()
    feats = list(X.columns)
    G.add_nodes_from(feats)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if abs(corr[i, j]) >= CORR_THRESHOLD:
                G.add_edge(feats[i], feats[j], weight=float(abs(corr[i, j])),
                           sign=1 if corr[i, j] > 0 else -1)
    return G


def short_name(col: str) -> str:
    parts = col.split("|s__")
    return parts[-1].replace("_", " ") if parts else col


# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------

def topology_metrics(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)

    # largest connected component for path-length metrics
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    lcc_frac = lcc.number_of_nodes() / n_nodes
    try:
        avg_path = nx.average_shortest_path_length(lcc)
    except Exception:
        avg_path = float("nan")

    # Louvain modularity
    partition = community_louvain.best_partition(G, random_state=SEED)
    modularity = community_louvain.modularity(partition, G)
    n_communities = len(set(partition.values()))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 4),
        "avg_clustering": round(avg_clustering, 4),
        "lcc_fraction": round(lcc_frac, 4),
        "avg_path_length_lcc": round(avg_path, 4) if not np.isnan(avg_path) else None,
        "modularity": round(modularity, 4),
        "n_communities": n_communities,
    }


# ---------------------------------------------------------------------------
# Exact betweenness centrality keystones
# ---------------------------------------------------------------------------

def get_keystones(G: nx.Graph, k: int = TOP_K_HUBS) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    df = pd.Series(bc, name="betweenness").sort_values(ascending=False).head(k).reset_index()
    df.columns = ["feature", "betweenness"]
    df["short_name"] = df["feature"].apply(short_name)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X, y = load_data()
    pd_mask = y == 1
    ctrl_mask = y == 0

    print(f"Species-level features after CLR load: {X.shape[1]}")

    X_pd   = filter_features(X, pd_mask)
    X_ctrl = filter_features(X, ctrl_mask)

    print(f"PD   subgroup features after filtering: {X_pd.shape[1]}")
    print(f"Ctrl subgroup features after filtering: {X_ctrl.shape[1]}")

    G_pd   = build_network(X_pd)
    G_ctrl = build_network(X_ctrl)

    print(f"PD   network: {G_pd.number_of_nodes()} nodes, {G_pd.number_of_edges()} edges")
    print(f"Ctrl network: {G_ctrl.number_of_nodes()} nodes, {G_ctrl.number_of_edges()} edges")

    # Topology comparison
    topo_pd   = topology_metrics(G_pd)
    topo_ctrl = topology_metrics(G_ctrl)
    topo_df = pd.DataFrame({"Metric": list(topo_pd.keys()),
                             "PD": list(topo_pd.values()),
                             "Control": [topo_ctrl.get(k) for k in topo_pd]})
    topo_df.to_csv(OUT_DIR / "topology_comparison.csv", index=False)
    print("\nTopology comparison:")
    print(topo_df.to_string(index=False))

    # Keystones
    hubs_pd   = get_keystones(G_pd).assign(cohort="PD")
    hubs_ctrl = get_keystones(G_ctrl).assign(cohort="Control")
    hubs_all  = pd.concat([hubs_pd, hubs_ctrl], ignore_index=True)
    hubs_all.to_csv(OUT_DIR / "keystones_species_level.csv", index=False)
    print("\nPD keystones (top 10):")
    print(hubs_pd[["short_name", "betweenness"]].head(10).to_string(index=False))
    print("\nControl keystones (top 10):")
    print(hubs_ctrl[["short_name", "betweenness"]].head(10).to_string(index=False))

    # Delta betweenness (PD - Control) for shared features
    bc_pd   = hubs_pd.set_index("feature")["betweenness"]
    bc_ctrl = hubs_ctrl.set_index("feature")["betweenness"]
    shared  = bc_pd.index.intersection(bc_ctrl.index)
    delta_df = pd.DataFrame({
        "feature": shared,
        "short_name": [short_name(f) for f in shared],
        "bc_pd": bc_pd[shared].values,
        "bc_ctrl": bc_ctrl[shared].values,
        "delta_bc": bc_pd[shared].values - bc_ctrl[shared].values,
    }).sort_values("delta_bc", key=abs, ascending=False)
    delta_df.to_csv(OUT_DIR / "keystones_delta.csv", index=False)

    # Louvain communities
    part_pd   = community_louvain.best_partition(G_pd,   random_state=SEED)
    part_ctrl = community_louvain.best_partition(G_ctrl, random_state=SEED)
    comm_pd   = pd.DataFrame({"feature": list(part_pd.keys()), "community": list(part_pd.values())})
    comm_ctrl = pd.DataFrame({"feature": list(part_ctrl.keys()), "community": list(part_ctrl.values())})
    comm_pd.assign(short_name=comm_pd["feature"].apply(short_name)).to_csv(OUT_DIR / "communities_pd.csv", index=False)
    comm_ctrl.assign(short_name=comm_ctrl["feature"].apply(short_name)).to_csv(OUT_DIR / "communities_ctrl.csv", index=False)

    # Save summary JSON
    summary = {
        "pd_network":   topo_pd,
        "ctrl_network": topo_ctrl,
        "top_pd_keystones":   hubs_pd[["short_name", "betweenness"]].head(10).to_dict("records"),
        "top_ctrl_keystones": hubs_ctrl[["short_name", "betweenness"]].head(10).to_dict("records"),
    }
    (OUT_DIR / "network_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAll outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
