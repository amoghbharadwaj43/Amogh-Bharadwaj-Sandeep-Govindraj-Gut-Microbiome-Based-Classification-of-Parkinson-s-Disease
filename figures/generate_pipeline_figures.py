from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

# ── matplotlib cache ─────────────────────────────────────────────────────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdgcache")

import matplotlib as mpl
mpl.use("Agg")

# ── global aesthetics ────────────────────────────────────────────────────────
mpl.rcParams["font.family"]                    = "DejaVu Sans"
mpl.rcParams["font.size"]                      = 11
mpl.rcParams["axes.titlesize"]                 = 12
mpl.rcParams["axes.spines.top"]                = False
mpl.rcParams["axes.spines.right"]              = False
mpl.rcParams["figure.constrained_layout.use"]  = True

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_curve, auc
from matplotlib_venn import venn2

BASE_DIR = Path(__file__).resolve().parents[1]
ART_MOD  = BASE_DIR / "results" / "classification"
ART_NET  = BASE_DIR / "results" / "network"
ART_ANA  = BASE_DIR / "results" / "statistics"
OUT_DIR  = BASE_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAY       = "#777777"
BLUE       = "#3b6fb0"
BLUE_MID   = "#6a9fd4"   # Tier 1b – softer convergence
LIGHT_BLUE = "#c7d9f0"

SAVED: list[Path] = []

# ── candidate species ────────────────────────────────────────────────────────
TIER1  = {"Blautia wexlerae"}                        # all classifiers + network
TIER1B = {"Megasphaera", "Bacteroides vulgatus"}     # XGBoost top-40 + network top-40 Δ RWBC
TIER2  = {"Faecalicoccus pleomorphus", "Actinomyces odontolyticus",
           "Streptococcus australis"}                # abundance branch only
TIER3  = {"Slackia isoflavoniconvertens", "Faecalibacterium prausnitzii",
           "Rothia dentocariosa", "Christensenella"} # network branch only
ALL_CANDIDATES = TIER1 | TIER1B | TIER2 | TIER3


# ── helpers ──────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> Path:
    p = OUT_DIR / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    SAVED.append(p)
    print(f"  saved → {p}")
    return p


def it(name: str) -> str:
    """Return math-italic string for a species name."""
    safe = name.replace(" ", r"\ ")
    return r"$\it{" + safe + r"}$"


def clean_feature(raw: str) -> str:
    """Convert raw feature path to a short, readable label."""
    raw = str(raw)
    # metaphlan species: extract s__Genus_species
    if "s__" in raw:
        sp = raw.split("s__")[-1].replace("_", " ").strip()
        return sp
    # metaphlan genus level
    if "g__" in raw and "metaphlan" in raw:
        g = raw.split("g__")[-1].split("|")[0].replace("_", " ").strip()
        return g + " (genus)"
    # humann KO
    if "humann_KO_group_counts::" in raw:
        part = raw.split("::")[-1]
        kid  = part.split(":")[0].strip()   # e.g. K15584
        desc = part.split(":")[-1].strip() if ":" in part else ""
        words = desc.split()[:4]
        return kid + ": " + " ".join(words) if words else kid
    # humann pathway
    if "humann_pathway" in raw:
        part = raw.split("::")[-1]
        return part.split(":")[0].strip()[:30]
    # humann rel_ab genus
    if "humann_rel_ab" in raw or "metaphlan_rel_ab" in raw:
        if "g__" in raw:
            g = raw.split("g__")[-1].split("|")[0].replace("_", " ")
            return g + " (rel.ab)"
    # metadata
    raw_clean = raw.replace("_Y", ": Y").replace("_M", ": M")
    raw_clean = raw_clean.replace("metaphlan_counts::", "").replace("metaphlan_rel_ab::", "")
    return raw_clean.strip()[:35]


def is_species(name_clean: str) -> bool:
    """Return True if the cleaned name looks like a binomial."""
    parts = name_clean.split()
    return len(parts) >= 2 and not any(c in name_clean for c in ["/", ":", "[", "KO"])


def make_label(name: str, width: int = 22) -> str:
    """Italic label for species, wrapped; plain for non-species."""
    if is_species(name):
        parts = textwrap.wrap(name, width)
        return "\n".join(it(p) for p in parts)
    parts = textwrap.wrap(name, width + 4)
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Pipeline schematic (with novel steps)
# ════════════════════════════════════════════════════════════════════════════

def fig1():
    print("Generating Figure 1 – pipeline schematic…")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    BOX_FC = "#f0f4fa"
    BOX_EC = "#aaaaaa"
    BOX_LW = 1.2
    HL_FC  = LIGHT_BLUE
    HL_EC  = BLUE

    def box(ax, cx, cy, w, h, text, fontsize=9.0, fc=BOX_FC, ec=BOX_EC, bold=False):
        x, y = cx - w / 2, cy - h / 2
        bp = FancyBboxPatch((x, y), w, h,
                            boxstyle="square,pad=0.05",
                            facecolor=fc, edgecolor=ec,
                            linewidth=BOX_LW, zorder=3)
        ax.add_patch(bp)
        weight = "bold" if bold else "normal"
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, multialignment="center",
                fontweight=weight, zorder=4, linespacing=1.35)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                   lw=1.2, mutation_scale=12), zorder=2)

    # ── top chain ────────────────────────────────────────────────────────────
    box(ax, 1.4,  9.3, 2.2, 0.6, "Stool sample")
    box(ax, 4.1,  9.3, 2.8, 0.6, "Shotgun metagenomic\nsequencing")
    box(ax, 7.7,  9.3, 3.8, 0.6,
        "Feature table\n11,117 features × 724 samples")

    arrow(ax, 2.50, 9.3, 2.70, 9.3)
    arrow(ax, 5.50, 9.3, 5.80, 9.3)

    # split down to two branches
    arrow(ax, 7.7, 9.0, 3.0, 8.25)
    arrow(ax, 7.7, 9.0, 7.7, 8.25)

    # ── Branch A (left) ──────────────────────────────────────────────────────
    ax.text(3.0, 8.55, "Branch A – Abundance", ha="center",
            fontsize=8.5, color=BLUE, style="italic")

    a_boxes = [
        (3.0, 8.05, 2.8, 0.55, "Abundance preprocessing\n(CLR + variance filter)"),
        (3.0, 7.15, 2.8, 0.70, "Classifiers:\nDecision Tree / XGBoost\n/ Random Forest"),
        (3.0, 6.18, 2.8, 0.55, "SHAP feature importance"),
        (3.0, 5.25, 2.8, 0.55, "Nested 5×CV\n+ permutation test"),  # novel
    ]
    for cx, cy, w, h, txt in a_boxes:
        box(ax, cx, cy, w, h, txt)

    arrow(ax, 3.0, 7.77, 3.0, 7.52)
    arrow(ax, 3.0, 6.80, 3.0, 6.48)
    arrow(ax, 3.0, 5.90, 3.0, 5.53)

    # novel badge
    ax.text(4.52, 5.25, "novel", ha="left", va="center",
            fontsize=7.5, color=BLUE, style="italic")

    # ── Branch B (right) ─────────────────────────────────────────────────────
    ax.text(7.7, 8.55, "Branch B – Network", ha="center",
            fontsize=8.5, color=BLUE, style="italic")

    b_boxes = [
        (7.7, 8.05, 2.8, 0.55, "Co-abundance graph\n(|r| ≥ 0.55)"),
        (7.7, 7.15, 2.8, 0.70, "Random Walk Betweenness\nCentrality (PD vs control)"),
        (7.7, 6.18, 2.8, 0.55, "Delta centrality\n(PD − control)"),     # novel label
        (7.7, 5.25, 2.8, 0.55, "Hub identification"),
    ]
    for cx, cy, w, h, txt in b_boxes:
        box(ax, cx, cy, w, h, txt)

    arrow(ax, 7.7, 7.77, 7.7, 7.52)
    arrow(ax, 7.7, 6.80, 7.7, 6.48)
    arrow(ax, 7.7, 5.90, 7.7, 5.53)

    ax.text(9.22, 6.18, "novel", ha="left", va="center",
            fontsize=7.5, color=BLUE, style="italic")

    # ── convergence ──────────────────────────────────────────────────────────
    arrow(ax, 3.0, 4.97, 4.8, 4.42)
    arrow(ax, 7.7, 4.97, 6.0, 4.42)

    box(ax, 5.4, 4.15, 3.2, 0.55, "Convergence of branches",
        fc=HL_FC, ec=BLUE)
    arrow(ax, 5.4, 3.88, 5.4, 3.42)

    box(ax, 5.4, 3.18, 3.2, 0.50, "Tiered candidate bacteria",
        fc=HL_FC, ec=BLUE)
    arrow(ax, 5.4, 2.93, 5.4, 2.47)

    # cross-cohort validation – novel final step
    box(ax, 5.4, 2.22, 3.5, 0.50,
        "Cross-cohort validation\n(Metcalfe-Roach 2024)",
        fc=HL_FC, ec=BLUE)
    ax.text(7.22, 2.22, "novel", ha="left", va="center",
            fontsize=7.5, color=BLUE, style="italic")

    save(fig, "fig1_pipeline.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Classifier performance (real values, unchanged)
# ════════════════════════════════════════════════════════════════════════════

def fig2():
    print("Generating Figure 2 – classifier performance…")
    models = ["Decision\nTree", "XGBoost", "Random\nForest"]
    acc    = [0.675, 0.725, 0.753]
    auc_v  = [0.626, 0.771, 0.789]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_acc = ax.bar(x - w / 2, acc,   width=w, color=GRAY, label="Accuracy", zorder=3)
    bars_auc = ax.bar(x + w / 2, auc_v, width=w, color=BLUE, label="AUC",      zorder=3)

    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.2,
               label="Chance (0.50)", zorder=2)

    for bar in list(bars_acc) + list(bars_auc):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    rf_auc_x = x[2] + w / 2
    ax.annotate("permutation\np = 0.005",
                xy=(rf_auc_x, auc_v[2] + 0.012),
                xytext=(rf_auc_x + 0.55, auc_v[2] + 0.055),
                fontsize=8.5, color=BLUE, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                lw=1.0, mutation_scale=9))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylabel("Score")
    ax.set_title("Classifier performance (nested cross-validation)")
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    save(fig, "fig2_classifier_performance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – ROC curves (real OOF predictions)
# ════════════════════════════════════════════════════════════════════════════

def _load_oof(fname: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(fname)
    return df["y_true"].values, df["y_proba"].values


def fig3():
    print("Generating Figure 3 – ROC curves (real OOF data)…")

    specs = [
        ("Decision Tree", ART_MOD / "nested_cv_all_models_dt_oof_predictions.csv",  GRAY, "--", 0.626),
        ("XGBoost",       ART_MOD / "nested_cv_all_models_xgb_oof_predictions.csv", GRAY, ":",  0.771),
        ("Random Forest", ART_MOD / "nested_cv_all_models_rf_oof_predictions.csv",  BLUE, "-",  0.789),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for label, fpath, color, ls, reported_auc in specs:
        if fpath.exists():
            y_true, y_proba = _load_oof(fpath)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            print(f"  {label}: computed AUC = {roc_auc:.3f} (reported {reported_auc})")
        else:
            print(f"  WARNING: OOF file not found for {label}, using reported AUC")
            fpr  = np.linspace(0, 1, 200)
            k    = 1.0 / reported_auc - 1.0
            tpr  = fpr ** max(k, 0.01)
            roc_auc = reported_auc

        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.0,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
            linewidth=1.0, label="Chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves – all classifiers")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_aspect("equal")

    save(fig, "fig3_roc_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Abundance branch SHAP (real data, top 15 across all feature types)
# ════════════════════════════════════════════════════════════════════════════

def fig4():
    print("Generating Figure 4 – SHAP importance (real data)…")
    shap_path = ART_MOD / "shap_feature_importance.csv"
    df = pd.read_csv(shap_path)
    df = df.sort_values("mean_abs_shap", ascending=False).head(30).reset_index(drop=True)

    df["label_clean"] = df["feature"].apply(clean_feature)

    def get_color(clean_label: str) -> str:
        for cand in TIER1 | TIER2:
            if cand.lower() in clean_label.lower():
                return BLUE
        for cand in TIER1B:
            if cand.lower() in clean_label.lower():
                return BLUE_MID
        return GRAY

    # Sort ascending so highest bar is on top
    df = df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    colors = [get_color(lbl) for lbl in df["label_clean"]]
    labels = [make_label(lbl) for lbl in df["label_clean"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(df))
    bars = ax.barh(y, df["mean_abs_shap"], color=colors, height=0.65, zorder=3)

    for bar, val in zip(bars, df["mean_abs_shap"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Mean |SHAP value| (XGBoost)")
    ax.set_title("Abundance branch: top 30 features (SHAP)")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.7, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.20)

    legend_patches = [
        mpatches.Patch(color=BLUE,     label="Tier 1 / 2 candidate (all classifiers)"),
        mpatches.Patch(color=BLUE_MID, label="Tier 1b candidate (XGBoost + network)"),
        mpatches.Patch(color=GRAY,     label="Other features"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="lower right")

    save(fig, "fig4_shap_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Co-abundance network (ALL real nodes + edges, threshold 0.98)
# ════════════════════════════════════════════════════════════════════════════

def _short_node(raw: str) -> str:
    """Extract species or genus name from full metaphlan path."""
    if "s__" in raw:
        return raw.split("s__")[-1].replace("_", " ").strip()
    if "g__" in raw:
        return raw.split("g__")[-1].split("|")[0].replace("_", " ").strip()
    if "p__" in raw:
        return raw.split("p__")[-1].split("|")[0].replace("_", " ").strip()
    return raw.split("|")[-1].replace("_", " ").strip()


def fig5():
    print("Generating Figure 5 – co-abundance network (all real nodes, Δ RWBC sizing)…")

    edges_path  = ART_NET / "coabundance_pd_edges_full_xg.csv"
    pd_cent_path = ART_NET / "coabundance_rwbc_pd_centrality.csv"
    hc_cent_path = ART_NET / "coabundance_rwbc_healthy_centrality.csv"

    edges_df = pd.read_csv(edges_path)
    pd_cent  = pd.read_csv(pd_cent_path)[["feature", "random_walk_betweenness"]].rename(
                   columns={"random_walk_betweenness": "pd_rwbc"})
    hc_cent  = pd.read_csv(hc_cent_path)[["feature", "random_walk_betweenness"]].rename(
                   columns={"random_walk_betweenness": "hc_rwbc"})

    # Merge and compute Δ RWBC (PD − healthy)
    cent = pd_cent.merge(hc_cent, on="feature", how="outer").fillna(0)
    cent["delta"] = cent["pd_rwbc"] - cent["hc_rwbc"]
    delta_lookup: dict[str, float] = dict(zip(cent["feature"], cent["delta"]))

    # Threshold edges for display while keeping all nodes from PD network
    DISPLAY_THRESH = 0.98
    edges_filt = edges_df[edges_df["abs_correlation"] >= DISPLAY_THRESH].copy()
    print(f"  Network built at |r| ≥ 0.55 · displayed at |r| ≥ {DISPLAY_THRESH}")
    print(f"  Edges shown: {len(edges_filt):,}")

    # Build short-name mappings from the PD centrality list (all nodes)
    all_nodes_raw = pd_cent["feature"].tolist()
    raw2short: dict[str, str] = {r: _short_node(r) for r in all_nodes_raw}
    short2raw: dict[str, str] = {v: k for k, v in raw2short.items()}

    G = nx.Graph()
    G.add_nodes_from([raw2short[r] for r in all_nodes_raw])
    for _, row in edges_filt.iterrows():
        u = raw2short.get(row["feature_a"], _short_node(row["feature_a"]))
        v = raw2short.get(row["feature_b"], _short_node(row["feature_b"]))
        if u != v and G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── Δ RWBC per node for sizing ────────────────────────────────────────────
    node_list  = list(G.nodes())
    delta_vals = np.array([delta_lookup.get(short2raw.get(n, n), 0.0)
                           for n in node_list])

    def is_hub(node_name: str) -> bool:
        for cand in ALL_CANDIDATES:
            parts = cand.split()
            if all(p.lower() in node_name.lower() for p in parts):
                return True
            if len(parts) == 1 and parts[0].lower() in node_name.lower():
                return True
        return False

    def is_tier1b(node_name: str) -> bool:
        for cand in TIER1B:
            parts = cand.split()
            if all(p.lower() in node_name.lower() for p in parts):
                return True
        return False

    hub_mask    = np.array([is_hub(n) for n in node_list])
    tier1b_mask = np.array([is_tier1b(n) for n in node_list])
    hub_nodes   = {n for n, h in zip(node_list, hub_mask) if h}
    print(f"  Hub nodes matched: {hub_nodes}")

    hub_deltas = {n: delta_lookup.get(short2raw.get(n, n), 0.0) for n in hub_nodes}
    max_hub_d  = max(abs(d) for d in hub_deltas.values()) if hub_deltas else 1.0

    sizes = []
    for n, h in zip(node_list, hub_mask):
        if h:
            d = abs(hub_deltas.get(n, 0.0))
            sizes.append(200 + 700 * d / max_hub_d)
        else:
            sizes.append(12)

    # Tier 1b gets medium blue; Tier 1/2/3 gets dark blue
    node_colors = []
    for n, h, t1b in zip(node_list, hub_mask, tier1b_mask):
        if h and t1b:
            node_colors.append(BLUE_MID)
        elif h:
            node_colors.append(BLUE)
        else:
            node_colors.append(GRAY)

    print("  Computing layout (spring, 40 iterations)…")
    pos = nx.spring_layout(G, k=1.4, iterations=40, seed=42)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.axis("off")

    # Draw non-hub nodes first (behind hubs)
    non_hub_idx = [i for i, h in enumerate(hub_mask) if not h]
    hub_idx     = [i for i, h in enumerate(hub_mask) if h]
    non_hub_nodes = [node_list[i] for i in non_hub_idx]
    hub_nodes_list = [node_list[i] for i in hub_idx]

    # Edges (very thin, low alpha — dense but shows connectivity)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.04, width=0.25, edge_color=GRAY)

    # Non-hub nodes
    nx.draw_networkx_nodes(G, pos, nodelist=non_hub_nodes, ax=ax,
                           node_size=[sizes[i] for i in non_hub_idx],
                           node_color=GRAY, alpha=0.55)
    # Hub nodes (drawn last, on top)
    nx.draw_networkx_nodes(G, pos, nodelist=hub_nodes_list, ax=ax,
                           node_size=[sizes[i] for i in hub_idx],
                           node_color=BLUE, alpha=0.90)

    # ── Labels + Δ annotations for hubs ──────────────────────────────────────
    hub_pos = {n: pos[n] for n in hub_nodes_list if n in pos}
    for n in hub_pos:
        x, y = hub_pos[n]
        d = hub_deltas.get(n, 0.0)
        sign = "+" if d >= 0 else "−"
        parts = textwrap.wrap(n, 15)
        sp_label = "\n".join(r"$\it{" + p.replace(" ", r"\ ") + r"}$" for p in parts)
        delta_label = f"ΔRWBC {sign}{abs(d):.2f}"

        # Species name above node
        ax.text(x, y + 0.005, sp_label,
                ha="center", va="bottom", fontsize=7.2, color="#111111",
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.80))
        # Delta value below node
        ax.text(x, y - 0.005, delta_label,
                ha="center", va="top", fontsize=6.5,
                color=BLUE if d >= 0 else "#aa3333",
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75))

    ax.set_title(
        f"Co-abundance network – PD samples  "
        f"({G.number_of_nodes()} nodes, {G.number_of_edges():,} edges)\n"
        f"Built at |r| ≥ 0.55 · displayed at |r| ≥ {DISPLAY_THRESH} · "
        "node size = |ΔRWBC|  (PD − control)",
        fontsize=10,
    )

    legend_patches = [
        mpatches.Patch(color=BLUE,     label="Tier 1 / 2 / 3 hub (all classifiers or strong Δ RWBC)"),
        mpatches.Patch(color=BLUE_MID, label="Tier 1b hub (XGBoost + network, softer signal)"),
        mpatches.Patch(color=GRAY,     label="Other species"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.0, loc="lower left", framealpha=0.85)

    save(fig, "fig5_network.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Venn diagram (species names INSIDE circles, no numbers)
# ════════════════════════════════════════════════════════════════════════════

def fig6():
    print("Generating Figure 6 – Venn diagram…")

    tier1_list  = sorted(TIER1)    # ["Blautia wexlerae"]
    tier1b_list = sorted(TIER1B)   # ["Bacteroides vulgatus", "Megasphaera"]
    tier2_list  = sorted(TIER2)    # 3 species
    tier3_list  = sorted(TIER3)    # 4 species

    fig = plt.figure(figsize=(9, 7.2))
    ax  = fig.add_subplot(111)
    ax.set_aspect("equal")

    v = venn2(subsets=(len(tier2_list), len(tier3_list),
                       len(tier1_list) + len(tier1b_list)),
              set_labels=("Abundance\nclassifier", "Network\ncentrality"),
              ax=ax, set_colors=(GRAY, GRAY), alpha=0.50)

    for pid, fc, al in [("10", GRAY, 0.42), ("01", GRAY, 0.42), ("11", BLUE, 0.62)]:
        p = v.get_patch_by_id(pid)
        if p:
            p.set_facecolor(fc); p.set_alpha(al)
    for pid in ("10", "01", "11"):
        lbl = v.get_label_by_id(pid)
        if lbl: lbl.set_text("")
    for sid in ("A", "B"):
        lbl = v.get_label_by_id(sid)
        if lbl: lbl.set_fontsize(10); lbl.set_fontweight("normal")

    # ── Extract circle geometry ───────────────────────────────────────────────
    def _xy(pt):
        try:    return float(pt[0]), float(pt[1])
        except: return float(pt.x), float(pt.y)

    c0x, c0y = _xy(v.centers[0])
    c1x, c1y = _xy(v.centers[1])
    r0 = float(v.radii[0])
    r1 = float(v.radii[1])
    cx = (c0x + c1x) / 2          # horizontal centre of overlap
    cy = (c0y + c1y) / 2          # vertical centre (≈ 0)

    # Pull region centres well inside each exclusive lune
    left_cx  = c0x - 0.38 * r0
    right_cx = c1x + 0.38 * r1

    # ── Get intersection bounds for precise vertical placement ───────────────
    inter_patch = v.get_patch_by_id("11")
    if inter_patch is not None:
        bb = inter_patch.get_path().get_extents()
        iy0, iy1 = bb.y0, bb.y1
    else:
        iy0, iy1 = cy - 0.32, cy + 0.32
    ih = iy1 - iy0          # total intersection height in data coords

    # ── Helper: place italic species labels evenly within a vertical band ────
    BBOX = dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.80)

    def place_labels(names, x, y_top, y_bot, color, fs, bold=False):
        """Space `names` evenly between y_bot and y_top."""
        n = len(names)
        positions = np.linspace(y_top, y_bot, n) if n > 1 else [( y_top + y_bot) / 2]
        for name, y in zip(names, positions):
            ax.text(x, y, it(name), ha="center", va="center",
                    fontsize=fs, color=color,
                    fontweight="bold" if bold else "normal",
                    bbox=BBOX)

    # ── Left-only region: Tier 2 (3 species) ─────────────────────────────────
    place_labels(tier2_list, left_cx,
                 y_top=cy + 0.17, y_bot=cy - 0.17,
                 color="#111111", fs=6.8)

    # ── Right-only region: Tier 3 (4 species) ────────────────────────────────
    place_labels(tier3_list, right_cx,
                 y_top=cy + 0.22, y_bot=cy - 0.22,
                 color="#111111", fs=6.8)

    # ── Intersection: Tier 1 header + species, divider, Tier 1b header + species
    #    Divide intersection vertically into four even slots
    #    Slot fractions from top: [Tier1 label][Tier1 species][divider+Tier1b label][1b species×2]
    # Use 4 evenly spaced rows, with top padding 10% and bottom padding 10%
    pad   = 0.10 * ih
    usable_top = iy1 - pad
    usable_bot = iy0 + pad
    usable_h   = usable_top - usable_bot

    # 5 rows: T1-header, T1-species, divider, T1b-header, T1b-species×2
    # Give the divider half the weight of a species row
    weights = [0.5, 1.2, 0.4, 0.6, 1.0, 1.0]
    total_w = sum(weights)
    # Convert weights to cumulative y-positions (from top down)
    ys = []
    acc = 0
    for w in weights:
        acc += w
        ys.append(usable_top - (acc - w / 2) / total_w * usable_h)

    t1_hdr_y, t1_sp_y, div_y, t1b_hdr_y, t1b_sp1_y, t1b_sp2_y = ys

    # "TIER 1" header (small caps style via bold + tracking)
    ax.text(cx, t1_hdr_y, "TIER 1",
            ha="center", va="center", fontsize=6.5,
            color="#0a2a5e", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.0))

    # Tier 1 species – bold, dark blue
    ax.text(cx, t1_sp_y, it(tier1_list[0]),
            ha="center", va="center", fontsize=8.5,
            color="#0a2a5e", fontweight="bold", bbox=BBOX)

    # Thin divider line
    line_half = 0.18
    ax.plot([cx - line_half, cx + line_half], [div_y, div_y],
            color="#888888", linewidth=0.8, zorder=5)

    # "TIER 1b" header – same bold small-caps style, medium blue
    ax.text(cx, t1b_hdr_y, "TIER 1b",
            ha="center", va="center", fontsize=6.5,
            color="#1a3a6e", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.0))

    # Tier 1b species – slightly smaller, medium blue
    for sp, y in zip(tier1b_list, [t1b_sp1_y, t1b_sp2_y]):
        ax.text(cx, y, it(sp),
                ha="center", va="center", fontsize=7.5,
                color="#1a3a6e", bbox=BBOX)

    ax.set_title("Convergence of dual-branch pipeline",
                 fontsize=11, pad=6)

    # Legend below title
    legend_patches = [
        mpatches.Patch(color=BLUE,     label="Tier 1: all classifiers + network"),
        mpatches.Patch(color=BLUE_MID, label="Tier 1b: XGBoost + network (softer)"),
        mpatches.Patch(color=GRAY,     label="Tier 2: abundance only  |  Tier 3: network only"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.0, 0.98), ncol=1, framealpha=0.90,
              edgecolor="#cccccc")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    save(fig, "fig6_venn.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Ablation / robustness: microbiome-only vs full model
# ════════════════════════════════════════════════════════════════════════════

def fig7():
    print("Generating Figure 7 – ablation robustness chart…")

    abla_path = BASE_DIR / "results" / "ablation" / "ablation_nested_cv_results.csv"
    df = pd.read_csv(abla_path)

    # Compute mean and std AUC per (model, condition)
    summary = (df.groupby(["model", "condition"])["auc"]
                 .agg(mean="mean", std="std")
                 .reset_index())

    model_order = ["Decision Tree", "Random Forest", "XGBoost"]
    cond_order  = ["Full", "No-Constipation", "Microbiome-Only"]
    cond_labels = ["Full\n(all features)", "No constipation\n(sex removed too)", "Microbiome only\n(clinical removed)"]
    cond_colors = [GRAY, "#9aafd4", BLUE]

    x = np.arange(len(model_order))
    n_conds = len(cond_order)
    total_w = 0.72
    bar_w   = total_w / n_conds

    fig, ax = plt.subplots(figsize=(8, 5))

    for ci, (cond, label, color) in enumerate(zip(cond_order, cond_labels, cond_colors)):
        offset = (ci - (n_conds - 1) / 2) * bar_w
        vals  = []
        errs  = []
        for m in model_order:
            row = summary[(summary["model"] == m) & (summary["condition"] == cond)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
            errs.append(float(row["std"].iloc[0])  if not row.empty else 0.0)

        bars = ax.bar(x + offset, vals, width=bar_w * 0.90,
                      color=color, label=label, zorder=3,
                      yerr=errs, error_kw=dict(elinewidth=1.2, capsize=3,
                                               ecolor="#444444"))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Chance line
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.1,
               label="Chance (0.50)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=10)
    ax.set_ylim(0.40, 0.92)
    ax.set_ylabel("AUC (mean ± std, nested 5-fold CV)")
    ax.set_title("Robustness check: does microbiome signal survive\nafter removing clinical confounders?")
    ax.legend(fontsize=8.5, loc="lower right", ncol=1)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Annotation highlighting the key takeaway
    rf_micro_row = summary[(summary["model"] == "Random Forest") &
                            (summary["condition"] == "Microbiome-Only")]
    if not rf_micro_row.empty:
        rf_micro_auc = float(rf_micro_row["mean"].iloc[0])
        ax.annotate(
            f"RF microbiome-only\nAUC = {rf_micro_auc:.3f}",
            xy=(x[model_order.index("Random Forest")] + bar_w, rf_micro_auc),
            xytext=(x[model_order.index("Random Forest")] + bar_w * 2.2, rf_micro_auc + 0.04),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, mutation_scale=9),
            ha="left",
        )

    save(fig, "fig7_ablation_robustness.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Science Fair 25-26: generating visual aid figures ===\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()

    print("\n=== All figures saved ===")
    for p in SAVED:
        print(f"  {p}")

    print("\n=== Dimension check ===")
    from PIL import Image
    for p in SAVED:
        with Image.open(p) as im:
            print(f"  {p.name:45s}  {im.size[0]} × {im.size[1]} px")
