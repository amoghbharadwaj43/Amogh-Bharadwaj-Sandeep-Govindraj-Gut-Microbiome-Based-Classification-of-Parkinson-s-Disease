"""Regenerate bar chart, ego-network, and table with cosmetic fixes."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

# Paths resolved relative to this script so the bundle is portable.
_HERE = Path(__file__).resolve().parent          # .../paper_figures_RF
_ROOT = _HERE.parent                             # bundle root
OUT = str(_ROOT / "paper_figures")

# ─── Shared palette ────────────────────────────────────────────────────────────
PD_COL  = "#D94F3D"   # red
HC_COL  = "#4878CF"   # blue
EDGE_COL = "#CCCCCC"  # light grey for all edges

# ══════════════════════════════════════════════════════════════════════════════
# 1. BAR CHART  (no arrow)
# ══════════════════════════════════════════════════════════════════════════════
data = {
    "species":  ["Absiella\ndolichum",  "Absiella\ndolichum",
                 "Lactonifactor\nlongoviformis", "Lactonifactor\nlongoviformis"],
    "condition": ["PD", "HC", "PD", "HC"],
    "RWBC":     [4.529, 3.412, 5.570, 5.273],
    "Degree":   [15,    28,    19,    43],
}
df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor("white")

species_list = ["Absiella\ndolichum", "Lactonifactor\nlongoviformis"]
x = np.arange(len(species_list))
width = 0.35
palette = {"PD": PD_COL, "HC": HC_COL}

for ax, metric, ylabel in zip(axes, ["RWBC", "Degree"], ["RWBC Score", "Co-abundance Degree"]):
    pd_vals = [df.loc[(df.species==s)&(df.condition=="PD"), metric].values[0] for s in species_list]
    hc_vals = [df.loc[(df.species==s)&(df.condition=="HC"), metric].values[0] for s in species_list]
    bars_pd = ax.bar(x - width/2, pd_vals, width, color=PD_COL, label="PD",    edgecolor="white", linewidth=0.8)
    bars_hc = ax.bar(x + width/2, hc_vals, width, color=HC_COL, label="Healthy", edgecolor="white", linewidth=0.8)
    # value labels on bars
    for bar in bars_pd:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.05*max(pd_vals+hc_vals),
                f"{h:.2f}" if metric=="RWBC" else f"{int(h)}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=PD_COL)
    for bar in bars_hc:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.05*max(pd_vals+hc_vals),
                f"{h:.2f}" if metric=="RWBC" else f"{int(h)}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=HC_COL)
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(pd_vals+hc_vals)*1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Novel RWBC Keystones: Network Centrality & Connectivity\n"
             "Absiella dolichum and Lactonifactor longoviformis",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT}/fig_rwbc_novel_species_barchart.png", dpi=180, bbox_inches="tight")
plt.close()
print("Bar chart saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LACTONIFACTOR EGO-NETWORK
# ══════════════════════════════════════════════════════════════════════════════
LACTO_FEAT = ("metaphlan_counts::k__Bacteria|p__Firmicutes|c__Clostridia|"
              "o__Clostridiales|f__Clostridiaceae|g__Lactonifactor|"
              "s__Lactonifactor_longoviformis")

def short_name(feat: str) -> str:
    """Return genus + species abbreviation."""
    sp = feat.split("s__")[-1].replace("_", " ")
    parts = sp.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return sp

def build_ego(edge_csv: str, focal: str):
    df_e = pd.read_csv(edge_csv)
    G = nx.Graph()
    G.add_node(focal)
    for _, row in df_e.iterrows():
        if row.feature_a == focal or row.feature_b == focal:
            nbr = row.feature_b if row.feature_a == focal else row.feature_a
            G.add_edge(focal, nbr, weight=row.abs_correlation)
    return G

ABSIELLA_FEAT = ("metaphlan_counts::k__Bacteria|p__Firmicutes|c__Erysipelotrichia|"
                 "o__Erysipelotrichales|f__Erysipelotrichaceae|g__Absiella|"
                 "s__Absiella_dolichum")

G_pd    = build_ego(f"{OUT}/network_pd_edges_filtered.csv", LACTO_FEAT)
G_hc    = build_ego(f"{OUT}/network_hc_edges_filtered.csv", LACTO_FEAT)
G_pd_ab = build_ego(f"{OUT}/network_pd_edges_filtered.csv", ABSIELLA_FEAT)
G_hc_ab = build_ego(f"{OUT}/network_hc_edges_filtered.csv", ABSIELLA_FEAT)

def draw_ego(ax, G, focal, center_label, title, condition_color, nbr_color, max_show=43):
    """Draw ego-network with labels outside nodes, light-grey edges, tight layout."""
    nodes = list(G.nodes())
    nbrs  = [n for n in nodes if n != focal]

    if len(nbrs) > max_show:
        weights = {n: G[focal][n]["weight"] for n in nbrs if G.has_edge(focal, n)}
        nbrs = sorted(weights, key=weights.get, reverse=True)[:max_show]

    angle_step = 2 * np.pi / len(nbrs) if nbrs else 1
    radius = 1.1
    pos = {focal: np.array([0.0, 0.0])}
    for i, nbr in enumerate(nbrs):
        theta = i * angle_step - np.pi / 2
        pos[nbr] = np.array([radius * np.cos(theta), radius * np.sin(theta)])

    subG = G.subgraph([focal] + nbrs)

    nx.draw_networkx_edges(subG, pos, ax=ax,
                           edge_color=EDGE_COL, width=1.0, alpha=0.75)
    nx.draw_networkx_nodes(subG, pos, nodelist=nbrs, ax=ax,
                           node_size=140, node_color=nbr_color,
                           edgecolors="white", linewidths=0.6, alpha=0.85)
    nx.draw_networkx_nodes(subG, pos, nodelist=[focal], ax=ax,
                           node_size=9000, node_color=condition_color,
                           edgecolors="white", linewidths=2.5)

    label_offset = 0.18
    for node, (x, y) in pos.items():
        if node == focal:
            ax.text(x, y, center_label,
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color="white", zorder=5)
        else:
            dx, dy = x, y
            dist = np.hypot(dx, dy)
            ux, uy = (dx / dist, dy / dist) if dist > 0 else (0, 1)
            lx = x + ux * label_offset
            ly = y + uy * label_offset
            ha = "left" if ux > 0.1 else ("right" if ux < -0.1 else "center")
            va = "bottom" if uy > 0.1 else ("top" if uy < -0.1 else "center")
            ax.text(lx, ly, short_name(node), ha=ha, va=va, fontsize=5.0,
                    color="#222222", zorder=5)

    ax.set_title(f"{title}  (degree = {len(nbrs)})",
                 fontsize=10, fontweight="bold", color=condition_color, pad=6)
    ax.set_aspect("equal")
    ax.axis("off")
    if len([n for n in G.nodes() if n != focal]) > max_show:
        total = len([n for n in G.nodes() if n != focal])
        ax.text(0, -1.7, f"(Top {max_show} of {total} by |r|)",
                ha="center", va="top", fontsize=6, color="#666666", style="italic")


# 2×2 grid — same aspect ratio as the original 1×2 figure (16:8)
fig2, axes = plt.subplots(2, 2, figsize=(16, 8))
fig2.patch.set_facecolor("white")

# Row 0 — Lactonifactor longoviformis
draw_ego(axes[0, 0], G_hc, LACTO_FEAT,    "Lactonifactor\nlongoviformis",
         "Healthy Controls",     HC_COL, nbr_color=HC_COL, max_show=43)
draw_ego(axes[0, 1], G_pd, LACTO_FEAT,    "Lactonifactor\nlongoviformis",
         "Parkinson's Disease",  PD_COL, nbr_color=PD_COL, max_show=19)

# Row 1 — Absiella dolichum
draw_ego(axes[1, 0], G_hc_ab, ABSIELLA_FEAT, "Absiella\ndolichum",
         "Healthy Controls",     HC_COL, nbr_color=HC_COL, max_show=28)
draw_ego(axes[1, 1], G_pd_ab, ABSIELLA_FEAT, "Absiella\ndolichum",
         "Parkinson's Disease",  PD_COL, nbr_color=PD_COL, max_show=15)

# Row labels on the left margin
for row_idx, label in enumerate(["Lactonifactor longoviformis", "Absiella dolichum"]):
    fig2.text(0.01, 0.75 - row_idx * 0.5, label,
              va="center", ha="left", fontsize=10, fontweight="bold",
              rotation=90, color="#333333")

fig2.suptitle("Novel RWBC Keystone Species — 1-hop Ego-Networks\n"
              "Co-abundance neighbourhood collapse in Parkinson's Disease",
              fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0.03, 0, 1, 0.94])
fig2.savefig(f"{OUT}/fig_lactonifactor_ego_network.png", dpi=180, bbox_inches="tight")
plt.close()
print("Ego-network saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TABLE I-C  (trimmed columns)
# ══════════════════════════════════════════════════════════════════════════════
# Build table data manually from known statistics
table_data = {
    "Species": [
        "Absiella dolichum",
        "Lactonifactor longoviformis",
    ],
    "Prev.\nPD (%)": ["9.8", "13.9"],
    "Prev.\nHC (%)": ["12.8", "18.8"],
    "BH q": ["1.30e-3", "4.20e-3"],
    "Direction\nin PD": ["Depleted", "Depleted"],
    "RWBC\nPD": ["4.53\n(#12)", "5.57\n(#8)"],
    "RWBC\nHC": ["3.41\n(#13)", "5.27\n(#3)"],
    "Degree\nPD": ["15", "19"],
    "Degree\nHC": ["28", "43"],
    "Δ-RWBC": ["+1.12", "+0.30"],
}

# Quick check / override with real stats from CSV
eff_csv = str(_ROOT / "results" / "novel_species" / "novel_species_effect_sizes.csv")
try:
    eff = pd.read_csv(eff_csv)
    eff = eff[eff["feature"].str.contains("metaphlan_counts")]   # counts rows only
    for sp_key, sp_search in [("Absiella dolichum", "Absiella"), ("Lactonifactor longoviformis", "Lactonifactor")]:
        row = eff[eff["species"].str.contains(sp_search)]
        if not row.empty:
            r = row.iloc[0]
            idx = table_data["Species"].index(sp_key)
            table_data["Prev.\nPD (%)"][idx] = f"{r['prevalence_PD_pct']:.1f}"
            table_data["Prev.\nHC (%)"][idx] = f"{r['prevalence_ctrl_pct']:.1f}"
            table_data["BH q"][idx] = f"{r['wilcoxon_p']:.2e}"  # approximate
except Exception as e:
    print(f"[warning] could not load effect sizes: {e}")

# Load q-values if available
q_csv = str(_ROOT / "paper_figures" / "all_species_719_wilcoxon.csv")
try:
    qdf = pd.read_csv(q_csv)
    for sp_key, sp_search in [("Absiella dolichum", "Absiella_dolichum"),
                               ("Lactonifactor longoviformis", "Lactonifactor_longoviformis")]:
        row = qdf[qdf["feature"].str.contains(sp_search) & qdf["feature"].str.contains("metaphlan_counts")]
        if not row.empty:
            idx = table_data["Species"].index(sp_key)
            table_data["BH q"][idx] = f"{row.iloc[0]['qvalue_bh']:.2e}"
except Exception as e:
    print(f"[warning] could not load q-values: {e}")

col_labels = list(table_data.keys())
rows = list(zip(*table_data.values()))

fig3, ax3 = plt.subplots(figsize=(13, 3))
fig3.patch.set_facecolor("white")
ax3.axis("off")

col_widths = [0.20, 0.08, 0.08, 0.09, 0.10, 0.10, 0.10, 0.08, 0.08, 0.09]
tbl = ax3.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.2)

# Header styling
for j, _ in enumerate(col_labels):
    cell = tbl[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)

# Row styling (alternating)
row_colors = ["#FDECEA", "#EAF2FB"]  # light red, light blue
for i, row_vals in enumerate(rows):
    base_color = row_colors[i % 2]
    for j in range(len(col_labels)):
        cell = tbl[i+1, j]
        cell.set_facecolor(base_color)
        cell.set_edgecolor("#CCCCCC")
        if j == 0:
            cell.set_text_props(fontstyle="italic", fontsize=8.5)

ax3.set_title("Table I-C. Novel RWBC Keystone Species — Network Statistics",
              fontsize=11, fontweight="bold", pad=14, loc="left")

plt.tight_layout()
fig3.savefig(f"{OUT}/fig_table1c_rwbc_novel_species.png", dpi=180, bbox_inches="tight")
plt.close()
print("Table saved.")
print("All three figures done.")
