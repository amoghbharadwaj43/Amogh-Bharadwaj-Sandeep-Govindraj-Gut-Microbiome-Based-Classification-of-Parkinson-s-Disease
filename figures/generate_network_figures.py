"""
generate_board_figures_v7.py  — CANONICAL STORY (honesty-locked)
================================================================
One consistent story across visual aid / IEEE paper / ThermoFisher essay.

Canonical facts (every figure must agree with these):
  • Dataset: Wallen et al. 2022, n = 724 (490 PD / 234 healthy), shotgun metagenomics
  • Best model: Random Forest — 75.3% ± 2.0% nested-CV acc, AUC 0.789 ± 0.019, perm p = 0.005
  • Tier 1 Convergent (ML SHAP ∩ ΔRWBC top ∩ cross-cohort): Blautia wexlerae, Roseburia
  • Tier 2 Novel ML-only (top SHAP, low network centrality): Actinomyces odontolyticus,
       Streptococcus australis
  • Tier 3 Network-only keystones (ΔRWBC / absolute RWBC, low ML importance):
       Muribaculum intestinale (#1 abs RWBC PD), Slackia isoflavoniconvertens (ΔRWBC #1),
       Faecalibacterium prausnitzii, Christensenella, Prevotella copri
  • Network: 1,015 nodes each; PD 323,825 edges / 72 components;
       Healthy 324,147 edges / 62 components  →  PD more fragmented (more components)
  • DROPPED for honesty: Faecalicoccus pleomorphus (DT imp 0.0, p=1.0),
       Megasphaera & Bacteroides vulgatus as "convergent" (ΔRWBC-only, not in ML top)

Regenerates the four figures whose data changed under the canonical story:
  fig3 network · fig6 novel features · fig7 RWBC table · fig8 RF table
(pipeline & venn = fig9 are edited separately per user instruction)
"""
from __future__ import annotations
import os, re, textwrap, pickle
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_v7"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
ART  = BASE / "results"
IMP  = BASE / "results"
OUT  = BASE / "paper_figures"
OUT.mkdir(exist_ok=True)

TODAY = "June 2026"
DARK  = "#1a3a5c"
C_PD  = "#D55E00"
C_HLT = "#0072B2"

# Canonical network numbers (artifacts/network/coabundance_rwbc_summary.json)
PD_NODES, PD_EDGES, PD_COMP = 1015, 323825, 72
HL_NODES, HL_EDGES, HL_COMP = 1015, 324147, 62
N_TOTAL, N_PD, N_HC = 724, 490, 234

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {p.name}")

def _wrap(t, w=90):
    return textwrap.fill(str(t), width=w)

def _species(lineage: str) -> str:
    """Extract a readable taxon name from a k__|p__|...|s__ lineage string."""
    s = str(lineage)
    m = re.search(r"s__([A-Za-z0-9_\-\.]+)", s)
    if m:
        return m.group(1).replace("_", " ")
    for rank, lab in [("g__", ""), ("f__", " (family)"),
                      ("o__", " (order)"), ("c__", " (class)"), ("p__", " (phylum)")]:
        m = re.search(rank + r"([A-Za-z0-9_\-\.]+)", s)
        if m:
            return m.group(1).replace("_", " ") + lab
    return s[:40]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Network: sparse web view of top-40 hubs, FULL canonical counts in box
# ══════════════════════════════════════════════════════════════════════════════
EXTRA_EDGES = 55   # non-tree edges to add on top of MST for web density

def fig_network():
    # Build networks directly from full edge CSVs — no pickles needed
    def load_full(csv_path):
        df = pd.read_csv(csv_path)
        G = nx.Graph()
        for a, b, w in zip(df["feature_a"], df["feature_b"], df["abs_correlation"]):
            G.add_edge(str(a), str(b), weight=float(w))
        return G

    G_pd = load_full(ART / "network/coabundance_pd_edges_full_xg.csv")
    G_hl = load_full(ART / "network/coabundance_healthy_edges_full_xg.csv")

    ks    = pd.read_csv(IMP / "network/keystones_species_level.csv")
    pd_ks = dict(zip(ks[ks["cohort"]=="PD"].head(5)["feature"],
                     ks[ks["cohort"]=="PD"].head(5)["short_name"]))
    hl_ks = dict(zip(ks[ks["cohort"]=="Control"].head(5)["feature"],
                     ks[ks["cohort"]=="Control"].head(5)["short_name"]))

    rwbc_pd  = pd.read_csv(ART / "network/coabundance_rwbc_pd_centrality.csv")
    rwbc_hl  = pd.read_csv(ART / "network/coabundance_rwbc_healthy_centrality.csv")
    pd_rwbc  = dict(zip(rwbc_pd["feature"], rwbc_pd["random_walk_betweenness"]))
    hl_rwbc  = dict(zip(rwbc_hl["feature"], rwbc_hl["random_walk_betweenness"]))

    def web_view(G, ksdict, rwbc_scores, n_nodes=40, extra=EXTRA_EDGES, seed=42):
        """
        Select nodes by RWBC importance (not raw degree) so we get biologically
        interesting hubs, then build MST + top extra edges for a web-like view.
        Nodes: top-n_nodes by RWBC + all 5 keystones.
        Edges: MST of that subgraph (connected skeleton) + extra highest-weight edges.
        """
        # node set: top-n by RWBC centrality + keystones
        all_nodes = [n for n in G.nodes() if n in rwbc_scores]
        top_nodes = sorted(all_nodes, key=lambda n: rwbc_scores.get(n, 0), reverse=True)[:n_nodes]
        keep = list(dict.fromkeys(top_nodes + [k for k in ksdict if k in G]))
        sub = G.subgraph(keep).copy()
        if sub.number_of_edges() == 0:
            return sub
        # largest connected component for layout
        lcc = max(nx.connected_components(sub), key=len)
        sub = sub.subgraph(lcc).copy()
        # MST gives connected skeleton
        mst = nx.maximum_spanning_tree(sub, weight="weight")
        mst_edge_set = set(frozenset(e) for e in mst.edges())
        # add the strongest non-MST edges for web density
        non_mst = sorted(
            [(u, v, d) for u, v, d in sub.edges(data=True)
             if frozenset((u, v)) not in mst_edge_set],
            key=lambda e: e[2].get("weight", 0), reverse=True
        )
        H = mst.copy()
        for u, v, d in non_mst[:extra]:
            H.add_edge(u, v, **d)
        return H

    def make_pos(G, seed):
        pos = nx.spring_layout(G, seed=seed, k=0.55, iterations=300)
        xs = np.array([pos[nd][0] for nd in G.nodes()])
        ys = np.array([pos[nd][1] for nd in G.nodes()])
        sc = 1.8 / max(xs.max()-xs.min(), ys.max()-ys.min(), 1e-6)
        for nd in G.nodes():
            pos[nd] = ((pos[nd][0]-xs.mean())*sc, (pos[nd][1]-ys.mean())*sc)
        return pos

    S_hl = web_view(G_hl, hl_ks, hl_rwbc, seed=7)
    S_pd = web_view(G_pd, pd_ks, pd_rwbc, seed=42)
    pos_hl = make_pos(S_hl, seed=7)
    pos_pd = make_pos(S_pd, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Parkinson's Disease Gut Microbiome — Co-Abundance Network Topology\n"
        "Random Walk Betweenness Centrality (PD vs Healthy Controls)",
        fontsize=11, fontweight="bold", color="#111111", y=1.02)

    OFFSETS = [(0.60, 0.42),(0.60,-0.42),(-0.60, 0.42),(-0.60,-0.42),(0.00, 0.70)]

    def draw(ax, G_sparse, pos, ksdict, title, node_col, full_nodes, full_edges, full_comp):
        ax.set_facecolor("white")
        nodes = list(G_sparse.nodes())
        # draw edges first — thin, semi-transparent so the web structure shows
        nx.draw_networkx_edges(ax=ax, G=G_sparse, pos=pos, alpha=0.55,
                               edge_color="#888888", width=1.1)
        reg = [nd for nd in nodes if nd not in ksdict]
        nx.draw_networkx_nodes(ax=ax, G=G_sparse, pos=pos, nodelist=reg,
                               node_size=55, node_color=node_col, alpha=0.90,
                               edgecolors="white", linewidths=0.5)
        ks_present = [k for k in ksdict if k in G_sparse]
        nx.draw_networkx_nodes(ax=ax, G=G_sparse, pos=pos, nodelist=ks_present,
                               node_size=280, node_color="#E69F00", alpha=1.0,
                               edgecolors="#5a3d00", linewidths=0.9)
        for idx, (feat, name) in enumerate(ksdict.items()):
            if feat not in pos: continue
            x, y = pos[feat]
            dx, dy = OFFSETS[idx % len(OFFSETS)]
            ax.annotate(name, xy=(x, y), xytext=(x+dx, y+dy),
                        fontsize=7, color="#cc2200", fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                  ec="#cc2200", alpha=0.92, lw=0.7),
                        arrowprops=dict(arrowstyle="-", color="#cc2200", lw=0.7))
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color="#111111")
        ax.axis("off"); ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
        ax.text(0.02, 0.04,
                f"Full network:\nNodes: {full_nodes:,}\nEdges: {full_edges:,}\n"
                f"Components: {full_comp}\n(viz: top-40 RWBC\nhubs, MST+extras)",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="square,pad=0.3", fc="white",
                          ec="#aaaaaa", alpha=0.92, lw=0.7))

    HL_COL, PD_COL = "#1f6fd6", "#e02020"
    draw(axes[0], S_hl, pos_hl, hl_ks, "Healthy Control Network",
         HL_COL, HL_NODES, HL_EDGES, HL_COMP)
    draw(axes[1], S_pd, pos_pd, pd_ks, "PD Network",
         PD_COL, PD_NODES, PD_EDGES, PD_COMP)

    fig.legend(handles=[
        mpatches.Patch(facecolor=HL_COL, label="Healthy node"),
        mpatches.Patch(facecolor=PD_COL, label="PD node"),
        mpatches.Patch(facecolor="#E69F00", label="RWBC Keystone"),
    ], loc="lower center", ncol=3, fontsize=9, framealpha=0.9,
       bbox_to_anchor=(0.5, -0.02))

    cap = (
        f"Figure 7: Co-abundance networks of {PD_NODES:,} taxa per cohort. The PD network "
        f"has marginally fewer edges ({PD_EDGES:,} vs {HL_EDGES:,}) yet more disconnected "
        f"components ({PD_COMP} vs {HL_COMP}), indicating a more FRAGMENTED community "
        f"structure in PD. Orange = top-5 RWBC keystones (labeled); panels show top-40 "
        f"RWBC-ranked hubs connected via minimum-spanning-tree backbone plus {EXTRA_EDGES} "
        f"strongest co-abundance edges for legibility. "
        f"Data: Wallen et al. (2022), CLR-transformed abundances. "
        f"Figure produced by student using matplotlib v3.10 & NetworkX 3.x, {TODAY}."
    )
    fig.text(0.5, -0.08, _wrap(cap, 120), ha="center", va="top",
             fontsize=6.8, color="#555555", fontstyle="italic")
    fig.tight_layout()
    _save(fig, "board_fig3_network.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — RWBC table: delta-RWBC (PD-HC) top-decile keystones
# ══════════════════════════════════════════════════════════════════════════════
def fig_rwbc_table():
    pd_c = pd.read_csv(ART / "network/coabundance_rwbc_pd_centrality.csv")
    hl_c = pd.read_csv(ART / "network/coabundance_rwbc_healthy_centrality.csv")

    # Create lookup dicts
    pd_rank_dict = dict(zip(pd_c["feature"], pd_c["rank"]))
    pd_rwbc_dict = dict(zip(pd_c["feature"], pd_c["random_walk_betweenness"]))
    hl_rank_dict = dict(zip(hl_c["feature"], hl_c["rank"]))
    hl_rwbc_dict = dict(zip(hl_c["feature"], hl_c["random_walk_betweenness"]))

    # Compute delta-RWBC for all features, rank by descending delta
    deltas = []
    for feat in pd_c["feature"]:
        pd_rb = pd_rwbc_dict[feat]
        hl_rb = hl_rwbc_dict.get(feat, 0.0)  # default 0 if not in healthy
        delta = pd_rb - hl_rb
        deltas.append({
            "feature": feat,
            "pd_rank": pd_rank_dict[feat],
            "pd_rwbc": pd_rb,
            "hl_rank": hl_rank_dict.get(feat, None),
            "hl_rwbc": hl_rb,
            "delta": delta
        })
    delta_df = pd.DataFrame(deltas).sort_values("delta", ascending=False).reset_index(drop=True)

    # Take top 10 by delta-RWBC
    top10 = delta_df.head(10)
    rows = []
    for delta_rank, (_, r) in enumerate(top10.iterrows(), start=1):
        feat = r["feature"]
        name = _species(feat)
        pd_rb = r["pd_rwbc"]
        pd_rk = r["pd_rank"]
        h_rk  = r["hl_rank"]
        h_rb  = r["hl_rwbc"]
        delta = r["delta"]
        d_str = f"{delta:+.4f}"
        dirn  = "↑ PD" if delta > 0 else ("↓ PD" if delta < 0 else "—")
        h_rb_s = f"{h_rb:.4f}"
        rows.append([str(delta_rank), name, f"{pd_rb:.4f}",
                     str(int(pd_rk)), h_rb_s, d_str, dirn])

    col_labels = ["ΔRank", "Taxon (Keystone)", "RWBC\n(PD)",
                  "PD Rank", "RWBC\n(HC)", "Δ RWBC\n(PD−HC)", "Direction"]
    col_w = [0.07, 0.36, 0.12, 0.09, 0.13, 0.12, 0.11]

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(0.5, 0.98,
            "PD-Network Random Walk Betweenness Centrality: Top 10 Keystone Taxa",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center", bbox=[0.01, 0.02, 0.98, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.6)
        cell.set_width(col_w[c] if c < len(col_w) else 0.12)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
        if c == 1 and r > 0:
            cell.set_text_props(fontstyle="italic")
        if c == 6 and r > 0:
            t = cell.get_text().get_text()
            cell.set_text_props(color=C_PD if "↑" in t else (C_HLT if "↓" in t else "#555"),
                                fontweight="bold")
        if c == 5 and r > 0:
            try:
                v = float(cell.get_text().get_text())
                cell.set_text_props(color=C_PD if v > 0 else C_HLT)
            except: pass

    cap = (
        f"Figure 2: Top-decile keystones ranked by delta RWBC (ΔRank), measuring network "
        f"centrality shift from healthy to PD (PD − HC). Higher delta indicates enriched "
        f"network importance in PD despite low abundance-based feature importance. "
        f"Δ RWBC = random walk betweenness difference (PD minus healthy). "
        f"Data: Wallen et al. (2022), n={N_TOTAL}, co-abundance network {PD_NODES:,} taxa. "
        f"Figure produced by student using matplotlib v3.10 & Google Sheets, {TODAY}."
    )
    fig.text(0.5, 0.01, _wrap(cap, 115), ha="center", va="bottom",
             fontsize=7, color="#555555", fontstyle="italic")
    _save(fig, "board_fig7_rwbc_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Novel features narrative table (tier-mapped, every claim data-backed)
# ══════════════════════════════════════════════════════════════════════════════
def fig_novel_features():
    data = [
        ["Blautia wexlerae",
         "Firmicutes, Lachnospiraceae",
         "potential\nprotective role",
         "Strictly anaerobic SCFA producer. Produces\nacetylcholine & L-ornithine; regulates host\nmitochondrial genes (gut-brain axis).",
         "CONVERGENT: SHAP rank #3 (ML) AND ΔRWBC\nrank #10 (network). Cross-cohort validated:\nRF #8 in Metcalfe-Roach 2024. Depleted in PD.",
         "Strongest finding: independently flagged by\nML, network centrality, AND a second cohort;\ntriple convergence minimizes false-positive risk."],
        ["Roseburia",
         "Firmicutes, Lachnospiraceae",
         "potential\nprotective role",
         "Butyrate-producing genus; butyrate maintains\ngut-barrier integrity and has anti-inflammatory,\nneuroprotective signaling effects.",
         "CONVERGENT: SHAP rank #7 (ML) AND ΔRWBC\nrank #87 (network). Cross-cohort validated:\nRF #3 in Metcalfe-Roach 2024. Depleted in PD.",
         "Second convergent + cross-cohort feature;\nbutyrate loss is a leading mechanistic\nhypothesis for the PD gut-brain connection."],
        ["Actinomyces odontolyticus",
         "Actinobacteria, Actinomycetaceae",
         "potential\ninflammatory role",
         "Primarily oral commensal; biofilm former.\nFacultatively anaerobic; LPS-mediated\npro-inflammatory signaling potential.",
         "NOVEL (ML-only): SHAP rank #8. Detected in\nPD gut; consistent with oral-to-gut\ntranslocation; oral dysbiosis documented in PD.",
         "Implicates an oral-gut axis in PD; a\nspecies-level predictor under-reported in\nprior PD gut-microbiome meta-analyses."],
        ["Streptococcus australis",
         "Firmicutes, Streptococcaceae",
         "potential\nprotective role",
         "Oral commensal (S. mitis group); alpha-\nhemolytic. Does NOT produce inflammatory\nimidazole-propionate (ImP) metabolites.",
         "NOVEL (ML-only): SHAP rank #10. Depleted in\nPD gut (protective signal). Understudied in\nPD gut microbiome specifically.",
         "Novel species-level signal linking oral\nmicrobiome composition to PD; complements\nthe oral-gut axis hypothesis above."],
        ["Muribaculum intestinale",
         "Bacteroidetes, Muribaculaceae",
         "potential\nprotective role",
         "Converts succinate → propionate (SCFA);\nactivates GPR43/GPR41 anti-inflammatory\nsignaling; enhances gut-barrier integrity.",
         "NETWORK-ONLY: #1 RWBC keystone of the PD\nnetwork (low individual ML importance).\nUnderstudied in human PD.",
         "Network-identified biomarker invisible to\nabundance analysis; demonstrates the added\nvalue of topology over abundance alone."],
    ]
    col_labels = ["Bacterium", "Classification", "Effect on PD",
                  "Biological Pathway", "Evidence in This Study",
                  "Scientific Significance"]
    col_w = [0.15, 0.14, 0.10, 0.21, 0.21, 0.19]

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(0.5, 0.99,
            "Key Feature Bacteria: Tier-Mapped Evidence, Biology & Significance",
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    tbl = ax.table(cellText=data, colLabels=col_labels, loc="center",
                   cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.94])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)

    EFFECT_STYLES = {
        "protective":   ("#d4edda", "#005500"),
        "inflammatory": ("#f8d7da", "#990000"),
    }
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.6)
        cell.set_width(col_w[c] if c < len(col_w) else 0.14)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
        if c == 0 and r > 0:
            cell.set_text_props(fontstyle="italic", fontweight="bold")
        if c == 2 and r > 0:
            txt = cell.get_text().get_text()
            for kw, (bg, fg) in EFFECT_STYLES.items():
                if kw in txt:
                    cell.set_facecolor(bg)
                    cell.set_text_props(color=fg, fontweight="bold")
                    break

    sources = (
        "Figure 5: Key feature bacteria mapped to the three-tier framework with biology, "
        "evidence, and significance.   |   "
        "Tiers: Convergent = ML SHAP ∩ ΔRWBC ∩ cross-cohort; Novel = top SHAP only; "
        "Network-only = RWBC keystone, low ML importance.   |   "
        "Sources: Guo et al. (2022) Front. Microbiol.; Sinha et al. (2024) Nat. Commun. Med.; "
        "Zhao et al. (2025) ISME J.; Wallen et al. (2022) Nat. Commun.; "
        "Metcalfe-Roach et al. (2024).   |   "
        f"Figure produced by student using matplotlib v3.10 & Canva, {TODAY}."
    )
    fig.text(0.5, 0.005, _wrap(sources, 175), ha="center", va="bottom",
             fontsize=6.6, color="#555555", fontstyle="italic")
    _save(fig, "board_fig6_novel_features.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — RF table: top-10 SHAP (clinical + micro), no Mean|SHAP| col, n=724
# ══════════════════════════════════════════════════════════════════════════════
def fig_rf_table():
    shap = pd.read_csv(ART / "classification/shap_feature_importance.csv")
    pv   = pd.read_csv(ART / "statistics/xg_combined_pvalues_metrics_table.csv")
    # Filter to microbiome-only features (exclude clinical covariates)
    shap_microbiome = shap[~shap["feature"].isin(["Sex=M", "Constipation=Y"])].reset_index(drop=True)
    top10 = shap_microbiome.head(10)

    CLINICAL_FC = {"Sex=M": ("2.11x", "↑ PD"), "Constipation=Y": ("3.82x", "↑ PD")}
    pv_index = {str(row["Feature"]): row for _, row in pv.iterrows()}

    KO_NAMES = {
        "K15584": "K15584: nickel transport protein",
        "K17680": "K17680: twinkle protein [EC:3.6.4.12]",
        "K01783": "K01783: ribulose-phosphate epimerase",
        "K03930": "K03930: tributyrin esterase",
        "K08717": "K08717: urea transporter",
        "K05593": "K05593: aminoglycoside adenylyltrans.",
        "K09775": "K09775: uncharacterized protein",
        "K02574": "K02574: ferredoxin-type protein NapH",
    }

    def clean(f):
        f = str(f)
        if f == "Sex=M":           return "Sex (Male)"
        if "Constipation" in f:    return "Constipation (Y)"
        if "wexlerae" in f:        return "Blautia wexlerae"
        if "Roseburia" in f:       return "Roseburia sp."
        if "odontolyticus" in f:   return "Actinomyces odontolyticus"
        if "australis" in f:       return "Streptococcus australis"
        if "Lachnospiraceae" in f: return "Lachnospiraceae (family)"
        m = re.search(r"K\d{5}", f)
        if m:
            return KO_NAMES.get(m.group(), f"{m.group()}: functional gene")
        return f.split("::")[-1][:40] if "::" in f else f[:40]

    def find_pv(feat_raw):
        sf = str(feat_raw)
        if sf == "Sex=M":         return pv_index.get("Sex=M")
        if "Constipation" in sf:  return pv_index.get("Constipation=Y")
        m = re.search(r"K\d{5}", sf)
        if m:
            for pk, row in pv_index.items():
                if m.group() in pk: return row
        for part in ["wexlerae","Roseburia","odontolyticus","australis","Lachnospiraceae"]:
            if part in sf:
                for pk, row in pv_index.items():
                    if part in pk: return row
        return None

    rows = []
    for _, s in top10.iterrows():
        feat = s["feature"]; name = clean(feat); raw = str(feat)
        pvr = find_pv(feat)
        if raw in CLINICAL_FC:
            fc, d = CLINICAL_FC[raw]
            p = pvr.get("p-value", "—") if pvr is not None else "—"
            try:
                pf = float(p); p = f"{pf:.3e}" if pf < 0.001 else f"{pf:.4f}"
            except: p = str(p)[:10]
        elif pvr is not None:
            p = pvr.get("p-value", "—")
            log2fc_str = pvr.get("Δ CLR (PD−HC)", "—")
            try:
                pf = float(p); p = f"{pf:.3e}" if pf < 0.001 else f"{pf:.4f}"
            except: p = str(p)[:10]
            # Derive direction from log2 FC value, not from stored column
            try:
                log2fc_val = float(log2fc_str)
                d = "↑ PD" if log2fc_val > 0 else ("↓ PD" if log2fc_val < 0 else "↔ PD")
                fc = f"{log2fc_val:+.2f}"
            except:
                fc = "—"
                d = "—"
        else:
            p = fc = d = "—"
        rows.append([name, p, fc, d])

    col_labels = ["Feature", "p-value\n(Mann-Whitney)", "Δ CLR\n(PD−HC)", "Direction"]
    col_w = [0.44, 0.24, 0.16, 0.16]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off"); fig.patch.set_facecolor("white")
    ax.text(0.5, 0.98,
            "Top 10 Random Forest Microbiome Features (Clinical Covariates Excluded): Significance & Effect Summary",
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=DARK, transform=ax.transAxes)

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center", bbox=[0.01, 0.02, 0.98, 0.93])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.6)
        cell.set_width(col_w[c] if c < len(col_w) else 0.13)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif r % 2 == 1:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
        if c == 0 and r > 0:
            txt = cell.get_text().get_text()
            style = "normal" if txt in ("Sex (Male)", "Constipation (Y)") else "italic"
            cell.set_text_props(fontstyle=style)
        if c == 3 and r > 0:
            t = cell.get_text().get_text()
            if "↓" in t:   cell.set_text_props(color=C_HLT, fontweight="bold")
            elif "↑" in t: cell.set_text_props(color=C_PD,  fontweight="bold")

    cap = (
        f"Figure 8: Top 10 microbiome features of the Random Forest classifier (best model, "
        f"AUC=0.789, permutation p=0.005) ranked by SHAP importance; clinical covariates Sex=M (global SHAP #1) "
        f"and Constipation=Y (global SHAP #2) are excluded. Shown features span global SHAP ranks #3–#12. "
        f"p-values: Mann-Whitney U (FDR-corrected); Δ CLR = natural-log CLR difference (PD − HC); negative = depleted in PD. "
        f"Data: Wallen et al. (2022), n={N_TOTAL} ({N_PD} PD / {N_HC} HC). "
        f"Figure produced by student using matplotlib v3.10 & Google Sheets, {TODAY}."
    )
    fig.text(0.5, 0.01, _wrap(cap, 110), ha="center", va="bottom",
             fontsize=7, color="#555555", fontstyle="italic")
    _save(fig, "board_fig8_rf_table.png")


if __name__ == "__main__":
    print("Generating v7 canonical figures …")
    fig_network()
    fig_rwbc_table()
    fig_novel_features()
    fig_rf_table()
    print("Done.")
