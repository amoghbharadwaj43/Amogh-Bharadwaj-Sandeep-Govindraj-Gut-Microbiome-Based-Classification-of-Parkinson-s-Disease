# Gut Microbiome-Based Classification of Parkinson's Disease — Reproducibility Bundle

**Paper:** *Gut Microbiome-Based Classification of Parkinson's Disease Using
Ensemble Learning and Graph-Theoretic Network Analysis.*
Amogh Bharadwaj, Sandeep Govindraj.

Everything needed to reproduce the **dual-branch pipeline** and every number,
table, and figure in the paper:

1. **AI branch** (`classification/`) — Decision Tree, Random Forest, XGBoost under
   nested 5×3 cross-validation, with clinical-feature ablation, label-permutation
   significance testing, and Gaussian-noise robustness.
2. **RWBC branch** (`network_analysis/`) — CLR-transformed Spearman co-abundance
   networks (PD vs. healthy) with Random Walk Betweenness Centrality keystone
   detection, ΔRWBC analysis, and species-level novel-keystone screening.

Primary cohort: **Wallen et al. 2022** (724 samples, 490 PD / 234 controls).
Cross-cohort validation: **Metcalfe-Roach et al. 2024** (276 samples).

---

## Quick start

```bash
# Python 3.11+ (macOS arm64 — a libomp runtime for XGBoost is vendored)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Run every command below from THIS directory (the repository root).
```

> **Large files (Git LFS).** The 148 MB raw data file, the two ~84 MB network
> edge lists, and the model `.joblib` binaries are tracked with
> [Git LFS](https://git-lfs.com/) (see `.gitattributes`), so they travel with
> the repo. Install Git LFS (`git lfs install`) before cloning so they download
> automatically. See [`data/README.md`](data/README.md).

---

## Repository layout

```
data/
  wallen_2022_primary_cohort/        primary cohort: raw + microbiome_V2[_clr].csv
  metcalfe_roach_2024_validation/    cross-cohort validation dataset
classification/                      AI branch (nested CV, ablation, permutation, models, noise)
network_analysis/                    RWBC branch (co-abundance, RWBC, species-level, ΔRWBC)
statistical_validation/              novel-species significance, cross-cohort, robustness checks
figures/                             figure-generation scripts
results/
  classification/   network/   statistics/   cross_cohort/   novel_species/   ablation/
paper_figures/                       final figure images (+ their source CSVs)
libomp_macos_arm64/                  vendored OpenMP runtime for XGBoost (auto-loaded)
README.md   requirements.txt   .gitignore
```

Every script resolves paths relative to the repository root, so the layout must
be preserved. Re-running a script overwrites the matching files under `results/`.

### Scripts

| Folder | Script | Produces |
|--------|--------|----------|
| `classification/` | `nested_cv_all_models.py` | **Table II** — nested 5×3 CV (DT/RF/XGB) |
| | `ablation_nested_cv.py` | **Table II-A** — Full / No-Constipation / Microbiome-Only |
| | `permutation_test_rf.py` | **Fig 10** — RF label-permutation (n=200) |
| | `permutation_test_nested_cv.py` | DT permutation under nested CV |
| | `model_pipeline_xgboost.py` | XGBoost SHAP features (XGB rank in Table I-B) |
| | `model_pipeline_random_forest.py`, `..._decision_tree.py` | single-split RF / DT importances |
| | `stress_test_gaussian_noise.py` | **Fig 11** family — noise robustness |
| `network_analysis/` | `coabundance_network.py` | PD / healthy co-abundance edge lists |
| | `random_walk_betweenness.py` | RWBC implementation (imported) |
| | `coabundance_rwbc_keystones.py` | **Figs 1-2** — keystone centralities |
| | `species_level_network_analysis.py` | novel keystones *Absiella*, *Lactonifactor* |
| | `build_rwbc_score_and_delta_tables.py` | **Fig 3** — ΔRWBC tables |
| `statistical_validation/` | `novel_species_wilcoxon.py`, `..._effect_sizes.py` | **Table I-B** — novel species stats |
| | `cross_cohort_concordance.py` | **Table I** — Wallen vs. Metcalfe-Roach |
| | `kingdom_feature_robustness.py` | kingdom-feature exclusion robustness (Methods) |
| | `feature_concentration_analysis.py` | DT feature-concentration robustness |
| `figures/` | `regenerate_novel_keystone_figures.py` | **Figs 4-5**, Table 1C panel |
| | `generate_pipeline_figures.py` | pipeline / classifier / ROC / network / ablation panels |
| | `generate_network_figures.py` | network + RWBC keystone + novel-feature panels |

---

## Reproduction order

Precomputed outputs are already in `results/`, so figure steps run immediately.
To regenerate from scratch:

```bash
# ── AI branch ───────────────────────────────────────────────────────────────
python classification/nested_cv_all_models.py        # Table II  → results/classification/
python classification/ablation_nested_cv.py          # Table II-A → results/ablation/
python classification/permutation_test_rf.py         # Fig 10 (RF perm, n=200)
python classification/permutation_test_nested_cv.py --n-permutations 200
python classification/model_pipeline_xgboost.py      # SHAP features
python classification/model_pipeline_random_forest.py
python classification/model_pipeline_decision_tree.py
python classification/stress_test_gaussian_noise.py  # Fig 11 family

# ── RWBC branch ─────────────────────────────────────────────────────────────
python network_analysis/coabundance_network.py            # edge lists (~84 MB each)
python network_analysis/coabundance_rwbc_keystones.py     # Figs 1-2
python network_analysis/species_level_network_analysis.py # novel keystones
python network_analysis/build_rwbc_score_and_delta_tables.py  # Fig 3

# ── Statistical validation ──────────────────────────────────────────────────
python statistical_validation/novel_species_wilcoxon.py       # Table I-B
python statistical_validation/novel_species_effect_sizes.py
python statistical_validation/cross_cohort_concordance.py     # Table I
python statistical_validation/kingdom_feature_robustness.py
python statistical_validation/feature_concentration_analysis.py

# ── Figures ─────────────────────────────────────────────────────────────────
python figures/regenerate_novel_keystone_figures.py   # Figs 4, 5, Table 1C → paper_figures/
python figures/generate_pipeline_figures.py
python figures/generate_network_figures.py
```

---

## Where each result comes from

| Paper item | Script | Output |
|------------|--------|--------|
| **Table II** | `classification/nested_cv_all_models.py` | `results/classification/nested_cv_all_models_summary.csv` (RF .753/.789, XGB .725/.771, DT .675/.626) |
| **Table II-A** | `classification/ablation_nested_cv.py` | `results/ablation/ablation_nested_cv_results.csv` |
| **Table I** | `statistical_validation/cross_cohort_concordance.py` | `results/cross_cohort/concordance_summary.csv` |
| **Table I-B** | `statistical_validation/novel_species_*` | `results/novel_species/novel_species_*.csv` |
| **Figs 1-2** | network branch | `results/network/coabundance_rwbc_*_centrality.csv` |
| **Fig 3** | `build_rwbc_score_and_delta_tables.py` | `results/statistics/rwbc_delta_table.csv` |
| **Figs 4-5** | `figures/regenerate_novel_keystone_figures.py` | `paper_figures/fig_rwbc_novel_species_barchart.png`, `fig_lactonifactor_ego_network.png` |
| **Fig 6** | nested-CV RF stability | `paper_figures/fig4_rf_feature_importance.png` |
| **Fig 7** | RF stable features + FDR | `paper_figures/fig5_rf_feature_significance.png` |
| **Figs 8-9** | `figures/generate_pipeline_figures.py` | `results/classification/nested_cv_all_models_*_oof_predictions.csv` |
| **Fig 10** | `classification/permutation_test_rf.py` | `paper_figures/fig8_rf_permutation_test.png`, `results/classification/permutation_test_rf_summary.json` |
| **Fig 11** | RF Gaussian-noise stress test | `paper_figures/fig9_rf_gaussian_noise.png` |

**Figure note.** `paper_figures/` holds the exact images used in the paper.
`regenerate_novel_keystone_figures.py` fully reproduces three of them; the other
RF panels (`fig4_*`, `fig5_*`, `fig8_*`, `fig9_*`) are provided as final images
with all of their source data (the CSVs in `paper_figures/` and the tables in
`results/`), so the underlying numbers are fully reproducible. The
`paper_figures/figN_*` filenames use an internal numbering that differs from the
paper's figure numbers — use the mapping above.

---

## Notes

- **Determinism.** `random_state=42` throughout; nested CV uses a shared
  outer-fold seed so DT/RF/XGBoost see identical splits. SHAP/Optuna and the RWBC
  random-walk approximation may vary negligibly run-to-run.
- **macOS / XGBoost.** `libomp_macos_arm64/libomp.dylib` is loaded automatically
  via `ctypes`. On Linux, install `libomp`/`libgomp` from your package manager;
  the vendored dylib is then ignored.
- **No stale artifacts.** Single-split threshold-tuning and n=100 permutation
  outputs from earlier iterations (which reported different numbers) have been
  removed so every file in `results/` is consistent with the paper.

## Citations

- Wallen ZD et al. Nature Communications 13, 6958 (2022).
- Metcalfe-Roach A et al. Mov. Disord. 39 (2024).
- Brandes U, Fleischer D. *Centrality measures based on current flow.* Proc. STACS, 2005.
