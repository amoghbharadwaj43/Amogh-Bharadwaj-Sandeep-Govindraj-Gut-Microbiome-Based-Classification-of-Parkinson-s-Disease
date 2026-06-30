# Data

| Subfolder | Contents |
|-----------|----------|
| `wallen_2022_primary_cohort/` | Primary cohort (Wallen et al. 2022): raw `Source_Data_24Oct2022_prepped.csv` + the `microbiome_V2*.csv` feature/CLR matrices |
| `metcalfe_roach_2024_validation/` | Cross-cohort validation cohort (Metcalfe-Roach et al. 2024) + its published R/Python scripts and supplementary data |

## `wallen_2022_primary_cohort/`

| File | Size | Used by |
|------|------|---------|
| `Source_Data_24Oct2022_prepped.csv` | ~148 MB | `classification/nested_cv_all_models.py`, `ablation_nested_cv.py`, `permutation_test_rf.py`, `model_pipeline_xgboost.py`, `statistical_validation/novel_species_*.py`, `kingdom_feature_robustness.py`, `feature_concentration_analysis.py` |
| `microbiome_V2.csv` | ~40 MB | `classification/model_pipeline_random_forest.py`, `model_pipeline_decision_tree.py`, `stress_test_gaussian_noise.py` |
| `microbiome_V2_clr.csv` | ~49 MB | `network_analysis/coabundance_network.py`, `species_level_network_analysis.py` |

Target column: `Case_status=PD` (or `Case_status_PD` in the CLR matrix).

**These three files are tracked with Git LFS** (see `.gitattributes`), so they
travel with the repo — run `git lfs install` before cloning. Original source:

- Wallen ZD et al. *Metagenomics of Parkinson's disease implicates the gut
  microbiome in multiple disease mechanisms.* Nature Communications 13, 6958
  (2022). Source Data: https://doi.org/10.1038/s41467-022-34667-x

Place them at `data/wallen_2022_primary_cohort/<exact filename>`.
`microbiome_V2.csv` is a feature-filtered variant of the raw source data;
`microbiome_V2_clr.csv` is its centred log-ratio (CLR) transform.

## `metcalfe_roach_2024_validation/`

Used by `statistical_validation/cross_cohort_concordance.py`, which reads three
supplementary spreadsheets:

- `Results/Supplementary Data/Differential Abundance/Taxonomy_Abundance_and_Prevalence.xlsx`
- `Results/Supplementary Data/Random Forest/RF Importance Values - Status.xlsx`
- `Results/Supplementary Data/Network Analysis/Network Analysis Species-Level Data.xlsx`

Source: Metcalfe-Roach A et al. *Metagenomic analysis reveals large-scale
disruptions of the gut microbiome in Parkinson's disease.* Mov. Disord. 39 (2024).
