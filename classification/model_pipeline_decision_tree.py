import os
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "wallen_2022_primary_cohort", "microbiome_V2.csv")
TARGET_COLUMN = "Case_status_PD"
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "results", "classification")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(DATA_PATH)
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {DATA_PATH}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")
exclude_cols = [
    col for col in X.columns
    if any(substr in col.lower() for substr in EXCLUDE_SUBSTRINGS)
]
if exclude_cols:
    print(f"Dropping {len(exclude_cols)} columns by name filter.")
    X = X.drop(columns=exclude_cols)

feature_names = X.columns.tolist()

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# IMPUTE MISSING VALUES
# -----------------------------

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index,
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_names,
    index=X_test.index,
)

# -----------------------------
# STEP 1: FIND TOP 20 FEATURES
# -----------------------------

print("Step 1: Training on all features to find top 20...")

variance_filter = VarianceThreshold(threshold=0.0001)
X_train_filtered = variance_filter.fit_transform(X_train)
X_test_filtered = variance_filter.transform(X_test)

selected_mask = variance_filter.get_support()
selected_features = np.array(feature_names)[selected_mask]

base_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)
base_model.fit(X_train_filtered, y_train)

importances = base_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {
        "feature": selected_features,
        "importance": importances,
    }
).sort_values(by="importance", ascending=False)

# Preserve tree-based importances for reference
tree_importance_df = feature_importance_df.copy()

# If one feature dominates, drop it for top-10 selection to avoid a trivial tree
DOMINANCE_THRESHOLD = 0.95
total_importance = importances.sum()
dominant_feature = None
dominance_ratio = (importances.max() / total_importance) if total_importance > 0 else 0.0

if dominance_ratio >= DOMINANCE_THRESHOLD:
    dominant_feature = feature_importance_df.iloc[0]["feature"]
    print(
        f"Dominant feature detected ({dominant_feature}) "
        f"with {dominance_ratio * 100:.1f}% of total importance. "
        "Excluding it from top-20 selection."
    )
    # Use mutual information for ranking to avoid a single-feature tree bias
    mi_scores = mutual_info_classif(X_train_filtered, y_train, random_state=42)
    feature_importance_df = pd.DataFrame(
        {
            "feature": selected_features,
            "importance": mi_scores,
        }
    ).sort_values(by="importance", ascending=False)
    print("Using mutual information for feature ranking due to dominance.")

    filtered_importance_df = feature_importance_df[
        feature_importance_df["feature"] != dominant_feature
    ]
else:
    filtered_importance_df = feature_importance_df

# Pick the top 20 (or fewer if not enough features)
top_10_features = filtered_importance_df["feature"].head(20).tolist()

print("\nTop 20 Most Important Features:")
for i, (idx, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.6f}")

print("\nTop 20 Features Used for Training:")
for i, feat in enumerate(top_10_features, 1):
    print(f"{i}. {feat}")

# -----------------------------
# STEP 2: TRAIN MODEL ON TOP 20 ONLY
# -----------------------------

print("\n" + "=" * 50)
print("Step 2: Training final model on top 20 features only...")
print("=" * 50)

X_train_top10 = X_train[top_10_features]
X_test_top10 = X_test[top_10_features]
X_train_top10_np = X_train_top10.values
X_test_top10_np = X_test_top10.values

# -----------------------------
# CROSS-VALIDATION SETUP
# -----------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# -----------------------------
# OPTUNA HYPERPARAMETER TUNING
# -----------------------------

print("\nRunning Optuna hyperparameter tuning (100 trials)...")

OPTUNA_TRIALS = 150

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 60),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 40),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.06),
        "random_state": 42,
        "class_weight": "balanced",
    }

    model = DecisionTreeClassifier(**params)
    scores = cross_val_score(
        model,
        X_train_top10_np,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    return scores.mean()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

best_params = study.best_params
print(f"Best CV Accuracy from Optuna: {study.best_value:.4f}")
print(f"Best Params: {best_params}")

# -----------------------------
# FINAL MODEL WITH BEST PARAMS
# -----------------------------

DT_PARAMS = {
    "random_state": 42,
    "class_weight": "balanced",
}
DT_PARAMS.update(best_params)

pipeline = Pipeline(
    steps=[
        (
            "model",
            DecisionTreeClassifier(**DT_PARAMS),
        ),
    ]
)

cv_scores = cross_val_score(
    pipeline,
    X_train_top10_np,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print(f"\nCV Accuracy (Top 20, best params): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# FINAL TRAINING (TOP 20 ONLY)
# -----------------------------

pipeline.fit(X_train_top10_np, y_train)

# -----------------------------
# FEATURE IMPORTANCE (TOP 20 MODEL)
# -----------------------------

top10_model = pipeline.named_steps["model"]
top10_importances = top10_model.feature_importances_

if top10_importances.sum() > 0:
    top10_importance_pct = (top10_importances / top10_importances.sum()) * 100
else:
    top10_importance_pct = top10_importances

top10_feature_importance_df = pd.DataFrame(
    {
        "feature": top_10_features,
        "importance": top10_importance_pct,
    }
).sort_values(by="importance", ascending=False)

# -----------------------------
# HOLD-OUT EVALUATION
# -----------------------------

y_pred = pipeline.predict(X_test_top10_np)

print(f"\nHold-out Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE ARTIFACTS
# -----------------------------

joblib.dump(pipeline, os.path.join(ARTIFACT_DIR, "model_pipeline_dt.joblib"))
feature_importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance_dt.csv"),
    index=False
)
tree_importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance_dt_tree.csv"),
    index=False
)
top10_feature_importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance_top20_dt.csv"),
    index=False
)

# Save the list of top 10 features
with open(os.path.join(ARTIFACT_DIR, "top_20_features_dt.txt"), "w") as f:
    for feature in top_10_features:
        f.write(feature + "\n")

# Save training statistics for transparency
healthy_mask = y_train == 0
pd_mask = y_train == 1
training_stats = {
    "healthy_stats": X_train_top10[healthy_mask].describe().to_dict(),
    "pd_stats": X_train_top10[pd_mask].describe().to_dict(),
}
joblib.dump(training_stats, os.path.join(ARTIFACT_DIR, "training_stats_dt.joblib"))

print("\n" + "=" * 50)
print("Artifacts saved successfully")
print("  - model_pipeline_dt.joblib")
print("  - feature_importance_dt.csv")
print("  - feature_importance_dt_tree.csv")
print("  - feature_importance_top20_dt.csv")
print("  - top_20_features_dt.txt")
print("  - training_stats_dt.joblib")
print("=" * 50)
