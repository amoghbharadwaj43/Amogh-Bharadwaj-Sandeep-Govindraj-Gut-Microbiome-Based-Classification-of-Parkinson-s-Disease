import os
import warnings
import ctypes
import joblib
import numpy as np
import pandas as pd
import shap
import optuna

# Ensure local OpenMP runtime is discoverable for xgboost on macOS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_OMP_DIR = os.path.join(PROJECT_ROOT, "libomp_macos_arm64")
LOCAL_OMP_LIB = os.path.join(LOCAL_OMP_DIR, "libomp.dylib")
if os.path.exists(LOCAL_OMP_LIB):
    # Preload libomp so dyld can resolve xgboost's @rpath/libomp.dylib
    try:
        ctypes.CDLL(LOCAL_OMP_LIB)
    except OSError:
        pass

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "wallen_2022_primary_cohort", "Source_Data_24Oct2022_prepped.csv")
TARGET_COLUMN = "Case_status=PD"
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "results", "classification")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(DATA_PATH)
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {DATA_PATH}")

X = df.drop(columns=[TARGET_COLUMN])
y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
if y.isna().any():
    raise ValueError(f"Target column '{TARGET_COLUMN}' contains non-numeric values.")
y = y.astype(int)

EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")
exclude_cols = [
    col for col in X.columns
    if any(substr in col.lower() for substr in EXCLUDE_SUBSTRINGS)
]
if exclude_cols:
    print(f"Dropping {len(exclude_cols)} columns by name filter.")
    X = X.drop(columns=exclude_cols)

# Drop any text/object columns (e.g., sample IDs) before fitting.
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Dropping {len(non_numeric_cols)} non-numeric columns (e.g., IDs/text).")
    X = X.drop(columns=non_numeric_cols)

# Coerce to numeric and impute residual missing values with column medians.
X = X.apply(pd.to_numeric, errors="coerce")
all_nan_cols = [col for col in X.columns if X[col].isna().all()]
if all_nan_cols:
    print(f"Dropping {len(all_nan_cols)} columns that are entirely NaN after coercion.")
    X = X.drop(columns=all_nan_cols)
X = X.fillna(X.median(numeric_only=True))

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
# STEP 1: FIND TOP 20 FEATURES
# -----------------------------

print("Step 1: Training on all features to find top 20...")

# Variance filter first
variance_filter = VarianceThreshold(threshold=0.0001)
X_train_filtered = variance_filter.fit_transform(X_train)
X_test_filtered = variance_filter.transform(X_test)

selected_mask = variance_filter.get_support()
selected_features = np.array(feature_names)[selected_mask]

class_counts = y_train.value_counts().to_dict()
neg_count = class_counts.get(0, 1)
pos_count = class_counts.get(1, 1)
scale_pos_weight = neg_count / pos_count if pos_count else 1.0

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    min_child_weight=1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

base_model = XGBClassifier(**XGB_PARAMS)

base_model.fit(X_train_filtered, y_train)

# SHAP-based feature importance
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_train_filtered)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame(
    {
        "feature": selected_features,
        "importance": mean_abs_shap,
    }
).sort_values(by="importance", ascending=False)

def select_top_features_with_nonzero_shap(
    ordered_features, X_train_df, y_train_series, n_features=20, threshold=1e-6
):
    selected = ordered_features[:n_features]
    cursor = n_features
    max_rounds = 5
    last_importance = None

    for _ in range(max_rounds):
        X_sel = X_train_df[selected]
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_sel.values, y_train_series)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sel.values)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        last_importance = mean_abs

        zero_mask = mean_abs <= threshold
        if not np.any(zero_mask):
            return selected, mean_abs

        # Replace zero-importance features with next best from ordered list
        kept = [f for f, imp in zip(selected, mean_abs) if imp > threshold]
        while len(kept) < n_features and cursor < len(ordered_features):
            candidate = ordered_features[cursor]
            cursor += 1
            if candidate not in kept:
                kept.append(candidate)

        selected = kept
        if len(selected) < n_features:
            break

    return selected, last_importance


ordered_features = feature_importance_df["feature"].tolist()
top_10_features, selection_importance = select_top_features_with_nonzero_shap(
    ordered_features, X_train, y_train
)

selection_importance_df = pd.DataFrame(
    {
        "feature": top_10_features,
        "importance": selection_importance
        if selection_importance is not None
        else np.zeros(len(top_10_features)),
    }
).sort_values(by="importance", ascending=False)

print("\nTop 20 Most Important Features:")
for i, (idx, row) in enumerate(selection_importance_df.iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.6f}")

# -----------------------------
# STEP 2: TRAIN MODEL ON TOP 20 ONLY
# -----------------------------

print("\n" + "="*50)
print("Step 2: Training final model on top 20 features only...")
print("="*50)

# Select only top 10 features
X_train_top10 = X_train[top_10_features]
X_test_top10 = X_test[top_10_features]
X_train_top10_np = X_train_top10.values
X_test_top10_np = X_test_top10.values

# -----------------------------
# CROSS-VALIDATION (TOP 20)
# -----------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# -----------------------------
# OPTUNA HYPERPARAMETER TUNING
# -----------------------------

print("\nRunning Optuna hyperparameter tuning...")

OPTUNA_TRIALS = 30

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    model = XGBClassifier(**params)
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
XGB_PARAMS.update(best_params)
print(f"Best CV Accuracy from Optuna: {study.best_value:.4f}")
print(f"Best Params: {best_params}")

# Create new pipeline for top 10 features
pipeline = Pipeline(
    steps=[
        (
            "model",
            XGBClassifier(**XGB_PARAMS),
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

print(f"\nCV Accuracy (Top 20): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# FINAL TRAINING (TOP 20 ONLY)
# -----------------------------

pipeline.fit(X_train_top10_np, y_train)

# -----------------------------
# FEATURE IMPORTANCE (TOP 20 MODEL)
# -----------------------------

top10_model = pipeline.named_steps["model"]
top10_explainer = shap.TreeExplainer(top10_model)
top10_shap_values = top10_explainer.shap_values(X_train_top10_np)
if isinstance(top10_shap_values, list):
    top10_shap_values = top10_shap_values[1] if len(top10_shap_values) > 1 else top10_shap_values[0]
top10_mean_abs_shap = np.abs(top10_shap_values).mean(axis=0)
top10_total = top10_mean_abs_shap.sum()
top10_importance_pct = (
    (top10_mean_abs_shap / top10_total) * 100 if top10_total else top10_mean_abs_shap
)
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

joblib.dump(pipeline, os.path.join(ARTIFACT_DIR, "model_pipeline.joblib"))
feature_importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance.csv"),
    index=False
)
top10_feature_importance_df.to_csv(
    os.path.join(ARTIFACT_DIR, "feature_importance_top20.csv"),
    index=False
)

# Save SHAP values summary for auditability
shap_summary = pd.DataFrame(
    {
        "feature": selected_features,
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values(by="mean_abs_shap", ascending=False)
shap_summary.to_csv(
    os.path.join(ARTIFACT_DIR, "shap_feature_importance.csv"),
    index=False
)

# Save the list of top 20 features
with open(os.path.join(ARTIFACT_DIR, "top_20_features.txt"), "w") as f:
    for feature in top_10_features:
        f.write(feature + "\n")

# CRITICAL: Save training statistics so app.py uses the same reference values
X_train_top10 = X_train[top_10_features]
healthy_mask = y_train == 0
pd_mask = y_train == 1

training_stats = {
    'healthy_stats': X_train_top10[healthy_mask].describe().to_dict(),
    'pd_stats': X_train_top10[pd_mask].describe().to_dict(),
}

joblib.dump(training_stats, os.path.join(ARTIFACT_DIR, "training_stats.joblib"))

print("\n" + "="*50)
print("Artifacts saved successfully")
print("  - model_pipeline.joblib")
print("  - feature_importance.csv")
print("  - feature_importance_top20.csv")
print("  - shap_feature_importance.csv")
print("  - top_20_features.txt")
print("  - training_stats.joblib")
print("="*50)
