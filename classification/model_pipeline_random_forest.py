import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "wallen_2022_primary_cohort", "microbiome_V2.csv")
TARGET_COLUMN = "Case_status_PD"
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "results", "classification")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Drop broad "catch-all" columns that tend to be poorly defined / noisy.
EXCLUDE_SUBSTRINGS = ("ungrouped", "unintegrated", "unclassified", "unknown", "no_name")

# RF can be slow with repeated CV. Keep defaults reasonable; bump if needed.
N_ESTIMATORS_SELECTION = 200
N_ESTIMATORS_FINAL = 400
CV_SPLITS = 5
CV_REPEATS = 1


def drop_by_name_filter(df_features):
    exclude_cols = [
        col for col in df_features.columns
        if any(substr in col.lower() for substr in EXCLUDE_SUBSTRINGS)
    ]
    if exclude_cols:
        print(f"Dropping {len(exclude_cols)} columns by name filter.")
        return df_features.drop(columns=exclude_cols)
    return df_features


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(DATA_PATH)
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {DATA_PATH}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
X = drop_by_name_filter(X)
feature_names = X.columns.tolist()

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# -----------------------------
# IMPUTE MISSING VALUES
# -----------------------------

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_names, index=X_test.index)

# -----------------------------
# STEP 1: FIND TOP 20 FEATURES
# -----------------------------

print("Step 1: Training on all features to find top 20...")

variance_filter = VarianceThreshold(threshold=0.0001)
X_train_filtered = variance_filter.fit_transform(X_train)
X_test_filtered = variance_filter.transform(X_test)

selected_mask = variance_filter.get_support()
selected_features = np.array(feature_names)[selected_mask]

selector_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS_SELECTION,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
selector_model.fit(X_train_filtered, y_train)

feature_importance_df = pd.DataFrame(
    {"feature": selected_features, "importance": selector_model.feature_importances_}
).sort_values(by="importance", ascending=False)

top_10_features = feature_importance_df["feature"].head(20).tolist()

print("\nTop 20 Most Important Features:")
for i, (idx, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.6f}")

# -----------------------------
# STEP 2: TRAIN MODEL ON TOP 20 ONLY
# -----------------------------

print("\n" + "=" * 50)
print("Step 2: Training final model on top 20 features only...")
print("=" * 50)

X_train_top10 = X_train[top_10_features].values
X_test_top10 = X_test[top_10_features].values

pipeline = Pipeline(
    steps=[
        (
            "model",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS_FINAL,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                max_depth=None,
            ),
        ),
    ]
)

cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42)

# Avoid nested parallelism: RF already uses all cores; keep CV single-process.
cv_scores = cross_val_score(
    pipeline,
    X_train_top10,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,
)

print(f"\nCV Accuracy (Top 20): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# FINAL TRAINING + EVAL
# -----------------------------

pipeline.fit(X_train_top10, y_train)
y_pred = pipeline.predict(X_test_top10)

print(f"\nHold-out Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE ARTIFACTS (RF VARIANT)
# -----------------------------

joblib.dump(pipeline, os.path.join(ARTIFACT_DIR, "model_pipeline_rf.joblib"))
feature_importance_df.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance_rf.csv"), index=False)

top10_model = pipeline.named_steps["model"]
top10_importances = top10_model.feature_importances_
top10_total = top10_importances.sum()
top10_pct = (top10_importances / top10_total) * 100 if top10_total else top10_importances
pd.DataFrame({"feature": top_10_features, "importance": top10_pct}).sort_values(
    by="importance", ascending=False
).to_csv(os.path.join(ARTIFACT_DIR, "feature_importance_top20_rf.csv"), index=False)

with open(os.path.join(ARTIFACT_DIR, "top_20_features_rf.txt"), "w") as f:
    for feature in top_10_features:
        f.write(feature + "\n")

healthy_mask = y_train == 0
pd_mask = y_train == 1
X_train_top10_df = X_train[top_10_features]
training_stats = {
    "healthy_stats": X_train_top10_df[healthy_mask].describe().to_dict(),
    "pd_stats": X_train_top10_df[pd_mask].describe().to_dict(),
}
joblib.dump(training_stats, os.path.join(ARTIFACT_DIR, "training_stats_rf.joblib"))

print("\n" + "=" * 50)
print("Artifacts saved successfully")
print("  - model_pipeline_rf.joblib")
print("  - feature_importance_rf.csv")
print("  - feature_importance_top20_rf.csv")
print("  - top_20_features_rf.txt")
print("  - training_stats_rf.joblib")
print("=" * 50)

