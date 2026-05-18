"""
Unknown Dataset Processing Example
-------------------------------------
Synthetic anonymized telemetry dataset with no human-readable feature names:

  - COL_001 to COL_020 : anonymous numeric telemetry signals
  - COL_021            : constant column (low-variance, should be removed)
  - COL_CAT_A          : anonymous categorical variable (TYPE_1/2/3)
  - COL_CAT_B          : anonymous categorical variable (GRP_A/GRP_B)
  - COL_134            : target variable (continuous, unknown meaning)

Workflow:
  1.  Generate synthetic unknown dataset with missing values
  2.  Explore: shape, dtypes, null counts
  3.  Identify numerical and categorical feature types
  4.  Handle missing values (median imputation for numeric columns)
  5.  Encode categorical variables with LabelEncoder
  6.  Variance Threshold filter (remove near-constant features)
  7.  Pearson correlation analysis
  8.  Train-test split
  9.  Train Random Forest Regressor
  10. Evaluate: RMSE and R-squared
  11. Plot: correlation heatmap, feature importance,
             predicted vs actual, residual plot

Run from the Datasets/ directory:
    source venv/bin/activate
    python scripts/process_unknown_dataset.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate synthetic unknown dataset
# ---------------------------------------------------------------------------
np.random.seed(7)
n = 600

# Anonymous numeric columns
cols = {f"COL_{i:03d}": np.random.randn(n) for i in range(1, 21)}

# Constant column — should be eliminated by Variance Threshold
cols["COL_021"] = np.ones(n)

# Anonymous categorical columns
cols["COL_CAT_A"] = np.random.choice(["TYPE_1", "TYPE_2", "TYPE_3"], n)
cols["COL_CAT_B"] = np.random.choice(["GRP_A", "GRP_B"], n)

df = pd.DataFrame(cols)

# Introduce realistic missing values in three columns
for col in ["COL_003", "COL_007", "COL_015"]:
    missing_idx = df.sample(frac=0.05, random_state=1).index
    df.loc[missing_idx, col] = np.nan

# Target: COL_134 — linear combination of a few columns plus noise
df["COL_134"] = (
    2.5 * df["COL_001"]
    + 1.8 * df["COL_005"]
    - 1.2 * df["COL_010"]
    + 0.7 * df["COL_012"]
    + np.random.normal(0, 0.5, n)
)

# ---------------------------------------------------------------------------
# 2. Explore dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("  UNKNOWN DATASET")
print("=" * 60)
print(f"Shape            : {df.shape}")
print(f"\nDtype counts     :\n{df.dtypes.value_counts()}")

missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
print(f"\nMissing values   :\n{missing_summary}")

# ---------------------------------------------------------------------------
# 3. Identify feature types
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
print(f"\nNumeric columns  : {len(numeric_cols)}")
print(f"Categorical cols : {cat_cols}")

# ---------------------------------------------------------------------------
# 4. Handle missing values — median imputation for numeric columns
# ---------------------------------------------------------------------------
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------------
# 5. Encode categorical variables
# ---------------------------------------------------------------------------
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------------------------------------------------------
# 6. Variance Threshold filter — remove near-constant / zero-variance features
# ---------------------------------------------------------------------------
target = "COL_134"
feature_cols = [c for c in df.columns if c != target]

selector = VarianceThreshold(threshold=0.01)
selector.fit(df[feature_cols])
support_mask = selector.get_support()

selected_features = [f for f, keep in zip(feature_cols, support_mask) if keep]
removed_features = [f for f, keep in zip(feature_cols, support_mask) if not keep]

print(f"\nRemoved low-variance features : {removed_features}")
print(f"Features after VarianceThreshold: {len(selected_features)}")

X = df[selected_features]
y = df[target]

# ---------------------------------------------------------------------------
# 7. Pearson correlation of remaining features with target
# ---------------------------------------------------------------------------
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print(f"\nTop 5 features by absolute Pearson correlation with target:\n{corr_with_target.head()}")

# ---------------------------------------------------------------------------
# 8. Train-test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 9. Train Random Forest Regressor
#    (XGBoost / LightGBM / CatBoost can replace this for production use)
# ---------------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ---------------------------------------------------------------------------
# 10. Evaluate metrics
# ---------------------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"\nEvaluation Metrics")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------

# --- Plot 1: Correlation Heatmap (top 12 features by importance + target) ---
importances = model.feature_importances_
feat_importance_series = pd.Series(importances, index=selected_features).sort_values(
    ascending=False
)
top_features = feat_importance_series.head(12).index.tolist()
corr_cols = top_features + [target]

fig, ax = plt.subplots(figsize=(13, 11))
corr_matrix = df[corr_cols].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.4,
    ax=ax,
)
ax.set_title("Unknown Dataset: Pearson Correlation Heatmap (Top Features + Target)")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "unknown_correlation_heatmap.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 2: Feature Importance (top 15) ---
top15 = feat_importance_series.head(15).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top15.index, top15.values, color="teal")
ax.set_title("Unknown Dataset: Feature Importance (Random Forest, Top 15)")
ax.set_xlabel("Importance Score")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "unknown_feature_importance.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 3: Predicted vs Actual ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, predictions, alpha=0.45, color="teal", edgecolors="k", linewidths=0.3)
lo = min(y_test.min(), predictions.min())
hi = max(y_test.max(), predictions.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect Prediction")
ax.set_xlabel("Actual (COL_134)")
ax.set_ylabel("Predicted (COL_134)")
ax.set_title("Unknown Dataset: Predicted vs Actual")
ax.legend()
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "unknown_predicted_vs_actual.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 4: Residual Plot ---
residuals = y_test - predictions
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(predictions, residuals, alpha=0.45, color="purple", edgecolors="k", linewidths=0.3)
ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
ax.set_xlabel("Predicted (COL_134)")
ax.set_ylabel("Residual  (Actual - Predicted)")
ax.set_title("Unknown Dataset: Residual Plot")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "unknown_residual_plot.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll unknown dataset plots saved to: {PLOTS_DIR}/")
