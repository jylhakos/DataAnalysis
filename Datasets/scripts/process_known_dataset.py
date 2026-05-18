"""
Known Dataset Processing Example
---------------------------------
Synthetic house price dataset with known, human-readable features:

  - age              : age of the property in years
  - area             : floor area in square meters
  - rooms            : number of rooms
  - distance_to_center: distance to city center in kilometres
  - price            : target variable (sale price in currency units)

Workflow:
  1.  Generate synthetic known dataset
  2.  Explore data (shape, head, describe)
  3.  Handle missing values
  4.  Feature engineering (domain-based transformations)
  5.  Feature selection via Pearson correlation
  6.  Train-test split
  7.  Train Random Forest Regressor
  8.  Evaluate: RMSE and R-squared
  9.  Plot: correlation heatmap, feature importance,
            predicted vs actual, residual plot

Run from the Datasets/ directory:
    source venv/bin/activate
    python scripts/process_known_dataset.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate synthetic known dataset
# ---------------------------------------------------------------------------
np.random.seed(42)
n = 500

age = np.random.randint(1, 50, n)
area = np.random.randint(40, 300, n)
rooms = np.random.randint(1, 8, n)
distance_to_center = np.random.uniform(0.5, 30.0, n)
noise = np.random.normal(0, 10_000, n)

price = (
    area * 3_000
    + rooms * 15_000
    - distance_to_center * 2_000
    - age * 500
    + noise
)

df = pd.DataFrame(
    {
        "age": age,
        "area": area,
        "rooms": rooms,
        "distance_to_center": distance_to_center,
        "price": price,
    }
)

print("=" * 60)
print("  KNOWN DATASET")
print("=" * 60)
print(f"Shape        : {df.shape}")
print(f"\nFirst 5 rows :\n{df.head()}")
print(f"\nStatistics   :\n{df.describe().round(2)}")

# ---------------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------------
df = df.fillna(df.mean(numeric_only=True))

# ---------------------------------------------------------------------------
# 3. Feature engineering (domain knowledge enables these transformations)
# ---------------------------------------------------------------------------
df["price_per_area"] = df["price"] / df["area"]
df["rooms_per_area"] = df["rooms"] / df["area"]

# ---------------------------------------------------------------------------
# 4. Feature selection: inspect Pearson correlation with target
# ---------------------------------------------------------------------------
features = ["age", "area", "rooms", "distance_to_center", "price_per_area", "rooms_per_area"]
target = "price"

corr_with_target = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
print(f"\nPearson correlation with '{target}':\n{corr_with_target.round(4)}")

X = df[features]
y = df[target]

# ---------------------------------------------------------------------------
# 5. Train-test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 6. Train Random Forest Regressor
# ---------------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ---------------------------------------------------------------------------
# 7. Evaluate metrics
# ---------------------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"\nEvaluation Metrics")
print(f"  RMSE : {rmse:,.2f}")
print(f"  R²   : {r2:.4f}")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

# --- Plot 1: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[features + [target]].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Known Dataset: Pearson Correlation Heatmap")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "known_correlation_heatmap.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 2: Feature Importance ---
importances = model.feature_importances_
feat_df = (
    pd.DataFrame({"Feature": features, "Importance": importances})
    .sort_values("Importance", ascending=True)
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_df["Feature"], feat_df["Importance"], color="steelblue")
ax.set_title("Known Dataset: Feature Importance (Random Forest)")
ax.set_xlabel("Importance Score")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "known_feature_importance.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 3: Predicted vs Actual ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, predictions, alpha=0.45, color="steelblue", edgecolors="k", linewidths=0.3)
lo = min(y_test.min(), predictions.min())
hi = max(y_test.max(), predictions.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect Prediction")
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Known Dataset: Predicted vs Actual")
ax.legend()
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "known_predicted_vs_actual.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# --- Plot 4: Residual Plot ---
residuals = y_test - predictions
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(predictions, residuals, alpha=0.45, color="darkorange", edgecolors="k", linewidths=0.3)
ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residual  (Actual - Predicted)")
ax.set_title("Known Dataset: Residual Plot")
fig.tight_layout()
out = os.path.join(PLOTS_DIR, "known_residual_plot.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll known dataset plots saved to: {PLOTS_DIR}/")
