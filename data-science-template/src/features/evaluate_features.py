import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plot aesthetics
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 7)

# --------------------------------------------------------------
# 1. Load Engineered Dataset
# --------------------------------------------------------------
DATA_PATH = "../../data/interim/03_features_engineered.pkl"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Engineered dataset missing at: {DATA_PATH}")

df = pd.read_pickle(DATA_PATH)
print(f"✔ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} features.")

# --------------------------------------------------------------
# 2. Encode Operating Regime Ordinally
# --------------------------------------------------------------
# Map progressive operational power stages to ordinal integers
regime_mapping = {
    "REGIME 0: STOPPED / IDLE": 0,
    "REGIME 1: STARTUP / EMPTY MILL": 1,
    "REGIME 2: FULL PRODUCTION (2300 kW)": 2,
}

if "Operating_Regime" not in df.columns:
    raise KeyError("Column 'Operating_Regime' not found in dataset.")

df["regime_target"] = df["Operating_Regime"].map(regime_mapping)

# Drop any unmapped or missing regime rows
df_eval = df.dropna(subset=["regime_target"]).copy()

# --------------------------------------------------------------
# 3. Filter Numerical Features for Evaluation
# --------------------------------------------------------------
# Exclude raw categorical strings, regime targets, and binary transient masks
ignore_cols = [
    "Operating_Regime",
    "regime_target",
    "operational_state_cluster",
]
feature_candidates = [
    c
    for c in df_eval.select_dtypes(include=[np.number]).columns
    if c not in ignore_cols and not c.endswith("_is_mechanical_transient")
]

# --------------------------------------------------------------
# 4. Compute Linear & Non-Linear Monotonic Correlations
# --------------------------------------------------------------
pearson_corr = (
    df_eval[feature_candidates]
    .apply(lambda col: col.corr(df_eval["regime_target"], method="pearson"))
    .abs()
)

spearman_corr = (
    df_eval[feature_candidates]
    .apply(lambda col: col.corr(df_eval["regime_target"], method="spearman"))
    .abs()
)

# Compile into summary evaluation table
df_corr_summary = pd.DataFrame(
    {
        "Feature": feature_candidates,
        "Abs_Pearson": pearson_corr.values,
        "Abs_Spearman": spearman_corr.values,
    }
).sort_values(by="Abs_Spearman", ascending=False)

print("\n" + "=" * 80)
print("TOP 15 FEATURES RANKED BY CORRELATION TO OPERATING REGIME")
print("=" * 80)
print(df_corr_summary.head(15).to_string(index=False))

# --------------------------------------------------------------
# 5. Plot Top Feature Correlations
# --------------------------------------------------------------
top_n = 15
top_features = df_corr_summary.head(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
y_indices = np.arange(top_n)
bar_height = 0.38

ax.barh(
    y_indices - bar_height / 2,
    top_features["Abs_Spearman"],
    height=bar_height,
    color="#1f77b4",
    label="Spearman Rank (Monotonic)",
)
ax.barh(
    y_indices + bar_height / 2,
    top_features["Abs_Pearson"],
    height=bar_height,
    color="#aec7e8",
    label="Pearson (Linear)",
)

ax.set_yticks(y_indices)
ax.set_yticklabels(top_features["Feature"], fontweight="semibold", fontsize=9)
ax.invert_yaxis()  # Highest correlation on top
ax.set_xlabel("Absolute Correlation Coefficient (|r|)", fontweight="semibold")
ax.set_title(
    "Feature Importance: Alignment with Operating Regimes (0 kW -> 2300 kW)",
    fontweight="bold",
    fontsize=11,
)
ax.set_xlim(0, 1.05)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 6. Cluster vs Regime Cross-Tabulation
# --------------------------------------------------------------
# Evaluates how unsupervised KMeans clusters align with actual production regimes
if "operational_state_cluster" in df_eval.columns:
    print("\n" + "=" * 80)
    print("UNSUPERVISED CLUSTERS VS ACTUAL OPERATING REGIMES (CROSS-TABULATION)")
    print("=" * 80)
    ctab = pd.crosstab(
        df_eval["Operating_Regime"],
        df_eval["operational_state_cluster"],
        margins=True,
    )
    print(ctab)
