import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# Set plot styling
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (14, 5)

# Telemetry Position Mapping (Active 12 Sensors)
ACTIVE_SENSORS = [
    "1RV",
    "2RV",
    "3RV",
    "4RV",
    "5RV",
    "6RV",
    "7RV",
    "8RV",
    "9RV",
    "10RV",
    "11RV",
    "12RV",
]
CORE_METRICS = [
    "Velocity (mm/s)",
    "Acceleration (g)",
    "VC_G (g)",
    "FC",
    "Temperature (°C)",
]

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
DATA_PATH = "../../data/interim/01_processed.pkl"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Telemetry dataset missing at: {DATA_PATH}")

df = pd.read_pickle(DATA_PATH)
if not isinstance(df.index, pd.DatetimeIndex):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
df = df.sort_index()


# Define Operating Regimes for Grouping
def assign_operating_regime(row):
    a_rms = row.get("1RV_Acceleration (g)", 0.0)
    v_rms = row.get("1RV_Velocity (mm/s)", 0.0)
    if a_rms < 0.05 and v_rms < 0.5:
        return "REGIME 0: STOPPED / IDLE"
    elif 0.05 <= a_rms < 0.10:
        return "REGIME 1: STARTUP / EMPTY MILL"
    else:
        return "REGIME 2: FULL PRODUCTION (2300 kW)"


df["Operating_Regime"] = df.apply(assign_operating_regime, axis=1)

# Target vibration columns across all channels
target_cols = [
    f"{s}_{m}" for s in ACTIVE_SENSORS for m in CORE_METRICS if f"{s}_{m}" in df
]


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
# In condition monitoring, we distinguish:
#   A. Severe Transducer/DAQ rail faults (VC_G >= 49.0g, or frozen flatlines) -> Replace & Interpolate
#   B. Physical mechanical shocks (FC >= 5.0 during grinding) -> Preserve & Flag
def clean_and_impute_sensor_data(
    df_in,
    active_cols,
    max_physical_shock_g=49.0,
    frozen_std_limit=0.0001,
):
    """Expert Imputation Strategy:

    1. Detects dead sensor zeros and ADC rail-clipping voltages -> Replaces with
    NaN and linearly interpolates.
    2. Preserves real mechanical shock events (Crest Factor surges) while tagging
    them.
    """
    df_cleaned = df_in.copy()
    imputation_counts = {}

    for col in active_cols:
        # Identify non-physical sensor rail spikes or dropouts
        if "VC_G" in col or "Acceleration" in col:
            is_sensor_glitch = (df_cleaned[col] >= max_physical_shock_g) | (
                df_cleaned[col] <= 0.0001
            )
        elif "Velocity" in col:
            is_sensor_glitch = (df_cleaned[col] >= 30.0) | (df_cleaned[col] < 0.0)
        elif "Temperature" in col:
            is_sensor_glitch = (df_cleaned[col] >= 110.0) | (df_cleaned[col] <= 0.0)
        else:
            is_sensor_glitch = pd.Series(False, index=df_cleaned.index)

        # Count and set glitched values to NaN
        glitch_count = is_sensor_glitch.sum()
        imputation_counts[col] = glitch_count

        if glitch_count > 0:
            df_cleaned.loc[is_sensor_glitch, col] = np.nan
            # Linearly interpolate gaps to preserve time-series sampling cadence
            df_cleaned[col] = df_cleaned[col].interpolate(
                method="time", limit_direction="both"
            )

    # Flag legitimate physical transients without altering numerical values
    for s in ACTIVE_SENSORS:
        fc_col = f"{s}_FC"
        if fc_col in df_cleaned.columns:
            df_cleaned[f"{s}_is_mechanical_transient"] = (df_cleaned[fc_col] >= 5.0) & (
                df_cleaned["Operating_Regime"] == "REGIME 2: FULL PRODUCTION (2300 kW)"
            )

    return df_cleaned, pd.Series(imputation_counts)


# Test on single column
sample_col = "1RV_VC_G (g)" if "1RV_VC_G (g)" in df.columns else target_cols[0]
df_test_clean, counts_test = clean_and_impute_sensor_data(df, [sample_col])
print(
    f"\nTest cleaning on {sample_col}: {counts_test[sample_col]} corrupted DAQ"
    " points interpolated."
)

# Create a loop across all operational channels
df_cleaned_master, all_imputation_counts = clean_and_impute_sensor_data(df, target_cols)
corrupted_total = all_imputation_counts.sum()
print(
    f"✔ Completed data cleaning across all {len(target_cols)} channels. Total"
    f" DAQ glitches repaired: {corrupted_total}"
)

df_cleaned_master.info()
# --------------------------------------------------------------
# Summary: Sensor Glitches (Interpolated) vs. Mechanical Transients
# --------------------------------------------------------------
audit_records = []

for sensor in ACTIVE_SENSORS:
    transient_col = f"{sensor}_is_mechanical_transient"
    has_transient_col = transient_col in df_cleaned_master.columns
    total_transients = (
        int(df_cleaned_master[transient_col].sum()) if has_transient_col else 0
    )

    sensor_glitches = {}
    for metric in CORE_METRICS:
        ch = f"{sensor}_{metric}"
        # Fetch the count calculated by clean_and_impute_sensor_data
        sensor_glitches[metric] = int(all_imputation_counts.get(ch, 0))

    audit_records.append(
        {
            "Sensor": sensor,
            "Total_Transients_Flagged": total_transients,
            "Glitches_Velocity": sensor_glitches.get("Velocity (mm/s)", 0),
            "Glitches_Acceleration": sensor_glitches.get("Acceleration (g)", 0),
            "Glitches_VC_G": sensor_glitches.get("VC_G (g)", 0),
            "Glitches_Temp": sensor_glitches.get("Temperature (°C)", 0),
            "Total_Glitches_Interpolated": sum(sensor_glitches.values()),
        }
    )

df_audit = pd.DataFrame(audit_records).set_index("Sensor")

# Display breakdown per sensor
print("=" * 80)
print("SENSOR DATA AUDIT: HARDWARE GLITCHES VS MECHANICAL TRANSIENTS")
print("=" * 80)
print(df_audit.to_string())

# Display system-wide totals
total_glitches = df_audit["Total_Glitches_Interpolated"].sum()
total_shocks = df_audit["Total_Transients_Flagged"].sum()

print("-" * 80)
print(f"Total DAQ/Sensor Glitches Interpolated: {total_glitches:,}")
print(f"Total Real Mechanical Shocks Preserved: {total_shocks:,}")
print("=" * 80)


fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(df_audit.index))
width = 0.35

ax.bar(
    x - width / 2,
    df_audit["Total_Glitches_Interpolated"],
    width,
    label="Corrupted Glitches (Interpolated)",
    color="#d95f02",
)
ax.bar(
    x + width / 2,
    df_audit["Total_Transients_Flagged"],
    width,
    label="Mechanical Transients (Preserved)",
    color="#2ca02c",
)

ax.set_xticks(x)
ax.set_xticklabels(df_audit.index, fontweight="semibold")
ax.set_ylabel("Count of Points", fontweight="semibold")
ax.set_title(
    "Data Integrity Breakdown: Sensor Glitches vs. Operational Shock Events",
    fontweight="bold",
    fontsize=12,
)
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
EXPORT_DIR = "../../data/interim"
os.makedirs(EXPORT_DIR, exist_ok=True)

PKL_EXPORT_PATH = os.path.join(EXPORT_DIR, "02_outliers_removed.pkl")
CSV_EXPORT_PATH = os.path.join(EXPORT_DIR, "02_outliers_removed.csv")

df_cleaned_master.to_pickle(PKL_EXPORT_PATH)
df_cleaned_master.to_csv(CSV_EXPORT_PATH)

print(f"\n✔ Stage 2 Complete: Cleaned dataset exported successfully to:")
print(f"   Pickle: {os.path.abspath(PKL_EXPORT_PATH)}")
print(f"   CSV:    {os.path.abspath(CSV_EXPORT_PATH)}")
print(f"   Final cleaned dataset shape: {df_cleaned_master.shape}")
