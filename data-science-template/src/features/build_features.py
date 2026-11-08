import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import custom template modules with fallbacks
try:
    from DataTransformation import (
        LowPassFilter,
        PrincipalComponentAnalysis,
    )
    from TemporalAbstraction import NumericalAbstraction

    USE_CUSTOM_MODULES = True
except ImportError:
    USE_CUSTOM_MODULES = False

# --------------------------------------------------------------
# 0. Configuration & Sensor Topology (10-min i-Sense Cadence)
# --------------------------------------------------------------
ACTIVE_SENSORS = [f"{i}RV" for i in range(1, 13)]
CORE_METRICS = [
    "Velocity (mm/s)",
    "Acceleration (g)",
    "VC_G (g)",
    "FC",
    "Temperature (°C)",
]

# Primary vibration channels
VIB_VELOCITY_COLS = [f"{s}_Velocity (mm/s)" for s in ACTIVE_SENSORS]
VIB_ACCEL_COLS = [f"{s}_Acceleration (g)" for s in ACTIVE_SENSORS]
VIB_VCG_COLS = [f"{s}_VC_G (g)" for s in ACTIVE_SENSORS]
TEMP_COLS = [f"{s}_Temperature (°C)" for s in ACTIVE_SENSORS]

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
INPUT_PATH = "../../data/interim/02_outliers_removed.pkl"
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Cleaned dataset missing at: {INPUT_PATH}")

df = pd.read_pickle(INPUT_PATH)
if not isinstance(df.index, pd.DatetimeIndex):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
df = df.sort_index()

print(f"✔ Dataset loaded: {df.shape[0]} timestamps across {df.shape[1]} channels.")

# --------------------------------------------------------------
# 2. Dealing with missing values (imputation)
# --------------------------------------------------------------
# Linear time-interpolation with boundary forward/backward fill to guarantee zero gaps
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate(method="time").bfill().ffill()

# --------------------------------------------------------------
# 3. Calculating set duration (Continuous Regime Run Time)
# --------------------------------------------------------------
# Tracks uninterrupted run duration (in hours) within each operational regime.
# Critical for monitoring thermal expansion and lubrication saturation on 2300 kW drives.
if "Operating_Regime" in df.columns:
    regime_switches = (
        df["Operating_Regime"] != df["Operating_Regime"].shift(1)
    ).cumsum()
    # 1 sample = 10 minutes = 1/6 hour
    df["regime_continuous_runtime_hrs"] = df.groupby(regime_switches).cumcount() * (
        10.0 / 60.0
    )


# --------------------------------------------------------------
# 4. Butterworth lowpass filter (Shift-Cycle Thermal Trend)
# --------------------------------------------------------------
# Sampling interval dt = 600s (fs = 1/600 Hz).
# Cutoff at 12 hours (1/43200 Hz) isolates slow ambient diurnal/thermal trends
# from high-frequency operational shock transitions.
def apply_butterworth_lowpass(
    df_in, cols, cutoff_hours=12.0, sample_interval_min=10.0, order=2
):
    """Zero-phase Butterworth lowpass filter adapted to 10-minute SCADA intervals."""
    df_filtered = df_in.copy()
    fs_hz = 1.0 / (sample_interval_min * 60.0)
    cutoff_hz = 1.0 / (cutoff_hours * 3600.0)
    nyq = 0.5 * fs_hz
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, norm_cutoff, btype="low", analog=False)

    for col in cols:
        if col in df_filtered.columns:
            # filtfilt guarantees zero-phase distortion (no time lag)
            df_filtered[f"{col}_lowpass_trend"] = filtfilt(
                b, a, df_filtered[col].values
            )
    return df_filtered


# Apply lowpass filter to key bearing temperature and drive-end velocity channels
lowpass_targets = [
    col
    for col in [
        "1RV_Velocity (mm/s)",
        "6RV_Velocity (mm/s)",
        "1RV_Temperature (°C)",
        "6RV_Temperature (°C)",
    ]
    if col in df.columns
]
df = apply_butterworth_lowpass(df, lowpass_targets, cutoff_hours=12.0)

# --------------------------------------------------------------
# 5. Principal component analysis (Spatial Vibration Reduction)
# --------------------------------------------------------------
# Compresses spatial collinearity across all 12 bearing housings into orthogonal principal axes
valid_pca_cols = [c for c in VIB_VELOCITY_COLS if c in df.columns]

scaler_pca = StandardScaler()
scaled_vels = scaler_pca.fit_transform(df[valid_pca_cols])

pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(scaled_vels)

df["pca_spatial_vib_dim1"] = pca_transformed[:, 0]
df["pca_spatial_vib_dim2"] = pca_transformed[:, 1]
df["pca_spatial_vib_dim3"] = pca_transformed[:, 2]

print(
    f"✔ Spatial PCA Complete (3 components): {pca.explained_variance_ratio_.sum() * 100:.2f}% variance retained."
)

# --------------------------------------------------------------
# 6. Industrial Multi-Sensor Magnitude Attributes (Energy Proxies)
# --------------------------------------------------------------
# Total Drivetrain Kinetic Energy Proxy: sqrt(sum of velocity^2 across all 12 channels)
active_vels = [c for c in VIB_VELOCITY_COLS if c in df.columns]
df["drivetrain_total_kinetic_energy"] = np.sqrt(
    np.sum(df[active_vels].values ** 2, axis=1)
)

# Total Mill Structural Shock Intensity: sqrt(sum of VC_G^2 across all 12 channels)
active_vcg = [c for c in VIB_VCG_COLS if c in df.columns]
df["mill_total_impact_intensity"] = np.sqrt(np.sum(df[active_vcg].values ** 2, axis=1))

# Twin-Branch Output Load Disparity (Coupled branch output asymmetry between 10RV and 12RV)
if "10RV_Velocity (mm/s)" in df.columns and "12RV_Velocity (mm/s)" in df.columns:
    df["twin_branch_torque_disparity_abs"] = np.abs(
        df["10RV_Velocity (mm/s)"] - df["12RV_Velocity (mm/s)"]
    )

# --------------------------------------------------------------
# 7. Temporal Abstraction (Option A: 30 min, 1 hr, 8 hrs)
# --------------------------------------------------------------
# Windows: w=3 (30 min - shock), w=6 (1 hr - batch), w=48 (8 hr - shift baseline)
TEMPORAL_WINDOWS = {
    3: "30min",
    6: "1hr",
    48: "8hr",
}

key_telemetry_channels = [
    c
    for c in [
        "1RV_Velocity (mm/s)",
        "6RV_Velocity (mm/s)",
        "1RV_Temperature (°C)",
        "10RV_Velocity (mm/s)",
    ]
    if c in df.columns
]

for w, label in TEMPORAL_WINDOWS.items():
    for col in key_telemetry_channels:
        rolling = df[col].rolling(window=w, min_periods=1)
        df[f"{col}_roll_mean_{label}"] = rolling.mean()
        df[f"{col}_roll_std_{label}"] = rolling.std().fillna(0.0)
        df[f"{col}_roll_max_{label}"] = rolling.max()

# Dynamic Shift Divergence Index: Ratio of immediate 30-min surge to 8-hour shift baseline
for col in key_telemetry_channels:
    mean_30m = df[f"{col}_roll_mean_30min"]
    mean_8h = df[f"{col}_roll_mean_8hr"]
    df[f"{col}_shift_divergence_ratio"] = mean_30m / (mean_8h + 1e-6)

# --------------------------------------------------------------
# 8. Rate-of-Rise & Impact Severity Indices (Replacing Sub-Nyquist FFT)
# --------------------------------------------------------------
# Evaluates derivative/growth rates and impact sharpness directly
for s in ACTIVE_SENSORS:
    vel_col = f"{s}_Velocity (mm/s)"
    temp_col = f"{s}_Temperature (°C)"
    acc_col = f"{s}_Acceleration (g)"
    vcg_col = f"{s}_VC_G (g)"

    # 1-Hour Rate of Change (dV/dt, dT/dt over 6 samples)
    if vel_col in df.columns:
        df[f"{s}_velocity_rate_of_change_1hr"] = df[vel_col].diff(periods=6).fillna(0.0)

    if temp_col in df.columns:
        df[f"{s}_temp_rate_of_rise_1hr"] = df[temp_col].diff(periods=6).fillna(0.0)

    # Impact Severity Ratio: Peak Acceleration (VC_G) over RMS Acceleration
    if vcg_col in df.columns and acc_col in df.columns:
        df[f"{s}_impact_severity_index"] = df[vcg_col] / (df[acc_col] + 1e-6)

# --------------------------------------------------------------
# 9. Dealing with Overlapping Windows (Approval 1: Stride = 1)
# --------------------------------------------------------------
# 10-minute cadence yields only ~144 points/day; we preserve 100% of data points (zero decimation).
df_processed = df.dropna().copy()
print(
    f"✔ 100% of telemetry samples retained. Total records preserved: {len(df_processed)}"
)

# --------------------------------------------------------------
# 10. Clustering (Unsupervised Operational Degradation States)
# --------------------------------------------------------------
# Partitions machine operation into 3 distinct operational clusters:
# (e.g., Nominal Smooth Production, Heavy Choking/Surging, Idle/Coast-down)
cluster_features = [
    "drivetrain_total_kinetic_energy",
    "mill_total_impact_intensity",
    "pca_spatial_vib_dim1",
]
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_processed[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_processed["operational_state_cluster"] = kmeans.fit_predict(X_cluster)

# --------------------------------------------------------------
# 11. Export Engineered Dataset
# --------------------------------------------------------------
OUTPUT_DIR = "../../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PKL_EXPORT_PATH = os.path.join(OUTPUT_DIR, "03_features_engineered.pkl")
CSV_EXPORT_PATH = os.path.join(OUTPUT_DIR, "03_features_engineered.csv")

df_processed.to_pickle(PKL_EXPORT_PATH)
df_processed.to_csv(CSV_EXPORT_PATH)

print("\n" + "=" * 80)
print("STAGE 3 FEATURE ENGINEERING SUCCESSFULLY COMPLETED")
print("=" * 80)
print(
    f"Final feature dataset shape: {df_processed.shape[0]} rows x {df_processed.shape[1]} columns"
)
print(f"Exported Pickle: {os.path.abspath(PKL_EXPORT_PATH)}")
print(f"Exported CSV:    {os.path.abspath(CSV_EXPORT_PATH)}")
print("=" * 80)
