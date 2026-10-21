import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
data_path = "../../data/interim/01_processed.pkl"
df = pd.read_pickle(data_path)

# Ensure DatetimeIndex and chronological ordering
if not isinstance(df.index, pd.DatetimeIndex):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
df = df.sort_index()

# Define sensor groups matching the physical layout schematic
MOTOR_SENSORS = ["1RV", "2RV"]
GEARBOX_SENSORS = [f"{i}RV" for i in range(3, 13)]  # 3RV through 12RV
GEARBOX_INBOARD = ["11RV", "7RV", "3RV", "5RV", "9RV"]  # Facing Motor
GEARBOX_OUTBOARD = ["12RV", "8RV", "4RV", "6RV", "10RV"]  # Away from Motor
DRIVE_TRAIN_SENSORS = MOTOR_SENSORS + GEARBOX_SENSORS

print(f"Loaded dataset: {df.shape[0]} timestamps")
print(f"Active Drive Train Sensors ({len(DRIVE_TRAIN_SENSORS)}): {DRIVE_TRAIN_SENSORS}")
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# Motor: Compare 1RV (Non Drive End) vs 2RV (Drive End  Coupled)
V_motor = df[["1RV_Velocity (mm/s)", "2RV_Velocity (mm/s)"]]
gearbox_cols = [
    f"{s}_Velocity (mm/s)"
    for s in GEARBOX_SENSORS
    if f"{s}_Velocity (mm/s)" in df.columns
]
V_gearbox = df[gearbox_cols]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# 1. Motor bearings
ax1.plot(
    V_motor.index,
    V_motor["1RV_Velocity (mm/s)"],
    label="1RV (Motor NDE)",
    color="#aec7e8",
)
ax1.plot(
    V_motor.index,
    V_motor["2RV_Velocity (mm/s)"],
    label="2RV (Motor DE - Coupled)",
    color="#1f77b4",
    linewidth=1.6,
)
ax1.set_ylabel("Vibration Velocity (mm/s)", fontweight="semibold")
ax1.set_title("Motor Bearings Vibration: 1RV (NDE) vs. 2RV (DE)", fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

# 2. All 10 Gearbox bearings (3RV to 12RV)
cmap = plt.get_cmap("tab10", len(GEARBOX_SENSORS))
for i, s in enumerate(GEARBOX_SENSORS):
    col = f"{s}_Velocity (mm/s)"
    if col in V_gearbox.columns:
        ax2.plot(
            V_gearbox.index,
            V_gearbox[col],
            label=f"Gearbox {s}",
            color=cmap(i),
            alpha=0.8,
        )

ax2.set_ylabel("Vibration Velocity (mm/s)", fontweight="semibold")
ax2.set_xlabel("Acquisition Timestamp", fontweight="semibold")
ax2.set_title("Gearbox Bearings Vibration (3RV to 12RV)", fontweight="bold")
ax2.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2, fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
# --------------------------------------------------------------
# Plot all exercises (Subsystems: Motor vs. Gearbox Inboard vs. Outboard)
# --------------------------------------------------------------
motor_mean = df[[f"{s}_Velocity (mm/s)" for s in MOTOR_SENSORS]].mean(axis=1)
inboard_mean = df[
    [
        f"{s}_Velocity (mm/s)"
        for s in GEARBOX_INBOARD
        if f"{s}_Velocity (mm/s)" in df.columns
    ]
].mean(axis=1)
outboard_mean = df[
    [
        f"{s}_Velocity (mm/s)"
        for s in GEARBOX_OUTBOARD
        if f"{s}_Velocity (mm/s)" in df.columns
    ]
].mean(axis=1)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    df.index,
    motor_mean,
    label="Drive Motor (1RV, 2RV Mean)",
    color="#1f77b4",
    linewidth=1.6,
)
ax.plot(
    df.index,
    inboard_mean,
    label="Gearbox Inboard / Motor Side (3,5,7,9,11RV)",
    color="#ff7f0e",
    linewidth=1.4,
)
ax.plot(
    df.index,
    outboard_mean,
    label="Gearbox Outboard (4,6,8,10,12RV)",
    color="#2ca02c",
    linewidth=1.4,
)

ax.set_title(
    "Machine Subsystems: Motor vs. Gearbox Inboard vs. Outboard",
    fontsize=13,
    fontweight="bold",
)
ax.set_ylabel("RMS Velocity (mm/s)", fontsize=11, fontweight="semibold")
ax.set_xlabel("Acquisition Timestamp", fontsize=11, fontweight="semibold")
ax.legend(loc="upper right", frameon=True)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
mpl.rcParams["figure.figsize"] = (14, 6)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["lines.linewidth"] = 1.4
mpl.rcParams["axes.titlesize"] = 13
mpl.rcParams["axes.labelsize"] = 10.5
mpl.rcParams["xtick.labelsize"] = 9.5
mpl.rcParams["ytick.labelsize"] = 9.5
mpl.rcParams["legend.fontsize"] = 9.5
mpl.rcParams["boxplot.flierprops.markersize"] = 4

# --------------------------------------------------------------
# Compare medium vs. heavy sets (Operational Regimes)
# --------------------------------------------------------------
# 1. Clean operational threshold: Active Milling(>0.05 mm/s) vs. Idle/Stoppage
df["operational_regime"] = np.where(
    df["2RV_Velocity (mm/s)"] > 0.05, "Active Milling", "Idle / Stoppage"
)

fig, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(13, 5))
regimes = ["Active Milling", "Idle / Stoppage"]

motor_regime_data = [
    df[df["operational_regime"] == r]["2RV_Velocity (mm/s)"].dropna() for r in regimes
]
gearbox_regime_data = [
    df[df["operational_regime"] == r]["3RV_Velocity (mm/s)"].dropna() for r in regimes
]

# Color styling for boxes: Green for Active, Red for Stoppage
regime_colors = ["#2ca02c", "#d62728"]

# Motor DE (2RV) Boxplot
bp1 = ax_b1.boxplot(
    motor_regime_data,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, color in zip(bp1["boxes"], regime_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Explicitly set the x-tick positions and text labels
ax_b1.set_xticks([1, 2], labels=regimes, fontweight="semibold")
ax_b1.set_title("Motor DE (2RV) Velocity by Regime", fontweight="bold")
ax_b1.set_ylabel("Velocity (mm/s)", fontweight="semibold")
ax_b1.grid(True, linestyle="--", alpha=0.5)

# Gearbox Input DE (3RV) Boxplot
bp2 = ax_b2.boxplot(
    gearbox_regime_data,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, color in zip(bp2["boxes"], regime_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Explicitly set the x-tick positions and text labels
ax_b2.set_xticks([1, 2], labels=regimes, fontweight="semibold")
ax_b2.set_title("Gearbox Input DE (3RV) Velocity by Regime", fontweight="bold")
ax_b2.set_ylabel("Velocity (mm/s)", fontweight="semibold")
ax_b2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "Operational Regime Contrast: Active Grinding vs. Idle/Outage",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------
# Compare participants (Coupling Interface: Motor DE 2RV vs. Gearbox Input 3RV)
# --------------------------------------------------------------
# The schematic confirms 2RV and 3RV share the high-speed flexible coupling shaft
fig, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Direct Time Series Overlay
ax_p1.plot(
    df.index,
    df["2RV_Velocity (mm/s)"],
    label="Motor DE (2RV)",
    color="#1f77b4",
    alpha=0.85,
)
ax_p1.plot(
    df.index,
    df["3RV_Velocity (mm/s)"],
    label="Gearbox Input DE (3RV)",
    color="#ff7f0e",
    alpha=0.85,
)
ax_p1.set_title(
    "Coupled Shaft: Motor DE (2RV) vs. Gearbox Input (3RV)", fontweight="bold"
)
ax_p1.set_ylabel("Velocity (mm/s)")
ax_p1.set_xlabel("Acquisition Timestamp")
ax_p1.legend()
ax_p1.grid(True, linestyle="--", alpha=0.5)

# 2. Cross-Coupling Transmission Scatter
ax_p2.scatter(
    df["2RV_Velocity (mm/s)"],
    df["3RV_Velocity (mm/s)"],
    color="#2ca02c",
    alpha=0.5,
    edgecolors="none",
)
ax_p2.set_title("Coupling Transmission Correlation (2RV vs. 3RV)", fontweight="bold")
ax_p2.set_xlabel("Motor DE Velocity (mm/s)")
ax_p2.set_ylabel("Gearbox Input DE Velocity (mm/s)")
ax_p2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "Drive Train Interface: Motor-to-Gearbox High-Speed Coupling",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------
# Plot multiple axis (Physical Telemetry Suite per Bearing)
# --------------------------------------------------------------
# 6-channel comparative grid (3 rows x 2 columns):
# Row 1: Velocity (Motor DE vs. Gearbox Outboard)
# Row 2: Acceleration (Motor DE vs. Gearbox Outboard)
# Row 3: Temperature (Motor DE vs. Gearbox Outboard)

SENSOR_MOTOR = "2RV"  # Motor Drive-End (Coupled)
OUTBOARD_SENSOR = "4RV"  # Gearbox Input Shaft Outboard (or 6RV, 8RV, 10RV, 12RV)

fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)

# Colors matching each physical measurement
param_config = [
    ("Velocity (mm/s)", "#1f77b4", "mm/s"),
    ("Acceleration (g)", "#ff7f0e", "g"),
    ("Temperature (°C)", "#d62728", "°C"),
]

for row_idx, (feat, color, unit) in enumerate(param_config):
    # Left Column: Motor DE (2RV)
    col_motor = f"{SENSOR_MOTOR}_{feat}"
    ax_left = axes[row_idx, 0]
    if col_motor in df.columns:
        ax_left.plot(
            df.index,
            df[col_motor],
            color=color,
            linewidth=1.3,
            label=f"{SENSOR_MOTOR} {feat}",
        )
    ax_left.set_ylabel(feat, fontweight="semibold")
    ax_left.set_title(
        f"{SENSOR_MOTOR} (Motor DE) - {feat}", fontsize=11, fontweight="bold"
    )
    ax_left.grid(True, linestyle="--", alpha=0.5)
    ax_left.legend(loc="upper right")

    # Right Column: Gearbox Outboard (4RV)
    col_outboard = f"{OUTBOARD_SENSOR}_{feat}"
    ax_right = axes[row_idx, 1]
    if col_outboard in df.columns:
        ax_right.plot(
            df.index,
            df[col_outboard],
            color=color,
            linewidth=1.3,
            label=f"{OUTBOARD_SENSOR} {feat}",
        )
    ax_right.set_ylabel(feat, fontweight="semibold")
    ax_right.set_title(
        f"{OUTBOARD_SENSOR} (Gearbox Outboard) - {feat}", fontsize=11, fontweight="bold"
    )
    ax_right.grid(True, linestyle="--", alpha=0.5)
    ax_right.legend(loc="upper right")

# X-axis label on bottom plots
axes[-1, 0].set_xlabel("Acquisition Timestamp", fontweight="semibold")
axes[-1, 1].set_xlabel("Acquisition Timestamp", fontweight="semibold")

plt.suptitle(
    f"Multi-Axis Drive Train Telemetry: Motor DE ({SENSOR_MOTOR}) vs. Gearbox Outboard ({OUTBOARD_SENSOR})",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------
# Separate Full History vs. Operational Only
# --------------------------------------------------------------
# Set 1: Full dataset including outages/stoppages (428 timestamps)
df_full = df.copy()

# Set 2: Operational state only (clears shutdown / idle periods)
# Uses Motor DE (1RV) acceleration >= 0.10g as the running condition
is_running = df_full["1RV_Acceleration (g)"] >= 0.1
df_operational = df_full[is_running].copy()

print(f"Total timestamps: {len(df_full)}")
print(
    f"Operational milling timestamps: {len(df_operational)} (Stoppages cleared: {(~is_running).sum()})"
)


# 3. Standardized Suite Plotting Function
def plot_telemetry_suite(
    df_input, sensors, features, suite_label, export_dir=None, show=True
):
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    for sensor in sensors:
        fig, axes = plt.subplots(
            len(features), 1, figsize=(14, 2.3 * len(features)), sharex=True
        )
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Telemetry Suite ({suite_label}): Sensor {sensor}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )

        for ax, feat in zip(axes, features):
            col = f"{sensor}_{feat}"
            if col in df_input.columns:
                color = (
                    "#1f77b4"
                    if "Velocity" in feat
                    else (
                        "#ff7f0e"
                        if "Acceleration" in feat
                        else ("#d62728" if "Temperature" in feat else "#9467bd")
                    )
                )
                ax.plot(df_input.index, df_input[col], color=color, linewidth=1.2)
                ax.set_ylabel(feat, fontsize=9.5, fontweight="semibold")
                ax.grid(True, linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("Acquisition Timestamp", fontsize=10, fontweight="semibold")
        plt.tight_layout()

        if export_dir:
            out_file = os.path.join(export_dir, f"{sensor}_suite.png")
            plt.savefig(out_file, dpi=130)
            print(f"   -> Saved: {out_file}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# 4. Generate Both Sets

target_features = [
    "Velocity (mm/s)",
    "Acceleration (g)",
    "Temperature (°C)",
    "K Factor",
]
active_sensors = [f"{i}RV" for i in range(1, 13)]  # 1RV through 12RV

# Execute Set 1: Full History (with outages)
print("\n[Set 1] Generating Full Historical Telemetry...")
plot_telemetry_suite(
    df_input=df_full,
    sensors=active_sensors,
    features=target_features,
    suite_label="Full History - All States",
    export_dir="../../reports/figures/suite_with_stoppages",
    show=False,
)

# Execute Set 2: Operational Only (stoppages cleared)
print("\n[Set 2] Generating Operational Telemetry (Stoppages Cleared)...")
plot_telemetry_suite(
    df_input=df_operational,
    sensors=active_sensors,
    features=target_features,
    suite_label="Operational Only - Stoppages Cleared",
    export_dir="../../reports/figures/suite_operational_only",
    show=False,
)

print("\n✔ Both sets created successfully.")


# --------------------------------------------------------------
# Combine plots in one figure (Integrated Operational Dashboard)
# --------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
fig.patch.set_facecolor("white")

# 1. Top-Left: Motor Vibration (1RV NDE vs. 2RV DE)
axes[0, 0].plot(
    df_operational.index,
    df_operational["1RV_Velocity (mm/s)"],
    label="1RV (Motor NDE)",
    color="#aec7e8",
    linewidth=1.3,
)
axes[0, 0].plot(
    df_operational.index,
    df_operational["2RV_Velocity (mm/s)"],
    label="2RV (Motor DE)",
    color="#1f77b4",
    linewidth=1.5,
)
axes[0, 0].set_title("Motor Vibration (mm/s) [Operational]", fontweight="bold")
axes[0, 0].set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
axes[0, 0].legend(loc="upper right")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. Top-Right: Gearbox Vibration Across Key Stages (3RV, 4RV, 5RV, 7RV)
gearbox_dashboard_sensors = [
    "3RV",
    "4RV",
    "5RV",
    "7RV",
    "8RV9RV",
    "10RV",
    "11RV",
    "12RV",
]
cmap = plt.get_cmap("tab10", len(gearbox_dashboard_sensors))
for i, s in enumerate(gearbox_dashboard_sensors):
    col = f"{s}_Velocity (mm/s)"
    if col in df_operational.columns:
        axes[0, 1].plot(
            df_operational.index,
            df_operational[col],
            label=f"Gearbox {s}",
            color=cmap(i),
            linewidth=1.2,
            alpha=0.85,
        )
axes[0, 1].set_title("Gearbox Vibration (mm/s) [Operational]", fontweight="bold")
axes[0, 1].set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
axes[0, 1].legend(loc="upper right")
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. Bottom-Left: Thermal Dissipation Across the High-Speed Coupling (2RV vs. 3RV vs. 4RV)
axes[1, 0].plot(
    df_operational.index,
    df_operational["2RV_Temperature (°C)"],
    label="Motor DE (2RV)",
    color="#d62728",
    linewidth=1.4,
)
axes[1, 0].plot(
    df_operational.index,
    df_operational["3RV_Temperature (°C)"],
    label="Gearbox Input (3RV)",
    color="#ff9896",
    linewidth=1.4,
)
if "4RV_Temperature (°C)" in df_operational.columns:
    axes[1, 0].plot(
        df_operational.index,
        df_operational["4RV_Temperature (°C)"],
        label="Gearbox Outboard (4RV)",
        color="#7f7f7f",
        linewidth=1.3,
    )
if "12RV_Temperature (°C)" in df_operational.columns:
    axes[1, 0].plot(
        df_operational.index,
        df_operational["12RV_Temperature (°C)"],
        label="Gearbox Outboard (12RV)",
        color="#5c349d",
        linewidth=1.3,
    )
axes[1, 0].set_title(
    "Drive Train Thermal Profile (°C) [Operational]", fontweight="bold"
)
axes[1, 0].set_ylabel("Temperature (°C)", fontweight="semibold")
axes[1, 0].set_xlabel("Acquisition Timestamp", fontweight="semibold")
axes[1, 0].legend(loc="upper right")
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# 4. Bottom-Right: Impulsive Impact Severity (K-Factor)
if "2RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["2RV_K Factor"],
        label="Motor DE (2RV)",
        color="#9467bd",
        linewidth=1.3,
    )
if "3RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["3RV_K Factor"],
        label="Gearbox Input (3RV)",
        color="#8c564b",
        linewidth=1.3,
        alpha=0.85,
    )
if "4RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["4RV_K Factor"],
        label="Gearbox Outboard (4RV)",
        color="#1497bA",
        linewidth=1.3,
    )
if "5RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["5RV_K Factor"],
        label="Gearbox Outboard (5RV)",
        color="#4467bf",
        linewidth=1.3,
    )
if "8RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["8RV_K Factor"],
        label="Gearbox Outboard (8RV)",
        color="#d62728",
        linewidth=1.3,
    )
if "10RV_K Factor" in df_operational.columns:
    axes[1, 1].plot(
        df_operational.index,
        df_operational["10RV_K Factor"],
        label="Gearbox Outboard (10RV)",
        color="#8467dA",
        linewidth=1.3,
    )
axes[1, 1].set_title("Impact Kurtosis (K Factor) [Operational]", fontweight="bold")
axes[1, 1].set_ylabel("K Factor", fontweight="semibold")
axes[1, 1].set_xlabel("Acquisition Timestamp", fontweight="semibold")
axes[1, 1].legend(loc="upper right")
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "Integrated Condition Dashboard: Active Milling Operations (Stoppages Cleared)",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
export_dir = "../../reports/figures/operational_sensor_analysis"
os.makedirs(export_dir, exist_ok=True)

# Select target units: Motor bearings (1RV, 2RV) and Input Gearbox bearings (3RV, 4RV)
target_units = [
    "1RV",
    "2RV",
    "3RV",
    "4RV",
    "5RV",
    "7RV",
    "8RV9RV",
    "10RV",
    "11RV",
    "12RV",
]
features = [
    "Velocity (mm/s)",
    "Acceleration (g)",
    "Temperature (°C)",
    "K Factor",
    "FC",
    "VC_G (g)",
    "VCC_(g)",
]

print(
    f"\n[Exporting] Generating operational diagnostic figures into '{export_dir}/'..."
)

for sensor in target_units:
    for feat in features:
        col = f"{sensor}_{feat}"
        if col not in df_operational.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor("white")

        color = (
            "#1f77b4"
            if "Velocity" in feat
            else (
                "#ff7f0e"
                if "Acceleration" in feat
                else ("#d62728" if "Temperature" in feat else "#9467bd")
            )
        )
        ax.plot(df_operational.index, df_operational[col], color=color, linewidth=1.3)
        ax.set_title(
            f"Operational Diagnostic: Sensor {sensor} - {feat} (Active Milling)",
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Acquisition Timestamp", fontweight="semibold")
        ax.set_ylabel(feat, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.5)

        safe_feat = re.sub(r"[^\w\-_\. ]", "_", feat).replace(" ", "_")
        filename = f"{sensor}_{safe_feat}_operational.png"
        out_path = os.path.join(export_dir, filename)

        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"   -> Exported: {out_path}")

print("✔ Operational figures exported successfully.")


# --------------------------------------------------------------
# Upgraded Telemetry Suite (Including Peak Impacts & Crest Factor)
# --------------------------------------------------------------
# Expanded feature suite covering macro severity, thermal, and micro-impacts
features = [
    "Velocity (mm/s)",  # Structural / Rotational severity (ISO 10816)
    "Acceleration (g)",  # High-frequency RMS energy
    "VC_G (g)",  # Peak instantaneous shock impact
    "FC",  # Crest Factor (Peak / RMS)
    "Temperature (°C)",  # Thermal condition
]

target_sensors = ["1RV", "2RV", "3RV", "4RV"]

for sensor in target_sensors:
    fig, axes = plt.subplots(
        len(features), 1, figsize=(14, 2.2 * len(features)), sharex=True
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Advanced Health Suite: Sensor {sensor} (Operational)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for ax, feat in zip(axes, features):
        col = f"{sensor}_{feat}"
        if col in df_operational.columns:
            color = (
                "#1f77b4"
                if "Velocity" in feat
                else (
                    "#ff7f0e"
                    if feat == "Acceleration (g)"
                    else (
                        "#9467bd"
                        if "VC_G" in feat
                        else ("#d62728" if "Temperature" in feat else "#8c564b")
                    )
                )
            )
            ax.plot(
                df_operational.index, df_operational[col], color=color, linewidth=1.2
            )
            ax.set_ylabel(feat, fontsize=9, fontweight="semibold")
            ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Acquisition Timestamp", fontsize=10, fontweight="semibold")
    plt.tight_layout()
    plt.show()


# =============================================================================================================
output_dir = "../../reports/figures/expert_analysis"
os.makedirs(output_dir, exist_ok=True)
df_op = df_operational.copy()
# -------------------------------------------------------------
# 1. Master Naming Dictionary (Sensors 1 to 12)
# -------------------------------------------------------------
DRIVE_TRAIN_SENSOR_LABELS = {
    # --- Drive Motor ---
    "1RV": "Motor NDE (1RV)",
    "2RV": "Motor DE - Main Coupling (2RV)",
    # --- Primary Torque-Splitter: Center Input Shaft ---
    "3RV": "Splitter Input Inboard DE (3RV)",
    "4RV": "Splitter Input Outboard NDE (4RV)",
    # --- Primary Torque-Splitter: Lower Branch ---
    "5RV": "Splitter Lower Int-1 Inboard (5RV)",
    "6RV": "Splitter Lower Int-1 Outboard (6RV)",
    "9RV": "Splitter Lower Output Inboard (9RV)",
    "10RV": "Splitter Lower Output Outboard (10RV)",  # Drives Bottom Spacer Shaft
    # --- Primary Torque-Splitter: Upper Branch ---
    "7RV": "Splitter Upper Int-2 Inboard (7RV)",
    "8RV": "Splitter Upper Int-2 Outboard (8RV)",
    "11RV": "Splitter Upper Output Inboard (11RV)",
    "12RV": "Splitter Upper Output Outboard (12RV)",  # Drives Top Spacer Shaft
}

ACTIVE_SENSORS = list(DRIVE_TRAIN_SENSOR_LABELS.keys())


# -------------------------------------------------------------
# 3. Chart 1: Asset Risk Quadrant Matrix (12 Sensors)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("white")

# 12 distinct colors for the full drive train
colors = plt.cm.tab20(np.linspace(0, 1, len(ACTIVE_SENSORS)))

for idx, s in enumerate(ACTIVE_SENSORS):
    vel_col = f"{s}_Velocity (mm/s)"
    fc_col = f"{s}_FC"
    if vel_col not in df_op.columns or fc_col not in df_op.columns:
        continue

    vel = df_op[vel_col]
    fc = df_op[fc_col]
    ax.scatter(vel, fc, color=colors[idx], alpha=0.25, s=20)

    v_mean, f_mean = vel.mean(), fc.mean()
    ax.scatter(v_mean, f_mean, color=colors[idx], edgecolor="black", s=160, zorder=5)
    ax.annotate(
        s,
        (v_mean, f_mean),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="white", ec=colors[idx], lw=1.5, alpha=0.9
        ),
    )

# Threshold Lines (ISO 10816 & Crest Factor)
ax.axvline(
    1.8,
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="ISO Good/Acceptable Limit (1.8 mm/s)",
)
ax.axvline(
    2.8,
    color="red",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
    label="ISO Alarm Limit (2.8 mm/s)",
)
ax.axhline(
    5.0,
    color="crimson",
    linestyle="-.",
    linewidth=1.8,
    alpha=0.9,
    label="Incipient Microcrack Threshold (FC = 5.0)",
)
ax.axhline(
    3.5,
    color="green",
    linestyle=":",
    linewidth=1.4,
    alpha=0.8,
    label="Healthy Continuous Rolling (FC = 3.5)",
)

# Quadrant Text Overlays
ax.text(
    0.3,
    8.8,
    "QUADRANT II: INCIPIENT SURFACE FATIGUE\n(Microcracks / Spalling - Macro Velocity Still Low)",
    fontsize=9.5,
    fontweight="bold",
    color="crimson",
    bbox=dict(boxstyle="square", fc="#ffe6e6", ec="none", alpha=0.75),
)
ax.text(
    0.3,
    2.2,
    "QUADRANT I: HEALTHY OPERATIONAL STATE\n(Low Structural Vibration, Smooth Continuous Rolling)",
    fontsize=9.5,
    fontweight="bold",
    color="darkgreen",
    bbox=dict(boxstyle="square", fc="#e6ffe6", ec="none", alpha=0.75),
)
ax.text(
    3.1,
    8.8,
    "QUADRANT IV: CRITICAL ADVANCED FAILURE\n(Severe Structural Looseness + Raceway Breakdown)",
    fontsize=9.5,
    fontweight="bold",
    color="darkred",
    bbox=dict(boxstyle="square", fc="#ffd9cc", ec="none", alpha=0.75),
)
ax.text(
    3.1,
    2.2,
    "QUADRANT III: STRUCTURAL / MISALIGNMENT\n(High Velocity Energy, but No High-Freq Microcracks)",
    fontsize=9.5,
    fontweight="bold",
    color="#805500",
    bbox=dict(boxstyle="square", fc="#fff5e6", ec="none", alpha=0.75),
)

ax.set_title(
    "Asset Condition Quadrant Matrix: Structural Severity vs. Surface Microcrack Impulses (12 Positions)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel(
    "RMS Vibration Velocity [mm/s] (ISO 10816 Macro Severity)",
    fontsize=11,
    fontweight="semibold",
)
ax.set_ylabel(
    "Crest Factor [FC] (Impulsive Shock Severity / Microcracking)",
    fontsize=11,
    fontweight="semibold",
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(0, 5.8)
ax.set_ylim(1.8, 10.5)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_risk_quadrant_matrix.png"), dpi=140)
plt.show()

# -------------------------------------------------------------
# 4. Chart 2: Drive Train Telemetry Heatmap (12 Bearings)
# -------------------------------------------------------------
heatmap_rows = []
for s in ACTIVE_SENSORS:
    label = DRIVE_TRAIN_SENSOR_LABELS[s]
    heatmap_rows.append(
        {
            "Component": label,
            "Velocity (mm/s)": df_op[f"{s}_Velocity (mm/s)"].mean(),
            "Acceleration (g)": df_op[f"{s}_Acceleration (g)"].mean(),
            "Peak Shock VC_G (g)": df_op[f"{s}_VC_G (g)"].mean(),
            "Pk-to-Pk VCC_G (g)": df_op[f"{s}_VCC_G (g)"].mean(),
            "Crest Factor (FC)": df_op[f"{s}_FC"].mean(),
            "Kurtosis (K Factor)": df_op[f"{s}_K Factor"].mean(),
            "Temperature (°C)": df_op[f"{s}_Temperature (°C)"].mean(),
            "Microcrack Alert Rate (%)": (df_op[f"{s}_FC"] >= 5.0).mean() * 100,
        }
    )

df_heat = pd.DataFrame(heatmap_rows).set_index("Component")
df_norm = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min())

fig, ax = plt.subplots(figsize=(14, 8.5))
fig.patch.set_facecolor("white")
sns.heatmap(
    df_norm,
    annot=df_heat.values,
    fmt=".2f",
    cmap="YlOrRd",
    cbar_kws={"label": "Relative Stress Severity (0 = Plant Min, 1 = Plant Max)"},
    linewidths=1.2,
    linecolor="#e0e0e0",
    ax=ax,
)
ax.set_title(
    "Drive Train Shaft-by-Shaft Telemetry Matrix (12 Monitored Positions)",
    fontsize=13,
    fontweight="bold",
    pad=14,
)
ax.set_xlabel("Monitored Telemetry Channel", fontsize=11, fontweight="semibold")
ax.set_ylabel("Drive Train Component Position", fontsize=11, fontweight="semibold")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_drive_train_heatmap.png"), dpi=140)
plt.show()

# -------------------------------------------------------------
# 5. Chart 3: Actionable Vulnerability Ranking (12 Bearings)
# -------------------------------------------------------------
fig, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(16, 7.5))
fig.patch.set_facecolor("white")

# 1. Microcrack Alert Rate (%)
df_sorted_alert = df_heat.sort_values(by="Microcrack Alert Rate (%)", ascending=True)
colors1 = [
    "#d62728" if x > 10 else ("#ff7f0e" if x > 1 else "#2ca02c")
    for x in df_sorted_alert["Microcrack Alert Rate (%)"]
]
bars1 = ax_bar1.barh(
    df_sorted_alert.index,
    df_sorted_alert["Microcrack Alert Rate (%)"],
    color=colors1,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.85,
)
for bar in bars1:
    width = bar.get_width()
    ax_bar1.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

ax_bar1.set_title(
    "1. Impulsive Anomaly Exposure\n(% Operational Hours in Microcrack Zone: FC ≥ 5.0)",
    fontsize=11,
    fontweight="bold",
)
ax_bar1.set_xlabel("% of Active Grinding Hours", fontweight="semibold")
ax_bar1.axvline(
    5.0, color="red", linestyle="--", alpha=0.7, label="Critical Alert Boundary (5%)"
)
ax_bar1.grid(True, linestyle="--", alpha=0.5, axis="x")
ax_bar1.legend(loc="lower right", fontsize=9)
ax_bar1.set_xlim(0, max(20, df_sorted_alert["Microcrack Alert Rate (%)"].max() + 5))

# 2. Peak Impact Shock (VC_G in g)
df_sorted_peak = df_heat.sort_values(by="Peak Shock VC_G (g)", ascending=True)
colors2 = [
    "#d62728" if x > 5 else ("#ff7f0e" if x > 1 else "#2ca02c")
    for x in df_sorted_peak["Peak Shock VC_G (g)"]
]
bars2 = ax_bar2.barh(
    df_sorted_peak.index,
    df_sorted_peak["Peak Shock VC_G (g)"],
    color=colors2,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.85,
)
for bar in bars2:
    width = bar.get_width()
    ax_bar2.text(
        width + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}g",
        va="center",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

ax_bar2.set_title(
    "2. Maximum Instantaneous Impact Energy\n(Mean Peak Shock Amplitude: VC_G)",
    fontsize=11,
    fontweight="bold",
)
ax_bar2.set_xlabel("Peak Acceleration (g)", fontweight="semibold")
ax_bar2.axvline(
    2.0,
    color="orange",
    linestyle="--",
    alpha=0.7,
    label="Moderate Impact Boundary (2.0g)",
)
ax_bar2.grid(True, linestyle="--", alpha=0.5, axis="x")
ax_bar2.legend(loc="lower right", fontsize=9)
ax_bar2.set_xlim(0, max(14, df_sorted_peak["Peak Shock VC_G (g)"].max() + 2))

plt.suptitle(
    "Maintenance Action Priority: 12-Bearing Defect Susceptibility Ranking",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_vulnerability_ranking.png"), dpi=140)
plt.show()

# -------------------------------------------------------------
# 6. Chart 4: Thermal-Mechanical Operating Envelopes
# -------------------------------------------------------------
fig, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("white")

# 1. Motor DE (2RV)
sc1 = ax_t1.scatter(
    df_op["2RV_Temperature (°C)"],
    df_op["2RV_Velocity (mm/s)"],
    c=df_op["2RV_FC"],
    cmap="plasma",
    s=35,
    alpha=0.75,
)
cb1 = plt.colorbar(sc1, ax=ax_t1)
cb1.set_label("Crest Factor (FC - Impulsiveness)", fontweight="semibold")
ax_t1.set_title(
    "1. Motor DE - Main Coupling (2RV): Hydrodynamic vs. Mechanical Distress",
    fontsize=10.5,
    fontweight="bold",
)
ax_t1.set_xlabel("Bearing Temperature (°C)", fontweight="semibold")
ax_t1.set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
ax_t1.axhline(
    2.3,
    color="red",
    linestyle="--",
    alpha=0.6,
    label="Elevated Vibration Alert (2.3 mm/s)",
)
ax_t1.axvline(
    46.0,
    color="darkorange",
    linestyle="--",
    alpha=0.6,
    label="Thermal Saturation Limit (46°C)",
)
ax_t1.legend(loc="upper left", fontsize=8.5)
ax_t1.grid(True, linestyle="--", alpha=0.5)

# 2. Splitter Input Inboard (3RV)
sc2 = ax_t2.scatter(
    df_op["3RV_Temperature (°C)"],
    df_op["3RV_Acceleration (g)"],
    c=df_op["3RV_FC"],
    cmap="plasma",
    s=35,
    alpha=0.75,
)
cb2 = plt.colorbar(sc2, ax=ax_t2)
cb2.set_label("Crest Factor (FC - Impulsiveness)", fontweight="semibold")
ax_t2.set_title(
    "2. Splitter Input Inboard DE (3RV): Pinion Mesh Thermal-Vibration Correlation",
    fontsize=10.5,
    fontweight="bold",
)
ax_t2.set_xlabel("Bearing Temperature (°C)", fontweight="semibold")
ax_t2.set_ylabel("RMS Acceleration (g)", fontweight="semibold")
ax_t2.axvline(
    48.0,
    color="darkorange",
    linestyle="--",
    alpha=0.6,
    label="Thermal Saturation Limit (48°C)",
)
ax_t2.legend(loc="upper left", fontsize=8.5)
ax_t2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "Cross-Domain Diagnostic: Thermal-Mechanical Operating Envelopes",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_thermal_mechanical_envelope.png"), dpi=140)
plt.show()

print("✔ Expert diagnostic visualization suite updated for sensors 1 to 12.")
