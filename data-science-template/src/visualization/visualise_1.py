import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
data_path = "../../data/interim/01_processed.pkl"
df = pd.read_pickle(data_path)

# Uses Motor DE (1RV) acceleration >= 0.10g as the running condition
is_running = df["1RV_Acceleration (g)"] >= 0.1
df_operational = df[is_running].copy()


output_dir = "../../reports/figures/advanced_expert_analysis"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Dynamic Torque-Split Load Sharing Trend (12RV vs. 10RV)
# -------------------------------------------------------------
# Upper Output / Lower Output ratio over time
ratio_vel = df_op["12RV_Velocity (mm/s)"] / (df_op["10RV_Velocity (mm/s)"] + 1e-6)
ratio_peak = df_op["12RV_VC_G (g)"] / (df_op["10RV_VC_G (g)"] + 1e-6)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
fig.patch.set_facecolor("white")

# Velocity load-share ratio
ax1.plot(
    df_op.index,
    ratio_vel,
    color="#1f77b4",
    linewidth=1.3,
    label="Upper/Lower Velocity Ratio (12RV / 10RV)",
)
ax1.axhline(
    1.0, color="black", linestyle="-", linewidth=1.2, label="Ideal 50/50 Balance (1.0)"
)
ax1.axhspan(0.85, 1.15, color="green", alpha=0.15, label="Normal Operating Band (±15%)")
ax1.set_ylabel("Velocity Ratio", fontweight="semibold")
ax1.set_title(
    "Dynamic Torque-Split Load-Sharing Ratio: Upper vs. Lower Output Shafts",
    fontweight="bold",
)
ax1.legend(loc="upper right", fontsize=8.5)
ax1.grid(True, linestyle="--", alpha=0.5)

# Peak shock ratio
ax2.plot(
    df_op.index,
    ratio_peak,
    color="#d62728",
    linewidth=1.3,
    label="Upper/Lower Peak Shock Ratio (12RV / 10RV)",
)
ax2.axhline(1.0, color="black", linestyle="-", linewidth=1.2)
ax2.axhspan(
    0.80, 1.20, color="green", alpha=0.15, label="Normal Shock Balance Band (±20%)"
)
ax2.set_ylabel("Peak Shock Ratio", fontweight="semibold")
ax2.set_xlabel("Acquisition Timestamp", fontweight="semibold")
ax2.legend(loc="upper right", fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{output_dir}/01_dynamic_load_sharing_trend.png", dpi=140)
plt.show()

# -------------------------------------------------------------
# 2. Crest Factor vs. Kurtosis Microcrack Characterization Scatter
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")

target_pins = ["1RV", "2RV", "3RV", "4RV", "10RV", "12RV"]
colors = plt.cm.tab10(np.linspace(0, 1, len(target_pins)))

for i, s in enumerate(target_pins):
    fc = df_op[f"{s}_FC"]
    k_factor = df_op[f"{s}_K Factor"]
    ax.scatter(
        k_factor,
        fc,
        label=DRIVE_TRAIN_SENSOR_LABELS[s],
        color=colors[i],
        alpha=0.5,
        s=30,
    )

# Diagnostic boundaries
ax.axhline(
    5.0,
    color="red",
    linestyle="--",
    linewidth=1.4,
    label="Microcrack Threshold (FC ≥ 5.0)",
)
ax.axvline(
    3.2,
    color="orange",
    linestyle="--",
    linewidth=1.4,
    label="Non-Gaussian Kurtosis (K ≥ 3.2)",
)

ax.axhspan(
    0, 3.5, xmin=0, xmax=0.35, color="green", alpha=0.1, label="Healthy Baseline Zone"
)
ax.text(
    2.6,
    5.5,
    "DISCRETE SURFACE IMPACTS\n(Microcracks / Spalls)",
    color="darkred",
    fontweight="bold",
    fontsize=9.5,
)
ax.text(
    2.6,
    3.2,
    "GAUSSIAN OPERATING BASELINE",
    color="darkgreen",
    fontweight="bold",
    fontsize=9.5,
)

ax.set_title(
    "Impulse Characterization: Crest Factor (FC) vs. Kurtosis (K Factor)",
    fontweight="bold",
)
ax.set_xlabel("Kurtosis (K Factor)", fontweight="semibold")
ax.set_ylabel("Crest Factor (FC)", fontweight="semibold")
ax.set_xlim(2.4, 3.4)
ax.set_ylim(2.0, 7.5)
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{output_dir}/02_fc_vs_kurtosis_scatter.png", dpi=140)
plt.show()

# -------------------------------------------------------------
# 3. Axial Shaft Thermal Gradient (ΔT = T_Inboard - T_Outboard)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("white")

delta_t = {
    "Center Input Pinion\n(3RV - 4RV)": df_op["3RV_Temperature (°C)"]
    - df_op["4RV_Temperature (°C)"],
    "Lower Int-1 Shaft\n(5RV - 6RV)": df_op["5RV_Temperature (°C)"]
    - df_op["6RV_Temperature (°C)"],
    "Upper Int-2 Shaft\n(7RV - 8RV)": df_op["7RV_Temperature (°C)"]
    - df_op["8RV_Temperature (°C)"],
    "Lower Output Shaft\n(9RV - 10RV)": df_op["9RV_Temperature (°C)"]
    - df_op["10RV_Temperature (°C)"],
}

box_data = [d.dropna() for d in delta_t.values()]
bp = ax.boxplot(
    box_data,
    labels=list(delta_t.keys()),
    patch_artist=True,
    boxprops=dict(facecolor="#aec7e8", color="#1f77b4", linewidth=1.2),
    medianprops=dict(color="darkred", linewidth=1.5),
)

ax.axhline(
    0,
    color="black",
    linestyle="-",
    linewidth=1.2,
    label="Symmetrical Thermal Balance (0°C)",
)
ax.axhspan(-3.0, 3.0, color="green", alpha=0.15, label="Normal Expansion Band (±3°C)")
ax.set_ylabel("Temperature Difference ΔT (°C)", fontweight="semibold")
ax.set_title(
    "Axial Shaft Thermal Gradients (Inboard DE vs. Outboard NDE)", fontweight="bold"
)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")

plt.tight_layout()
plt.savefig(f"{output_dir}/03_axial_thermal_gradients.png", dpi=140)
plt.show()

# -------------------------------------------------------------
# 4. Extreme Impact Shock Exceedance Probability (CCDF)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("white")

ccdf_sensors = ["1RV", "2RV", "3RV", "4RV", "10RV"]
colors_ccdf = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]

for i, s in enumerate(ccdf_sensors):
    data = np.sort(df_op[f"{s}_VC_G (g)"].dropna().values)
    ccdf = 1.0 - np.arange(len(data)) / float(len(data))
    ax.step(
        data,
        ccdf,
        where="post",
        label=DRIVE_TRAIN_SENSOR_LABELS[s],
        color=colors_ccdf[i],
        linewidth=1.5,
    )

ax.set_yscale("log")
ax.axvline(
    2.0,
    color="orange",
    linestyle="--",
    alpha=0.7,
    label="Moderate Impact Barrier (2.0g)",
)
ax.axvline(
    8.0, color="red", linestyle="--", alpha=0.7, label="Severe Fatigue Barrier (8.0g)"
)
ax.set_title(
    "Microcrack Tail-Risk: Extreme Shock Exceedance Probability (CCDF)",
    fontweight="bold",
)
ax.set_xlabel("Peak Shock Amplitude VC_G (g)", fontweight="semibold")
ax.set_ylabel("Probability P(Shock > x)", fontweight="semibold")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{output_dir}/04_extreme_shock_exceedance_curve.png", dpi=140)
plt.show()
