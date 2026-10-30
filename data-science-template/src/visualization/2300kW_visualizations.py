import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 2300 kW HEAVY INDUSTRIAL SPECIFICATIONS & ISO GROUP 1 THRESHOLDS
# ==============================================================================
# ISO 20816-3 Group 1 (Large Industrial Machines > 300 kW, Rigid Foundation)
ISO_GROUP1_ACCEPTABLE = 2.3  # mm/s (Zone A/B Boundary)
ISO_GROUP1_ALERT = 4.5  # mm/s (Zone B/C Boundary)
ISO_GROUP1_TRIP = 7.1  # mm/s (Zone C/D Danger/Trip Boundary)

# Thermal Operating Envelopes
TEMP_WARNING = 65.0  # °C (Lubricant degradation / inspection advisory)
TEMP_TRIP = 75.0  # °C (Mandatory OEM Trip limit)

# Purely Mechanical Torque-Split Bounds (AGMA 6014-B15)
TORQUE_SPLIT_IDEAL = 1.00  # Perfect 50/50 balance
TORQUE_SPLIT_TOL_LOWER = 0.85  # -15% Mechanical lower limit
TORQUE_SPLIT_TOL_UPPER = 1.15  # +15% Mechanical upper limit

# Telemetry Position Mapping (RENK Reducer #2092421 & 2.3 MW Motor)
DRIVE_TRAIN_SENSOR_LABELS = {
    # 2.3 MW Electric Drive Motor
    "1RV": "Motor NDE (1RV)",
    "2RV": "Motor DE - Main Coupling (2RV)",
    # RENK Splitter: Center High-Speed Input Shaft
    "3RV": "Splitter Input Inboard DE (3RV)",
    "4RV": "Splitter Input Outboard NDE (4RV)",
    # RENK Splitter: Lower Reduction Line
    "5RV": "Splitter Lower Int-1 Inboard (5RV)",
    "6RV": "Splitter Lower Int-1 Outboard (6RV)",
    "9RV": "Splitter Lower Output Inboard (9RV)",
    "10RV": "Splitter Lower Output Outboard DE (10RV)",  # Drives Bottom Spacer Shaft
    # RENK Splitter: Upper Reduction Line
    "7RV": "Splitter Upper Int-2 Inboard (7RV)",
    "8RV": "Splitter Upper Int-2 Outboard (8RV)",
    "11RV": "Splitter Upper Output Inboard (11RV)",
    "12RV": "Splitter Upper Output Outboard DE (12RV)",  # Drives Top Spacer Shaft
}

ACTIVE_SENSORS = list(DRIVE_TRAIN_SENSOR_LABELS.keys())

KINEMATIC_GROUPS = {
    "Drive Motor (2300 kW)": (["1RV", "2RV"], "#1f77b4", "s"),
    "Splitter Input Shaft": (["3RV", "4RV"], "#2ca02c", "D"),
    "Upper Branch (To Top Pinion)": (["7RV", "8RV", "11RV", "12RV"], "#d62728", "^"),
    "Lower Branch (To Bottom Pinion)": (["5RV", "6RV", "9RV", "10RV"], "#ff7f0e", "o"),
}

SHAFT_ALIGNED_PAIRS = {
    "Center Input Shaft": ("3RV", "4RV"),
    "Upper Int-2 Shaft": ("7RV", "8RV"),
    "Lower Int-1 Shaft": ("5RV", "6RV"),
    "Lower Output Shaft": ("9RV", "10RV"),
    "Upper Output Shaft": ("11RV", "12RV"),
}

TWIN_BRANCH_PAIRS = [
    ("7RV", "5RV", "Intermediate Inboard"),
    ("8RV", "6RV", "Intermediate Outboard"),
    ("11RV", "9RV", "Output Inboard"),
    ("12RV", "10RV", "Output Outboard (Spacer Drive)"),
]

# Matplotlib Styling
mpl.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["lines.linewidth"] = 1.3
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 9.5
mpl.rcParams["legend.fontsize"] = 8.5


# ==============================================================================
# 2. DATA INGESTION & HEAVY MOTOR RUN STATE FILTERING
# ==============================================================================
def load_and_preprocess_telemetry(data_path="../../data/interim/01_processed.pkl"):
    """
    Standardizes timestamps, resamples to 10 minutes, and isolates active
    2300 kW grinding by removing mill shutdowns (a_RMS < 0.10g).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source telemetry missing at: {data_path}")

    df = pd.read_pickle(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()

    df_resampled = df.resample(rule="10min").mean().dropna(how="all")

    # 2.3 MW running threshold: Motor NDE RMS acceleration >= 0.10g
    is_operational = df_resampled["1RV_Acceleration (g)"] >= 0.10
    df_operational = df_resampled[is_operational].copy()

    print(
        f"✔ 2300 kW Mill Telemetry Ingested: {len(df_operational)} active grinding records."
    )
    return df_resampled, df_operational


# ==============================================================================
# 3. MODULE 1: ASSET RISK QUADRANT MATRIX (ISO GROUP 1 RIGID)
# ==============================================================================
def plot_risk_quadrant_matrix(df_op, export_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    sensor_centroids = {}
    for group_name, (sensors, color, marker) in KINEMATIC_GROUPS.items():
        first = True
        for s in sensors:
            vel_col, fc_col = f"{s}_Velocity (mm/s)", f"{s}_FC"
            if vel_col in df_op.columns and fc_col in df_op.columns:
                v_data, f_data = df_op[vel_col], df_op[fc_col]
                v_mean, f_mean = v_data.mean(), f_data.mean()
                sensor_centroids[s] = (v_mean, f_mean)

                ax.scatter(v_data, f_data, color=color, alpha=0.15, s=15, label=None)
                ax.scatter(
                    v_mean,
                    f_mean,
                    color=color,
                    edgecolor="black",
                    s=160,
                    marker=marker,
                    label=group_name if first else None,
                    zorder=5,
                )
                ax.annotate(
                    s,
                    (v_mean, f_mean),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9.5,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec=color,
                        lw=1.3,
                        alpha=0.9,
                    ),
                )
                first = False

    # Symmetry vectors across twin shafts
    for s_up, s_low, _ in TWIN_BRANCH_PAIRS:
        if s_up in sensor_centroids and s_low in sensor_centroids:
            p1, p2 = sensor_centroids[s_up], sensor_centroids[s_low]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="#555555",
                linestyle=":",
                linewidth=1.5,
                zorder=3,
            )

    # ISO 20816-3 Group 1 Rigid Severity Lines
    ax.axvline(
        ISO_GROUP1_ACCEPTABLE,
        color="green",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=f"ISO Group 1 Acceptable ({ISO_GROUP1_ACCEPTABLE} mm/s)",
    )
    ax.axvline(
        ISO_GROUP1_ALERT,
        color="darkorange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"ISO Group 1 Alert ({ISO_GROUP1_ALERT} mm/s)",
    )
    ax.axvline(
        ISO_GROUP1_TRIP,
        color="red",
        linestyle="-",
        linewidth=1.6,
        alpha=0.85,
        label=f"ISO Group 1 Danger/Trip ({ISO_GROUP1_TRIP} mm/s)",
    )

    # Impact Crest Factor Boundaries
    ax.axhline(
        5.0,
        color="crimson",
        linestyle="-.",
        linewidth=1.6,
        alpha=0.9,
        label="Incipient Microcrack Strike (FC = 5.0)",
    )
    ax.axhline(
        3.5,
        color="green",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8,
        label="Healthy Continuous Mesh (FC = 3.5)",
    )

    ax.text(
        0.3,
        9.4,
        "QUADRANT II: INCIPIENT SURFACE FATIGUE\n(Micro-Pitting / Tooth Cracks - ISO Velocity Normal)",
        fontsize=9,
        fontweight="bold",
        color="crimson",
        bbox=dict(boxstyle="square", fc="#ffe6e6", ec="none", alpha=0.75),
    )
    ax.text(
        0.3,
        2.2,
        "QUADRANT I: HEALTHY CONTINUOUS RUNNING\n(Low Structural Severity, Low High-Freq Shock)",
        fontsize=9,
        fontweight="bold",
        color="darkgreen",
        bbox=dict(boxstyle="square", fc="#e6ffe6", ec="none", alpha=0.75),
    )
    ax.text(
        4.8,
        9.4,
        "QUADRANT IV: CRITICAL ADVANCED BREAKDOWN\n(High ISO Velocity + Active Tooth Spalling)",
        fontsize=9,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="square", fc="#ffd9cc", ec="none", alpha=0.75),
    )
    ax.text(
        4.8,
        2.2,
        "QUADRANT III: MACRO-STRUCTURAL DEFECT\n(Rigid Footing Looseness / Coupling Misalignment)",
        fontsize=9,
        fontweight="bold",
        color="#805500",
        bbox=dict(boxstyle="square", fc="#fff5e6", ec="none", alpha=0.75),
    )

    ax.set_title(
        "Asset Quadrant Matrix (2300 kW Rigid Support): ISO 20816-3 Severity vs. Impulsive Impacts",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "RMS Vibration Velocity [mm/s] (ISO Group 1 Macro Severity)",
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Crest Factor [FC] (Peak / RMS Impact Concentration)", fontweight="semibold"
    )
    ax.set_xlim(0, 8.5)
    ax.set_ylim(1.8, 10.5)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "01_risk_quadrant_matrix_expert.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 4. MODULE 2: BIFURCATED HEATMAP & MECHANICAL ASYMMETRY DELTA
# ==============================================================================
def plot_bifurcated_heatmap(df_op, export_dir):
    kinematic_order = [
        ("1RV", "1. Motor NDE (1RV)"),
        ("2RV", "1. Motor DE [Coupled] (2RV)"),
        ("3RV", "2. Splitter Input Inboard DE (3RV)"),
        ("4RV", "2. Splitter Input Outboard NDE (4RV)"),
        ("7RV", "3. Upper Branch: Int-2 Inboard (7RV)"),
        ("8RV", "3. Upper Branch: Int-2 Outboard (8RV)"),
        ("11RV", "3. Upper Branch: Output Inboard (11RV)"),
        ("12RV", "3. Upper Branch: Output Outboard DE (12RV)"),
        ("5RV", "4. Lower Branch: Int-1 Inboard (5RV)"),
        ("6RV", "4. Lower Branch: Int-1 Outboard (6RV)"),
        ("9RV", "4. Lower Branch: Output Inboard (9RV)"),
        ("10RV", "4. Lower Branch: Output Outboard DE (10RV)"),
    ]

    heatmap_rows = []
    for s, label in kinematic_order:
        heatmap_rows.append(
            {
                "Component": label,
                "Velocity (mm/s)": df_op[f"{s}_Velocity (mm/s)"].mean(),
                "Acceleration (g)": df_op[f"{s}_Acceleration (g)"].mean(),
                "Peak Shock VC_G (g)": df_op[f"{s}_VC_G (g)"].mean(),
                "Pk-to-Pk VCC_G (g)": df_op[f"{s}_VCC_G (g)"].mean(),
                "Crest Factor (FC)": df_op[f"{s}_FC"].mean(),
                "Temperature (°C)": df_op[f"{s}_Temperature (°C)"].mean(),
                "Crack Alert Rate (%)": (df_op[f"{s}_FC"] >= 5.0).mean() * 100,
            }
        )

    df_heat = pd.DataFrame(heatmap_rows).set_index("Component")

    upper_labels = [
        label for s, label in kinematic_order if s in ["7RV", "8RV", "11RV", "12RV"]
    ]
    lower_labels = [
        label for s, label in kinematic_order if s in ["5RV", "6RV", "9RV", "10RV"]
    ]
    branch_delta = df_heat.loc[upper_labels].mean() - df_heat.loc[lower_labels].mean()
    branch_delta.name = "=== MECHANICAL DELTA (Upper - Lower) ==="

    df_heat_with_delta = pd.concat([df_heat, pd.DataFrame([branch_delta])])
    df_norm = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min())
    df_norm.loc["=== MECHANICAL DELTA (Upper - Lower) ==="] = np.nan

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        df_norm,
        annot=df_heat_with_delta.values,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Stress Intensity (0 = Min, 1 = Max)"},
        linewidths=1.2,
        linecolor="#e0e0e0",
        mask=df_norm.isna(),
        ax=ax,
    )

    ax.text(
        0.5,
        len(df_heat_with_delta) - 0.5,
        f"Pure Mechanical Load Differential: ΔVel={branch_delta['Velocity (mm/s)']:.2f} mm/s | ΔPeak={branch_delta['Peak Shock VC_G (g)']:.2f}g | ΔTemp={branch_delta['Temperature (°C)']:.1f}°C [Trip Limit: {TEMP_TRIP}°C]",
        fontsize=9.5,
        fontweight="bold",
        color="darkblue",
        va="center",
    )

    ax.set_title(
        "RENK #2092421 Kinematic Stress Heatmap: 2300 kW Pure Mechanical Split",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "02_drive_train_heatmap_expert.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 5. MODULE 3: COMPOSITE MHI & PURE MECHANICAL TORQUE-SPLIT GAUGE
# ==============================================================================
def plot_maintenance_priority_and_balance(df_op, export_dir):
    fig, (ax_rank, ax_gauge) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.2, 0.8]}
    )
    fig.patch.set_facecolor("white")

    mhi_scores = {}
    for s in ACTIVE_SENSORS:
        # Normalized against ISO Group 1 Danger Limit (7.1 mm/s) and OEM Trip Temp (75°C)
        norm_v = df_op[f"{s}_Velocity (mm/s)"].mean() / ISO_GROUP1_TRIP  # 7.1 mm/s
        norm_fc = df_op[f"{s}_FC"].mean() / 6.0  # Multi-crack spalling
        norm_peak = df_op[f"{s}_VC_G (g)"].mean() / 15.0  # Peak impact limit
        norm_temp = max(
            0.0, (df_op[f"{s}_Temperature (°C)"].mean() - 35.0) / (TEMP_TRIP - 35.0)
        )
        alert_rate = (df_op[f"{s}_FC"] >= 5.0).mean()

        # Weighted MHI Formulation
        if (
            0.30 * norm_v >= 1
            or 0.25 * norm_fc >= 1
            or 0.20 * norm_peak >= 1
            or 0.15 * norm_temp >= 1
            or 0.10 * alert_rate >= 1
        ):
            composite_mhi = 100
        else:
            composite_mhi = (
                min(1.0, 0.30 * norm_v)
                + min(1.0, 0.25 * norm_fc)
                + min(1.0, 0.20 * norm_peak)
                + min(1.0, 0.15 * norm_temp)
                + min(1.0, 0.10 * alert_rate)
            ) * 100
            mhi_scores[DRIVE_TRAIN_SENSOR_LABELS[s]] = composite_mhi

    df_mhi = pd.Series(mhi_scores).sort_values(ascending=True)
    colors_mhi = [
        "#d62728" if score >= 80 else ("#ff7f0e" if score >= 65 else "#2ca02c")
        for score in df_mhi.values
    ]
    bars = ax_rank.barh(
        df_mhi.index,
        df_mhi.values,
        color=colors_mhi,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    for bar in bars:
        w = bar.get_width()
        action = (
            "Priority Overhaul"
            if w >= 80
            else ("Field Inspection" if w >= 65 else "Routine Maintenance")
        )
        ax_rank.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f} [{action}]",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    ax_rank.axvline(
        80,
        color="red",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-1 Alarm (Score ≥ 80)",
    )
    ax_rank.axvline(
        65,
        color="orange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-2 Advisory (Score ≥ 65)",
    )
    ax_rank.set_title(
        "1. Maintenance Priority: MHI (Calibrated to 2300 kW Group 1)",
        fontsize=11,
        fontweight="bold",
    )
    ax_rank.set_xlabel(
        "Composite Severity Score (0 = Normal, 100 = ISO Trip / 75°C Boundary)",
        fontweight="semibold",
    )
    ax_rank.set_xlim(0, 85)
    ax_rank.legend(loc="lower right")

    # Mechanical Balance Metrics
    balance_metrics = {
        "Output Shaft Drive\n(12RV vs 10RV)": df_op["12RV_Velocity (mm/s)"].mean()
        / df_op["10RV_Velocity (mm/s)"].mean(),
        "Output Inboard\n(11RV vs 9RV)": df_op["11RV_Velocity (mm/s)"].mean()
        / df_op["9RV_Velocity (mm/s)"].mean(),
        "Intermediate Line\n(7RV vs 5RV)": df_op["7RV_Velocity (mm/s)"].mean()
        / df_op["5RV_Velocity (mm/s)"].mean(),
        "Int Outboard\n(8RV vs 6RV)": df_op["8RV_Velocity (mm/s)"].mean()
        / df_op["6RV_Velocity (mm/s)"].mean(),
    }

    df_gauge = pd.Series(balance_metrics)
    colors_gauge = [
        "#2ca02c"
        if TORQUE_SPLIT_TOL_LOWER <= r <= TORQUE_SPLIT_TOL_UPPER
        else ("#ff7f0e" if 0.70 <= r <= 1.30 else "#d62728")
        for r in df_gauge.values
    ]

    bars_g = ax_gauge.bar(
        df_gauge.index,
        df_gauge.values,
        color=colors_gauge,
        edgecolor="black",
        width=0.45,
    )
    ax_gauge.axhline(
        TORQUE_SPLIT_IDEAL,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Ideal 50/50 Mechanical Balance",
    )
    ax_gauge.axhspan(
        TORQUE_SPLIT_TOL_LOWER,
        TORQUE_SPLIT_TOL_UPPER,
        color="green",
        alpha=0.15,
        label="AGMA ±15% Pure Mechanical Band",
    )
    ax_gauge.set_ylabel(
        "Kinematic Velocity Ratio (Upper Branch / Lower Branch)", fontweight="semibold"
    )
    ax_gauge.set_title(
        "2. Pure Mechanical Torque-Split Symmetry Gauge", fontsize=11, fontweight="bold"
    )
    ax_gauge.legend(loc="upper right")

    for bar in bars_g:
        h = bar.get_height()
        ax_gauge.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{h:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "03_vulnerability_ranking_expert.png"), dpi=140
    )
    plt.close(fig)


# ==============================================================================
# 6. MODULE 4: THERMAL-MECHANICAL PHASE SPACE GRID (75°C OEM TRIP CALIBRATED)
# ==============================================================================
def plot_thermal_mechanical_phase_space(df_op, export_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("white")

    # Panel A: Main Motor Flexible Coupling Transmission
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(
        df_op["2RV_Velocity (mm/s)"],
        df_op["3RV_Velocity (mm/s)"],
        c=df_op["2RV_FC"],
        cmap="viridis",
        s=35,
        alpha=0.75,
    )
    cb1 = plt.colorbar(sc1, ax=ax1)
    cb1.set_label("Motor DE Crest Factor (FC)", fontweight="semibold")
    ax1.axvline(
        ISO_GROUP1_ACCEPTABLE,
        color="green",
        linestyle=":",
        alpha=0.6,
        label="ISO Acceptable (2.3 mm/s)",
    )
    ax1.set_title(
        "A. 2.3 MW Main Coupling Transmission (Motor 2RV vs Splitter 3RV)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Motor DE Velocity (mm/s) [Driving Side]", fontweight="semibold")
    ax1.set_ylabel(
        "Splitter Input DE Velocity (mm/s) [Driven Side]", fontweight="semibold"
    )
    ax1.legend(loc="upper left")

    # Panel B: Dual-Output Mechanical Load Split
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(
        df_op["10RV_Velocity (mm/s)"],
        df_op["12RV_Velocity (mm/s)"],
        c=df_op["12RV_FC"],
        cmap="plasma",
        s=35,
        alpha=0.75,
    )
    cb2 = plt.colorbar(sc2, ax=ax2)
    cb2.set_label("Upper Output Crest Factor (FC)", fontweight="semibold")
    max_lim = (
        max(df_op["10RV_Velocity (mm/s)"].max(), df_op["12RV_Velocity (mm/s)"].max())
        * 1.15
    )
    ax2.plot(
        [0, max_lim],
        [0, max_lim],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal 1:1 Symmetrical Load Line",
    )
    ax2.plot(
        [0, max_lim],
        [0, max_lim * TORQUE_SPLIT_TOL_UPPER],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="±15% AGMA Mechanical Boundary",
    )
    ax2.plot(
        [0, max_lim],
        [0, max_lim * TORQUE_SPLIT_TOL_LOWER],
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    ax2.fill_between(
        [0, max_lim],
        [0, max_lim * TORQUE_SPLIT_TOL_LOWER],
        [0, max_lim * TORQUE_SPLIT_TOL_UPPER],
        color="green",
        alpha=0.08,
    )
    ax2.set_title(
        "B. Dual-Output Mechanical Load Split (Lower 10RV vs Upper 12RV)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Lower Output Outboard DE Velocity (mm/s)", fontweight="semibold")
    ax2.set_ylabel("Upper Output Outboard DE Velocity (mm/s)", fontweight="semibold")
    ax2.legend(loc="upper left")

    # Panel C: Motor DE Bearing Thermal Stability Envelope
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(
        df_op["2RV_Temperature (°C)"],
        df_op["2RV_Velocity (mm/s)"],
        c=df_op["2RV_FC"],
        cmap="inferno",
        s=35,
        alpha=0.75,
    )
    cb3 = plt.colorbar(sc3, ax=ax3)
    cb3.set_label("Crest Factor (FC)", fontweight="semibold")
    ax3.axhline(
        ISO_GROUP1_ALERT,
        color="darkorange",
        linestyle="--",
        alpha=0.8,
        label=f"ISO Group 1 Alert ({ISO_GROUP1_ALERT} mm/s)",
    )
    ax3.axvline(
        TEMP_WARNING,
        color="orange",
        linestyle="--",
        alpha=0.8,
        label=f"Thermal Warning ({TEMP_WARNING}°C)",
    )
    ax3.axvline(
        TEMP_TRIP,
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.9,
        label=f"OEM Trip Limit ({TEMP_TRIP}°C)",
    )
    ax3.set_title(
        "C. 2300 kW Motor DE: Hydrodynamic Film vs Binding Runaway",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Bearing Temperature (°C)", fontweight="semibold")
    ax3.set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
    ax3.set_xlim(30, 80)
    ax3.legend(loc="upper left")

    # Panel D: RENK #2092421 Sump Friction Equilibrium
    ax4 = axes[1, 1]
    sc4 = ax4.scatter(
        df_op["3RV_Temperature (°C)"],
        df_op["3RV_Acceleration (g)"],
        c=df_op["3RV_FC"],
        cmap="magma",
        s=35,
        alpha=0.75,
    )
    cb4 = plt.colorbar(sc4, ax=ax4)
    cb4.set_label("Crest Factor (FC)", fontweight="semibold")
    ax4.axvline(
        TEMP_WARNING,
        color="orange",
        linestyle="--",
        alpha=0.8,
        label=f"Sump Warning ({TEMP_WARNING}°C)",
    )
    ax4.axvline(
        TEMP_TRIP,
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.9,
        label=f"Sump Trip ({TEMP_TRIP}°C)",
    )
    ax4.set_title(
        "D. Splitter Input Pinion: Mesh Friction vs Oil Sump Temperature",
        fontsize=11,
        fontweight="bold",
    )
    ax4.set_xlabel("Housing / Sump Temperature (°C)", fontweight="semibold")
    ax4.set_ylabel("RMS Acceleration (g)", fontweight="semibold")
    ax4.set_xlim(30, 80)
    ax4.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "04_thermal_mechanical_envelope_expert.png"), dpi=140
    )
    plt.close(fig)


# ==============================================================================
# 7. MODULE 5: STEWART'S DEFECT TRAJECTORY (HEAVY GEARMESH CALIBRATED)
# ==============================================================================
def plot_defect_lifecycle_trajectory(
    df_op, export_dir, target_stages=["3RV", "7RV", "5RV"]
):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("white")

    # Boundaries adapted for 2300 kW heavy meshing noise floors
    ax.axhspan(
        1.5,
        3.5,
        0.0,
        0.25,
        color="green",
        alpha=0.10,
        label="Zone 1: Healthy Continuous Mesh (FC < 3.5)",
    )
    ax.axhspan(
        5.0,
        8.5,
        0.0,
        0.25,
        color="orange",
        alpha=0.15,
        label="Zone 2: Incipient Tooth Microcrack (FC ≥ 5.0, Low RMS)",
    )
    ax.axhspan(
        4.0,
        8.5,
        0.25,
        0.80,
        color="red",
        alpha=0.15,
        label="Zone 3: Spall Propagation Across Mating Teeth",
    )
    ax.axhspan(
        1.5,
        4.0,
        0.25,
        0.80,
        color="darkred",
        alpha=0.20,
        label="Zone 4: Widespread Flank Shear (Severe RMS Explosion)",
    )

    colors = {"3RV": "#1f77b4", "7RV": "#d62728", "5RV": "#ff7f0e"}
    for sensor in target_stages:
        acc = df_op[f"{sensor}_Acceleration (g)"]
        fc = df_op[f"{sensor}_FC"]

        ax.plot(
            acc,
            fc,
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            color=colors.get(sensor, "gray"),
            label=f"Phase Drift Path: {DRIVE_TRAIN_SENSOR_LABELS[sensor]}",
        )
        ax.scatter(
            acc.iloc[0], fc.iloc[0], color=colors.get(sensor, "gray"), s=70, marker="o"
        )
        ax.scatter(
            acc.iloc[-1],
            fc.iloc[-1],
            color=colors.get(sensor, "gray"),
            s=160,
            marker="X",
            edgecolor="black",
            zorder=5,
            label=f"Current Operating Point ({sensor})",
        )

    ax.set_title(
        "Stewart's Defect Life-Cycle: Phase Trajectory (RENK Reducer Heavy Meshing)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "RMS Acceleration [g] (Continuous Meshing Noise Floor)", fontweight="semibold"
    )
    ax.set_ylabel(
        "Crest Factor [FC] (Impact Shock Concentration)", fontweight="semibold"
    )
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "05_gear_defect_lifecycle_trajectory.png"), dpi=140
    )
    plt.close(fig)


# ==============================================================================
# 8. MODULE 6: SHAFT DEFLECTION & CASING SKEW (IB vs. OB DIFFERENTIAL)
# ==============================================================================
def plot_shaft_deflection_matrix(df_op, export_dir):
    skews = []
    for shaft, (ib, ob) in SHAFT_ALIGNED_PAIRS.items():
        v_diff = (
            df_op[f"{ib}_Velocity (mm/s)"].mean()
            - df_op[f"{ob}_Velocity (mm/s)"].mean()
        )
        peak_diff = df_op[f"{ib}_VC_G (g)"].mean() - df_op[f"{ob}_VC_G (g)"].mean()
        temp_diff = (
            df_op[f"{ib}_Temperature (°C)"].mean()
            - df_op[f"{ob}_Temperature (°C)"].mean()
        )
        skews.append(
            {
                "Shaft Line": shaft,
                "Δ Velocity (mm/s)": v_diff,
                "Δ Peak Shock (g)": peak_diff,
                "Δ Temperature (°C)": temp_diff,
            }
        )

    df_skew = pd.DataFrame(skews).set_index("Shaft Line")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    # Dynamic Shock Skew (Separating forces tilting shafts)
    colors1 = [
        "#d62728" if abs(x) > 0.3 else "#2ca02c" for x in df_skew["Δ Peak Shock (g)"]
    ]
    bars1 = ax1.barh(
        df_skew.index,
        df_skew["Δ Peak Shock (g)"],
        color=colors1,
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.axvline(0, color="black", linewidth=1.2)
    ax1.axvspan(
        -0.2, 0.2, color="green", alpha=0.15, label="Parallel Contact Band (±0.2g)"
    )
    ax1.set_title(
        "1. Dynamic Shaft Deflection: Inboard vs. Outboard Peak Shock",
        fontweight="bold",
    )
    ax1.set_xlabel("Δ Peak Shock (Inboard - Outboard) [g]", fontweight="semibold")
    ax1.legend(loc="lower right")

    for bar in bars1:
        w = bar.get_width()
        ax1.text(
            w + (0.02 if w >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.2f}g",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    # Thermal Skew (Differential bearing axial/radial loading)
    colors2 = [
        "#ff7f0e" if abs(x) > 3.0 else "#1f77b4" for x in df_skew["Δ Temperature (°C)"]
    ]
    bars2 = ax2.barh(
        df_skew.index,
        df_skew["Δ Temperature (°C)"],
        color=colors2,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.axvspan(
        -2.0, 2.0, color="blue", alpha=0.1, label="Balanced Thermal Band (±2.0°C)"
    )
    ax2.set_title("2. Thermal Skew: Differential Bearing Loading", fontweight="bold")
    ax2.set_xlabel("Δ Temperature (Inboard - Outboard) [°C]", fontweight="semibold")
    ax2.legend(loc="lower right")

    for bar in bars2:
        w = bar.get_width()
        ax2.text(
            w + (0.2 if w >= 0 else -0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.1f}°C",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle(
        "Shaft Alignment & Gear Taper Contact Matrix (RENK #2092421 Casing Rigidity)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "06_shaft_deflection_matrix.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 9. MODULE 7: BACKLASH & REBOUND RATIO (VCC_G / VC_G)
# ==============================================================================
def plot_backlash_chatter_index(df_op, export_dir, target_stages=["3RV", "4RV"]):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    for sensor in target_stages:
        ratio = df_op[f"{sensor}_VCC_G (g)"] / (df_op[f"{sensor}_VC_G (g)"] + 1e-6)
        ax.plot(
            df_op.index,
            ratio,
            linewidth=1.3,
            label=f"Chatter Ratio: {DRIVE_TRAIN_SENSOR_LABELS[sensor]}",
        )

    ax.axhspan(
        1.6,
        2.1,
        color="green",
        alpha=0.12,
        label="Normal Unidirectional Contact (1.6 - 2.1)",
    )
    ax.axhspan(
        2.1,
        2.5,
        color="orange",
        alpha=0.15,
        label="Mild Tooth Flank Separation (2.1 - 2.5)",
    )
    ax.axhspan(
        2.5,
        4.0,
        color="red",
        alpha=0.15,
        label="Severe Backlash Chattering / Tooth Play (> 2.5)",
    )
    ax.axhline(2.5, color="red", linestyle="--", linewidth=1.5)

    ax.set_title(
        "Gear Mesh Dynamic Stability: Pure Mechanical Backlash & Rebound Ratio (VCC_G / VC_G)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Impact Rebound Ratio (Pk-to-Pk / Peak)", fontweight="semibold")
    ax.set_xlabel("Acquisition Timestamp", fontweight="semibold")
    ax.set_ylim(1.4, 3.2)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "07_tooth_backlash_chatter_index.png"), dpi=140
    )
    plt.close(fig)


# ==============================================================================
# 10. MODULE 8: TORQUE-SPLIT CO-VARIANCE (PURE MECHANICAL SPLIT)
# ==============================================================================
def plot_torque_split_covariance(df_op, export_dir):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor("white")

    lower_out = df_op["10RV_Velocity (mm/s)"]
    upper_out = df_op["12RV_Velocity (mm/s)"]
    fc_severity = df_op["12RV_FC"]

    scatter = ax.scatter(
        lower_out,
        upper_out,
        c=fc_severity,
        cmap="plasma",
        s=35,
        alpha=0.7,
        edgecolor="none",
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("Upper Output Crest Factor (FC)", fontweight="semibold")

    max_val = max(lower_out.max(), upper_out.max()) * 1.15
    ax.plot(
        [0, max_val],
        [0, max_val],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal 50/50 Mechanical Balance (y = x)",
    )
    ax.plot(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_UPPER],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="±15% AGMA 6014-B15 Mechanical Boundary",
    )
    ax.plot(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_LOWER],
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    ax.fill_between(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_LOWER],
        [0, max_val * TORQUE_SPLIT_TOL_UPPER],
        color="green",
        alpha=0.08,
    )

    ax.set_title(
        "Pure Mechanical Torque Split: Upper (12RV) vs. Lower (10RV) Driving Lines",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Lower Branch Output Velocity [10RV] (mm/s)", fontweight="semibold")
    ax.set_ylabel("Upper Branch Output Velocity [12RV] (mm/s)", fontweight="semibold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "08_torque_split_covariance.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 11. PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("BALL MILL CONDITION MONITORING & ADVANCED DIAGNOSTIC PIPELINE")
    print("Configuration: 2300 kW Rigid Support Motor | RENK #2092421 Reducer")
    print("Standards: ISO 20816-3 Group 1 Rigid | AGMA 6014-B15 (±15% Pure Mechanical)")
    print("=" * 80)

    export_dir = "../../reports/figures/expert_analysis"
    os.makedirs(export_dir, exist_ok=True)

    data_source_path = "../../data/interim/01_processed.pkl"
    try:
        _, df_operational = load_and_preprocess_telemetry(data_source_path)
        df_operational.info()
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    print("\n[Executing Calibrated Diagnostic Visualizations...]")
    plot_risk_quadrant_matrix(df_operational, export_dir)
    plot_bifurcated_heatmap(df_operational, export_dir)
    plot_maintenance_priority_and_balance(df_operational, export_dir)
    plot_thermal_mechanical_phase_space(df_operational, export_dir)
    plot_defect_lifecycle_trajectory(df_operational, export_dir)
    plot_shaft_deflection_matrix(df_operational, export_dir)
    plot_backlash_chatter_index(df_operational, export_dir)
    plot_torque_split_covariance(df_operational, export_dir)

    print(
        f"\n✔ All 8 calibrated industrial visualizations generated in: '{export_dir}'"
    )
    print("=" * 80)

    # Motor: Compare 10RV (Non Drive End) vs 12RV (Drive End  Coupled)
    df_op = df_operational.copy()
    df_op.info()
    lower_out = df_op["10RV_Velocity (mm/s)"]
    upper_out = df_op["12RV_Velocity (mm/s)"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # 1. Motor bearings
    ax1.plot(
        df_op.index,
        lower_out,
        label="10RV lower output",
        color="#e54c0b",
    )
    ax1.plot(
        df_op.index,
        upper_out,
        label="12RV upper ouput",
        color="#1f77b4",
        linewidth=1.6,
    )
    ax1.set_ylabel("Vibration Velocity (mm/s)", fontweight="semibold")
    ax1.set_title("Motor Bearings Vibration: 10RV  vs. 12RV ", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)


if __name__ == "__main__":
    main()
