import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==============================================================================
# 1. GLOBAL CONFIGURATION & KINEMATIC METADATA (12 ACTIVE SENSORS)
# ==============================================================================
# Dual-Pinion Torque-Split Reducer Mapping (Excludes uninstalled 13RV-16RV)
DRIVE_TRAIN_SENSOR_LABELS = {
    # --- Drive Motor ---
    "1RV": "Motor NDE (1RV)",
    "2RV": "Motor DE - Main Coupling (2RV)",
    # --- Primary Torque-Splitter: Center Input Pinion Shaft ---
    "3RV": "Splitter Input Inboard DE (3RV)",
    "4RV": "Splitter Input Outboard NDE (4RV)",
    # --- Primary Torque-Splitter: Lower Branch ---
    "5RV": "Splitter Lower Int-1 Inboard (5RV)",
    "6RV": "Splitter Lower Int-1 Outboard (6RV)",
    "9RV": "Splitter Lower Output Inboard (9RV)",
    "10RV": "Splitter Lower Output Outboard DE (10RV)",  # Drives Bottom Spacer Shaft
    # --- Primary Torque-Splitter: Upper Branch ---
    "7RV": "Splitter Upper Int-2 Inboard (7RV)",
    "8RV": "Splitter Upper Int-2 Outboard (8RV)",
    "11RV": "Splitter Upper Output Inboard (11RV)",
    "12RV": "Splitter Upper Output Outboard DE (12RV)",  # Drives Top Spacer Shaft
}

ACTIVE_SENSORS = list(DRIVE_TRAIN_SENSOR_LABELS.keys())

# Kinematic Functional Subsystems
KINEMATIC_GROUPS = {
    "Drive Motor": (["1RV", "2RV"], "#1f77b4", "s"),
    "Center Input Shaft": (["3RV", "4RV"], "#2ca02c", "D"),
    "Upper Branch (To Top Pinion)": (["7RV", "8RV", "11RV", "12RV"], "#d62728", "^"),
    "Lower Branch (To Bottom Pinion)": (["5RV", "6RV", "9RV", "10RV"], "#ff7f0e", "o"),
}

# Inboard (Motor Side) vs. Outboard (Far Side) Shaft Pairs for Alignment Skew
SHAFT_ALIGNED_PAIRS = {
    "Center Input Shaft": ("3RV", "4RV"),
    "Upper Int-2 Shaft": ("7RV", "8RV"),
    "Lower Int-1 Shaft": ("5RV", "6RV"),
    "Lower Output Shaft": ("9RV", "10RV"),
    "Upper Output Shaft": ("11RV", "12RV"),
}

# Mirrored Kinematic Stages (Upper vs. Lower Branch) for Symmetry Tracking
TWIN_BRANCH_PAIRS = [
    ("7RV", "5RV", "Intermediate Inboard"),
    ("8RV", "6RV", "Intermediate Outboard"),
    ("11RV", "9RV", "Output Inboard"),
    ("12RV", "10RV", "Output Outboard (Spacer Drive)"),
]

# Matplotlib Publication Styling
mpl.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["lines.linewidth"] = 1.3
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9


# ==============================================================================
# 2. DATA INGESTION, RESAMPLING & OPERATIONAL STATE FILTERING
# ==============================================================================
def load_and_preprocess_telemetry(data_path="../../data/interim/01_processed.pkl"):
    """
    Loads interim time series, standardizes DatetimeIndex, resamples to uniform
    10-minute intervals, and filters out non-operational machine outages.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source interim data not found at: {data_path}")

    df = pd.read_pickle(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()

    # Uniform 10-minute grid cleaning (drops empty acquisition intervals)
    df_resampled = df.resample(rule="10min").mean().dropna(how="all")
    df_resampled.info()

    # Operational Grinding Filter: clear machine shutdowns where Acc < 0.10g
    # (Eliminates division-by-zero K-Factor spikes and zero-load noise)
    is_operational = df_resampled["1RV_Acceleration (g)"] >= 0.10
    df_operational = df_resampled[is_operational].copy()

    print(f"✔ Data Pipeline Complete:")
    print(f"   - Full Resampled Records: {len(df_resampled)} timestamps")
    print(
        f"   - Active Milling Records: {len(df_operational)} timestamps (Outages Cleared: {(~is_operational).sum()})"
    )

    return df_resampled, df_operational


# ==============================================================================
# 3. DIAGNOSTIC MODULE 1: ASSET RISK QUADRANT MATRIX WITH TWIN TIE-LINES
# ==============================================================================
def plot_risk_quadrant_matrix(df_op, export_dir):
    """
    Evaluates ISO Velocity against Crest Factor (FC).
    Draws dashed symmetry vectors between mirrored kinematic twin pairs.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    sensor_centroids = {}
    for group_name, (sensors, color, marker) in KINEMATIC_GROUPS.items():
        first = True
        for s in sensors:
            vel_col, fc_col = f"{s}_Velocity (mm/s)", f"{s}_FC"
            if vel_col in df_op.columns and fc_col in df_op.columns:
                v_data = df_op[vel_col]
                f_data = df_op[fc_col]
                v_mean, f_mean = v_data.mean(), f_data.mean()
                sensor_centroids[s] = (v_mean, f_mean)

                ax.scatter(v_data, f_data, color=color, alpha=0.15, s=15, label=None)
                ax.scatter(
                    v_mean,
                    f_mean,
                    color=color,
                    edgecolor="black",
                    s=170,
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
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec=color,
                        lw=1.4,
                        alpha=0.9,
                    ),
                )
                first = False

    # Symmetry Tie-Lines across Upper vs. Lower Stages
    for s_up, s_low, pair_label in TWIN_BRANCH_PAIRS:
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
            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.annotate(
                f"Δ {s_up}-{s_low}",
                (mid_x, mid_y),
                fontsize=7.5,
                color="#444444",
                bbox=dict(
                    boxstyle="square,pad=0.1", fc="#f0f0f0", ec="none", alpha=0.7
                ),
            )

    # Standard Diagnostic Thresholds
    ax.axvline(
        2.3,
        color="darkorange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="ISO 10816 Good/Acceptable (2.3 mm/s)",
    )
    ax.axvline(
        4.5,
        color="red",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="ISO 10816 Alert Boundary (4.5 mm/s)",
    )
    ax.axhline(
        5.0,
        color="crimson",
        linestyle="-.",
        linewidth=1.6,
        alpha=0.9,
        label="Incipient Microcrack Threshold (FC = 5.0)",
    )
    ax.axhline(
        3.5,
        color="green",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8,
        label="Normal Continuous Rolling (FC = 3.5)",
    )

    ax.text(
        0.2,
        9.2,
        "QUADRANT II: INCIPIENT SURFACE FATIGUE\n(Microcracks / Tooth Spalling - Normal Macro Velocity)",
        fontsize=9,
        fontweight="bold",
        color="crimson",
        bbox=dict(boxstyle="square", fc="#ffe6e6", ec="none", alpha=0.75),
    )
    ax.text(
        0.2,
        2.2,
        "QUADRANT I: HEALTHY CONTINUOUS MESH\n(Low Structural Severity, Low Impulsive Shocks)",
        fontsize=9,
        fontweight="bold",
        color="darkgreen",
        bbox=dict(boxstyle="square", fc="#e6ffe6", ec="none", alpha=0.75),
    )
    ax.text(
        3.0,
        9.2,
        "QUADRANT IV: CRITICAL ADVANCED FAILURE\n(Severe Unbalance/Looseness + Surface Breakdown)",
        fontsize=9,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="square", fc="#ffd9cc", ec="none", alpha=0.75),
    )
    ax.text(
        3.0,
        2.2,
        "QUADRANT III: MACRO-STRUCTURAL DEFECT\n(Shaft Misalignment / Coupling Wear - No Surface Pitting)",
        fontsize=9,
        fontweight="bold",
        color="#805500",
        bbox=dict(boxstyle="square", fc="#fff5e6", ec="none", alpha=0.75),
    )

    ax.set_title(
        "Dual-Pinion Asset Quadrant Matrix: Kinematic Load Symmetry & Microcrack Impulses",
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
        "Crest Factor [FC] (Impulsive Impact Severity: Peak / RMS)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 5.8)
    ax.set_ylim(1.8, 10.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "01_risk_quadrant_matrix_expert.png"), dpi=140)
    plt.close(fig)
    print("   -> Generated: 01_risk_quadrant_matrix_expert.png")


# ==============================================================================
# 4. DIAGNOSTIC MODULE 2: BIFURCATED DRIVE TRAIN HEATMAP & ASYMMETRY DELTA
# ==============================================================================
def plot_bifurcated_heatmap(df_op, export_dir):
    """
    Renders heat intensity structured by kinematic power sequence.
    Computes an explicit Upper-to-Lower Branch Asymmetry Delta row.
    """
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
                "Kurtosis (K Factor)": df_op[f"{s}_K Factor"].mean(),
                "Temperature (°C)": df_op[f"{s}_Temperature (°C)"].mean(),
                "Microcrack Alert Rate (%)": (df_op[f"{s}_FC"] >= 5.0).mean() * 100,
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
    branch_delta.name = "=== BRANCH DELTA (Upper - Lower) ==="

    df_heat_with_delta = pd.concat([df_heat, pd.DataFrame([branch_delta])])

    df_norm = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min())
    df_norm.loc["=== BRANCH DELTA (Upper - Lower) ==="] = np.nan

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        df_norm,
        annot=df_heat_with_delta.values,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={
            "label": "Normalized Stress Intensity (0 = Baseline, 1 = Max Severity)"
        },
        linewidths=1.2,
        linecolor="#e0e0e0",
        mask=df_norm.isna(),
        ax=ax,
    )

    ax.text(
        0.5,
        len(df_heat_with_delta) - 0.5,
        f"Torque Split Symmetry Delta: ΔVel={branch_delta['Velocity (mm/s)']:.2f} mm/s | ΔPeak={branch_delta['Peak Shock VC_G (g)']:.2f}g | ΔTemp={branch_delta['Temperature (°C)']:.1f}°C",
        fontsize=9.5,
        fontweight="bold",
        color="darkblue",
        va="center",
    )

    ax.set_title(
        "Kinematic Stress Distribution Heatmap: Center Input & Dual Torque-Split Branches",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Monitored Telemetry Channel", fontsize=11, fontweight="semibold")
    ax.set_ylabel(
        "Drive Train Position (Power Flow Sequence)", fontsize=11, fontweight="semibold"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "02_drive_train_heatmap_expert.png"), dpi=140)
    plt.close(fig)
    print("   -> Generated: 02_drive_train_heatmap_expert.png")


# ==============================================================================
# 5. DIAGNOSTIC MODULE 3: COMPOSITE MHI PRIORITY & TORQUE-SPLIT GAUGE
# ==============================================================================
def plot_maintenance_priority_and_balance(df_op, export_dir):
    """
    Ranks bearings by Composite Machine Health Index (MHI) and displays
    the Upper/Lower branch kinematic balance ratio.
    """
    fig, (ax_rank, ax_gauge) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.2, 0.8]}
    )
    fig.patch.set_facecolor("white")

    mhi_scores = {}
    for s in ACTIVE_SENSORS:
        norm_v = df_op[f"{s}_Velocity (mm/s)"].mean() / 7.1
        norm_fc = df_op[f"{s}_FC"].mean() / 6.0
        norm_peak = df_op[f"{s}_VC_G (g)"].mean() / 15.0
        alert_rate = (df_op[f"{s}_FC"] >= 5.0).mean()
        composite_mhi = (
            (0.35 * norm_v)
            + (0.30 * norm_fc)
            + (0.20 * norm_peak)
            + (0.15 * alert_rate)
        ) * 100
        mhi_scores[DRIVE_TRAIN_SENSOR_LABELS[s]] = composite_mhi

    df_mhi = pd.Series(mhi_scores).sort_values(ascending=True)

    colors_mhi = [
        "#d62728" if score >= 75 else ("#ff7f0e" if score >= 51 else "#2ca02c")
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
            "Immediate Overhaul"
            if w >= 75
            else ("Field Inspection" if w >= 51 else "Routine Lube")
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
        75,
        color="red",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-1 Alarm (Score ≥ 75)",
    )
    ax_rank.axvline(
        51,
        color="orange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-2 Advisory (Score ≥ 51)",
    )
    ax_rank.set_title(
        "1. Bearing Maintenance Priority: Composite Health Index (MHI)",
        fontsize=11,
        fontweight="bold",
    )
    ax_rank.set_xlabel(
        "Composite Severity Score (0 = Pristine, 100 = Critical)", fontweight="semibold"
    )
    ax_rank.set_xlim(0, 85)
    ax_rank.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax_rank.legend(loc="lower right", fontsize=8.5)

    # Balance Ratios: Upper vs. Lower Stages
    balance_metrics = {
        "Output Drive Shaft\n(12RV vs 10RV)": df_op["12RV_Velocity (mm/s)"].mean()
        / df_op["10RV_Velocity (mm/s)"].mean(),
        "Output Inboard\n(11RV vs 9RV)": df_op["11RV_Velocity (mm/s)"].mean()
        / df_op["9RV_Velocity (mm/s)"].mean(),
        "Intermediate Stage\n(7RV vs 5RV)": df_op["7RV_Velocity (mm/s)"].mean()
        / df_op["5RV_Velocity (mm/s)"].mean(),
        "Int Outboard\n(8RV vs 6RV)": df_op["8RV_Velocity (mm/s)"].mean()
        / df_op["6RV_Velocity (mm/s)"].mean(),
    }

    df_gauge = pd.Series(balance_metrics)
    colors_gauge = [
        "#2ca02c"
        if 0.85 <= r <= 1.15
        else ("#ff7f0e" if 0.70 <= r <= 1.30 else "#d62728")
        for r in df_gauge.values
    ]

    bars_g = ax_gauge.bar(
        df_gauge.index,
        df_gauge.values,
        color=colors_gauge,
        edgecolor="black",
        linewidth=0.8,
        width=0.45,
    )
    ax_gauge.axhline(
        1.0,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Ideal 50/50 Balance (1.0)",
    )
    ax_gauge.axhspan(
        0.85, 1.15, color="green", alpha=0.15, label="Acceptable Band (±15%)"
    )
    ax_gauge.set_ylabel(
        "Kinematic Ratio (Upper Branch / Lower Branch)", fontweight="semibold"
    )
    ax_gauge.set_title(
        "2. Torque-Split Symmetry Gauge (Upper / Lower)", fontsize=11, fontweight="bold"
    )
    ax_gauge.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_gauge.legend(loc="upper right", fontsize=8.5)

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

    plt.suptitle(
        "Maintenance Action & Kinematic Balance Diagnostic Suite",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "03_vulnerability_ranking_expert.png"), dpi=140
    )
    plt.close(fig)
    print("   -> Generated: 03_vulnerability_ranking_expert.png")


# ==============================================================================
# 6. DIAGNOSTIC MODULE 4: THERMAL-MECHANICAL PHASE SPACE GRID
# ==============================================================================
def plot_thermal_mechanical_phase_space(df_op, export_dir):
    """
    Renders 4-quadrant phase space assessing coupling dynamics, twin output
    symmetry, hydrodynamic oil shear, and sump friction equilibrium.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("white")

    # Panel 1: Coupling Transmission (2RV vs. 3RV)
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(
        df_op["2RV_Velocity (mm/s)"],
        df_op["3RV_Velocity (mm/s)"],
        c=df_op["2RV_FC"],
        cmap="viridis",
        s=35,
        alpha=0.75,
        edgecolor="none",
    )
    cb1 = plt.colorbar(sc1, ax=ax1)
    cb1.set_label("Motor DE Crest Factor (FC)", fontweight="semibold")
    ax1.set_title(
        "A. High-Speed Coupling Transmission (Motor 2RV vs Splitter 3RV)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Motor DE Velocity (mm/s) [Drive Side]", fontweight="semibold")
    ax1.set_ylabel(
        "Splitter Input DE Velocity (mm/s) [Driven Side]", fontweight="semibold"
    )
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Twin Output Spacer Shaft Drives (10RV vs. 12RV)
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(
        df_op["10RV_Velocity (mm/s)"],
        df_op["12RV_Velocity (mm/s)"],
        c=df_op["12RV_FC"],
        cmap="plasma",
        s=35,
        alpha=0.75,
        edgecolor="none",
    )
    cb2 = plt.colorbar(sc2, ax=ax2)
    cb2.set_label("Upper Output Crest Factor (FC)", fontweight="semibold")
    max_lim = (
        max(df_op["10RV_Velocity (mm/s)"].max(), df_op["12RV_Velocity (mm/s)"].max())
        * 1.1
    )
    ax2.plot(
        [0, max_lim],
        [0, max_lim],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="1:1 Symmetrical Load Line",
    )
    ax2.set_title(
        "B. Dual-Output Load Split (Lower 10RV vs Upper 12RV)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Lower Output Outboard DE Velocity (mm/s)", fontweight="semibold")
    ax2.set_ylabel("Upper Output Outboard DE Velocity (mm/s)", fontweight="semibold")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Motor Bearing Thermal-Mechanical Envelope (2RV)
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(
        df_op["2RV_Temperature (°C)"],
        df_op["2RV_Velocity (mm/s)"],
        c=df_op["2RV_FC"],
        cmap="inferno",
        s=35,
        alpha=0.75,
        edgecolor="none",
    )
    cb3 = plt.colorbar(sc3, ax=ax3)
    cb3.set_label("Crest Factor (FC - Impulsiveness)", fontweight="semibold")
    ax3.axhline(
        4.5,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Elevated Vibration Alert (4.5 mm/s)",
    )
    ax3.axvline(
        46.0,
        color="darkorange",
        linestyle="--",
        alpha=0.7,
        label="Thermal Equilibrium Limit (46°C)",
    )
    ax3.set_title(
        "C. Motor DE Bearing (2RV): Viscous Shear vs. Mechanical Binding",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Bearing Operating Temperature (°C)", fontweight="semibold")
    ax3.set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Primary Splitter Sump Dissipation (3RV)
    ax4 = axes[1, 1]
    sc4 = ax4.scatter(
        df_op["3RV_Temperature (°C)"],
        df_op["3RV_Acceleration (g)"],
        c=df_op["3RV_FC"],
        cmap="magma",
        s=35,
        alpha=0.75,
        edgecolor="none",
    )
    cb4 = plt.colorbar(sc4, ax=ax4)
    cb4.set_label("Crest Factor (FC - Impulsiveness)", fontweight="semibold")
    ax4.axvline(
        48.0,
        color="darkorange",
        linestyle="--",
        alpha=0.7,
        label="Sump Thermal Saturation (48°C)",
    )
    ax4.set_title(
        "D. Splitter Center Input (3RV): Mesh Friction vs Acceleration",
        fontsize=11,
        fontweight="bold",
    )
    ax4.set_xlabel("Input Housing Temperature (°C)", fontweight="semibold")
    ax4.set_ylabel("RMS Acceleration (g)", fontweight="semibold")
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Cross-Domain Phase Space Diagnostic Suite: Dual-Pinion Drive Train",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "04_thermal_mechanical_envelope_expert.png"), dpi=140
    )
    plt.close(fig)
    print("   -> Generated: 04_thermal_mechanical_envelope_expert.png")


# ==============================================================================
# 7. DIAGNOSTIC MODULE 5: STEWART'S GEAR DEFECT PROGRESSION TRAJECTORY
# ==============================================================================
def plot_defect_lifecycle_trajectory(
    df_op,
    export_dir,
    target_stages=[
        "3RV",
        "7RV",
        "5RV",
        "11RV",
        "9RV",
        "12RV",
        "10RV",
        "8RV",
        "6RV",
        "4RV",
    ],
):
    """
    Tracks trajectory through Crest Factor vs. RMS Acceleration phase space.
    Prevents false negatives when Crest Factor drops as multi-tooth spalling spreads.
    """
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("white")

    # Diagnostic Severity Bands
    ax.axhspan(
        1.5,
        3.5,
        0.0,
        0.3,
        color="green",
        alpha=0.10,
        label="Zone 1: Healthy Continuous Mesh (FC < 3.5)",
    )
    ax.axhspan(
        5.0,
        8.0,
        0.0,
        0.3,
        color="orange",
        alpha=0.15,
        label="Zone 2: Incipient Microcrack / Local Spall (FC ≥ 5.0)",
    )
    ax.axhspan(
        4.0,
        8.0,
        0.3,
        1.0,
        color="pink",
        alpha=0.15,
        label="Zone 3: Spall Propagation Across Flanks",
    )
    ax.axhspan(
        1.5,
        4.0,
        0.3,
        1.0,
        color="red",
        alpha=0.20,
        label="Zone 4: Widespread Flank Shear (High RMS, Low FC)",
    )

    colors = {
        "3RV": "#1f77b4",
        "7RV": "#d62728",
        "5RV": "#ff7f0e",
        "11RV": "#2ca02c",
        "9RV": "#9467bd",
        "12RV": "#8c564b",
        "10RV": "#e377c2",
        "8RV": "#7f7f7f",
        "6RV": "#bcbd22",
        "4RV": "#031654",
    }
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
            acc.iloc[0],
            fc.iloc[0],
            color=colors.get(sensor, "gray"),
            s=80,
            marker="o",
            label=f"Start ({sensor})",
        )
        ax.scatter(
            acc.iloc[-1],
            fc.iloc[-1],
            color=colors.get(sensor, "gray"),
            s=160,
            marker="X",
            edgecolor="black",
            zorder=5,
            label=f"Current State ({sensor})",
        )

    ax.set_title(
        "Stewart's Gear Defect Life-Cycle Trajectory: Tracking Spall Progression & Saturation",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "RMS Acceleration [g] (Continuous Meshing Noise Floor)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Crest Factor [FC] (Impulsive Peak-to-RMS Concentration)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "05_gear_defect_lifecycle_trajectory.png"), dpi=140
    )
    plt.close(fig)
    print("   -> Generated: 05_gear_defect_lifecycle_trajectory.png")


# ==============================================================================
# 8. DIAGNOSTIC MODULE 6: SHAFT DEFLECTION & CASING SKEW MATRIX
# ==============================================================================
def plot_shaft_deflection_matrix(df_op, export_dir):
    """
    Computes Δ(Inboard - Outboard) across all 5 reducer shafts.
    Exposes shaft tilting, casing twist, and gear tooth edge-chipping risks.
    """
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

    # Panel 1: Peak Shock Skew (Dynamic Force Asymmetry)
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
        "1. Dynamic Shaft Skew: Inboard vs. Outboard Peak Shock", fontweight="bold"
    )
    ax1.set_xlabel("Δ Peak Shock (Inboard - Outboard) [g]", fontweight="semibold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")

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

    # Panel 2: Thermal Skew (Bearing Load Differential)
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
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

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
        "Shaft Alignment & Gear Taper Contact Diagnostic (IB vs. OB Differential)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "06_shaft_deflection_matrix.png"), dpi=140)
    plt.close(fig)
    print("   -> Generated: 06_shaft_deflection_matrix.png")


# ==============================================================================
# 9. DIAGNOSTIC MODULE 7: TOOTH BACKLASH & IMPACT CHATTERING RATIO
# ==============================================================================
def plot_backlash_chatter_index(df_op, export_dir, target_stages=["3RV", "4RV"]):
    """
    Evaluates impact rebound ratio (VCC_G / VC_G). Detects tooth flank separation
    and destructive rattling inside excessive backlash clearances.
    """
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
        "Gear Mesh Dynamic Stability: Tooth Backlash & Impact Rebound Ratio (VCC_G / VC_G)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Impact Rebound Ratio (Pk-to-Pk / Peak)", fontweight="semibold")
    ax.set_xlabel("Acquisition Timestamp", fontweight="semibold")
    ax.set_ylim(1.4, 3.2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "07_tooth_backlash_chatter_index.png"), dpi=140
    )
    plt.close(fig)
    print("   -> Generated: 07_tooth_backlash_chatter_index.png")


# ==============================================================================
# 10. DIAGNOSTIC MODULE 8: TORQUE-SPLIT CO-VARIANCE & PHASE LAG PLOT
# ==============================================================================
def plot_torque_split_covariance(df_op, export_dir):
    """
    Maps dynamic co-variance between Upper Output (12RV) and Lower Output (10RV).
    Detects torsional hunting, spacer shaft windup, and tooth pitch mismatch.
    """
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

    # 1:1 Reference Line and 15% Tolerance Envelope
    max_val = max(lower_out.max(), upper_out.max()) * 1.1
    ax.plot(
        [0, max_val],
        [0, max_val],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal 50/50 Torque Balance (y = x)",
    )
    ax.plot(
        [0, max_val],
        [0, max_val * 1.15],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="±15% Balance Tolerance Band",
    )
    ax.plot([0, max_val], [0, max_val * 0.85], color="gray", linestyle=":", alpha=0.7)
    ax.fill_between(
        [0, max_val],
        [0, max_val * 0.85],
        [0, max_val * 1.15],
        color="green",
        alpha=0.08,
    )

    ax.set_title(
        "Torque-Split Dynamic Co-Variance: Upper (12RV) vs. Lower (10RV) Output Lines",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Lower Branch Output Velocity [10RV] (mm/s)",
        fontsize=10.5,
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Upper Branch Output Velocity [12RV] (mm/s)",
        fontsize=10.5,
        fontweight="semibold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "08_torque_split_covariance.png"), dpi=140)
    plt.close(fig)
    print("   -> Generated: 08_torque_split_covariance.png")


# ==============================================================================
# 11. MAIN PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 80)
    print("BALL MILL ADVANCED CONDITION MONITORING & GEAR DIAGNOSTIC PIPELINE")
    print("Architecture: Dual-Pinion Torque-Split Reducer (Active Sensors 1RV - 12RV)")
    print("=" * 80)

    # 1. Output Directories
    figures_dir = "../../reports/figures/expert_analysis"
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Ingest and Clean Telemetry
    data_source_path = "../../data/interim/01_processed.pkl"
    try:
        df_resampled, df_operational = load_and_preprocess_telemetry(data_source_path)
        df_operational.info()
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    # 3. Execute Complete Diagnostic Suite
    print("\n[Executing Diagnostic Visualizations...]")
    plot_risk_quadrant_matrix(df_operational, figures_dir)
    plot_bifurcated_heatmap(df_operational, figures_dir)
    plot_maintenance_priority_and_balance(df_operational, figures_dir)
    plot_thermal_mechanical_phase_space(df_operational, figures_dir)
    plot_defect_lifecycle_trajectory(df_operational, figures_dir)
    plot_shaft_deflection_matrix(df_operational, figures_dir)
    plot_backlash_chatter_index(df_operational, figures_dir)
    plot_torque_split_covariance(df_operational, figures_dir)

    print(
        "\n✔ Diagnostic suite execution complete. All high-resolution figures saved to:"
    )
    print(f"   '{os.path.abspath(figures_dir)}'")
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
