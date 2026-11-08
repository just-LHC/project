import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
plt.rcParams["figure.figsize"] = (15, 6)

# --------------------------------------------------------------
# 1. Load Feature-Engineered Telemetry
# --------------------------------------------------------------
INPUT_PATH = "../../data/interim/03_features_engineered.pkl"
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Engineered feature dataset missing at: {INPUT_PATH}")

df = pd.read_pickle(INPUT_PATH)
if not isinstance(df.index, pd.DatetimeIndex):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
df = df.sort_index()

print(f"✔ Telemetry loaded: {df.shape[0]} timestamps across {df.shape[1]} features.")

# --------------------------------------------------------------
# 2. Granular Root-Cause Health Tagging Function
# --------------------------------------------------------------
ACTIVE_SENSORS = [f"{i}RV" for i in range(1, 13)]


def assign_iso_health_states_with_root_cause(df_in):
    """Evaluates ISO 20816-3 & thermodynamic limits on each timestamp and logs:

    - health_state (0 = NORMAL, 1 = WARNING, 2 = CRITICAL)
    - fault_sensor: Specific sensor node triggering the event (e.g., '1RV',
    '6RV', '10RV-12RV')
    - fault_metric: The specific physical channel (Velocity, Temp, Kurtosis,
    dT/dt, etc.)
    - fault_detail: Human-readable diagnosis with exact numerical value &
    threshold breached
    - active_violations_count: How many concurrent channel violations occurred
    - all_violations_list: Semicolon-delimited record of all concurrent channel
    breaches
    """
    df_out = df_in.copy()

    n_rows = len(df_out)
    health_states = np.zeros(n_rows, dtype=int)
    fault_sensors = ["NORMAL"] * n_rows
    fault_metrics = ["None"] * n_rows
    fault_details = ["Machine healthy within normal operating parameters"] * n_rows
    active_violations_counts = np.zeros(n_rows, dtype=int)
    all_violations_logs = ["None"] * n_rows

    is_prod_series = (
        df_out["Operating_Regime"] == "REGIME 2: FULL PRODUCTION (2300 kW)"
        if "Operating_Regime" in df_out.columns
        else pd.Series(True, index=df_out.index)
    ).values

    for i in range(n_rows):
        if not is_prod_series[i]:
            continue  # Idle / Startup regimes default to Normal (Class 0)

        row = df_out.iloc[i]
        timestamp_violations = []  # List of tuples: (severity_rank, sensor, metric, detail_str)

        # --- Evaluate each sensor individually ---
        for s in ACTIVE_SENSORS:
            # 1. ISO Vibration Velocity (Zone C: >=4.5, Zone D: >=7.1)
            v_col = f"{s}_Velocity (mm/s)"
            if v_col in df_out.columns:
                v = row[v_col]
                if v >= 7.1:
                    timestamp_violations.append(
                        (
                            2,
                            s,
                            "Velocity",
                            f"{s} Velocity ISO Zone D trip ({v:.2f} mm/s >= 7.1)",
                        )
                    )
                elif v >= 4.5:
                    timestamp_violations.append(
                        (
                            1,
                            s,
                            "Velocity",
                            f"{s} Velocity ISO Zone C warning ({v:.2f} mm/s >= 4.5)",
                        )
                    )

            # 2. Bearing Housing Temperature (Absolute & Rate of Rise)
            t_col = f"{s}_Temperature (°C)"
            if t_col in df_out.columns:
                t = row[t_col]
                if t >= 85.0:
                    timestamp_violations.append(
                        (
                            2,
                            s,
                            "Temperature",
                            f"{s} Bearing severe overheat ({t:.1f}°C >= 85°C)",
                        )
                    )
                elif t >= 70.0:
                    timestamp_violations.append(
                        (
                            1,
                            s,
                            "Temperature",
                            f"{s} Bearing elevated thermal wear ({t:.1f}°C >= 70°C)",
                        )
                    )

            tr_col = f"{s}_temp_rate_of_rise_1hr"
            if tr_col in df_out.columns:
                tr = row[tr_col]
                if tr >= 8.0:
                    timestamp_violations.append(
                        (2, s, "dT/dt", f"{s} Thermal runaway (+{tr:.2f}°C/hr >= 8.0)")
                    )
                elif tr >= 3.0:
                    timestamp_violations.append(
                        (
                            1,
                            s,
                            "dT/dt",
                            f"{s} Abnormal thermal rise (+{tr:.2f}°C/hr >= 3.0)",
                        )
                    )

            # 3. Kurtosis (ISO 13373 Impulsive Shock Standard)
            kurt_col = (
                f"{s}_Kurtosis"
                if f"{s}_Kurtosis" in df_out.columns
                else f"{s}_kurtosis"
            )
            if kurt_col in df_out.columns:
                kurt = row[kurt_col]
                if kurt >= 6.0:
                    timestamp_violations.append(
                        (
                            2,
                            s,
                            "Kurtosis",
                            (
                                f"{s} Severe impulsive shock / advanced spalling (Kurtosis"
                                f" {kurt:.2f} >= 6.0)"
                            ),
                        )
                    )
                elif kurt >= 3.8:
                    timestamp_violations.append(
                        (
                            1,
                            s,
                            "Kurtosis",
                            f"{s} Incipient raceway distress (Kurtosis {kurt:.2f} >= 3.8)",
                        )
                    )

            # 4. Shift Divergence Ratio (Immediate 30m vs Shift Baseline 8h)
            div_col = f"{s}_shift_divergence_ratio"
            if div_col in df_out.columns:
                div = row[div_col]
                if div >= 2.40:
                    timestamp_violations.append(
                        (
                            1,
                            s,
                            "Shift Divergence",
                            f"{s} Dynamic surge (+{(div - 1) * 100:.0f}% above 8h shift"
                            " baseline)",
                        )
                    )

        # --- Evaluate System-Wide Multi-Sensor Kinematics ---
        if "twin_branch_torque_disparity_abs" in df_out.columns:
            disp = row["twin_branch_torque_disparity_abs"]
            if disp >= 3.5:
                timestamp_violations.append(
                    (
                        2,
                        "10RV-12RV",
                        "Torque Split Disparity",
                        f"Critical pinion load asymmetry ({disp:.2f} mm/s >= 3.5)",
                    )
                )
            elif disp >= 2.0:
                timestamp_violations.append(
                    (
                        1,
                        "10RV-12RV",
                        "Torque Split Disparity",
                        f"Twin-branch torque unbalance ({disp:.2f} mm/s >= 2.0)",
                    )
                )

        # --- Triage and record violations for this timestamp ---
        if timestamp_violations:
            timestamp_violations.sort(key=lambda x: x[0], reverse=True)

            top_severity, top_sensor, top_metric, top_detail = timestamp_violations[0]

            health_states[i] = top_severity
            fault_sensors[i] = top_sensor
            fault_metrics[i] = top_metric
            fault_details[i] = top_detail
            active_violations_counts[i] = len(timestamp_violations)
            all_violations_logs[i] = " | ".join([v[3] for v in timestamp_violations])

    df_out["health_state"] = health_states
    df_out["health_state_label"] = pd.Series(health_states, index=df_out.index).map(
        {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
    )
    df_out["fault_sensor"] = fault_sensors
    df_out["fault_metric"] = fault_metrics
    df_out["fault_detail"] = fault_details
    df_out["active_violations_count"] = active_violations_counts
    df_out["all_violations_log"] = all_violations_logs

    return df_out


df_tagged = assign_iso_health_states_with_root_cause(df)

# --------------------------------------------------------------
# 3. Root-Cause Distribution Audit
# --------------------------------------------------------------
print("\n" + "=" * 85)
print("INDUSTRIAL ROOT-CAUSE BREAKDOWN: ALARMS BY SENSOR & METRIC")
print("=" * 85)

alarms_mask = df_tagged["health_state"] > 0
if alarms_mask.sum() > 0:
    sensor_fault_summary = pd.crosstab(
        df_tagged.loc[alarms_mask, "fault_sensor"],
        df_tagged.loc[alarms_mask, "health_state_label"],
        margins=True,
        margins_name="Total",
    )
    print("\n--- Alarms by Originating Sensor ---")
    print(sensor_fault_summary)

    print("\n--- Alarms by Triggering Physical Metric ---")
    metric_fault_summary = pd.crosstab(
        df_tagged.loc[alarms_mask, "fault_metric"],
        df_tagged.loc[alarms_mask, "health_state_label"],
        margins=True,
        margins_name="Total",
    )
    print(metric_fault_summary)
else:
    print("All operating records are classified as NORMAL.")

print("\n" + "-" * 85)
print("SAMPLE ROOT-CAUSE EVENT LOG (Most Recent 5 Alarms):")
print("-" * 85)
sample_alarms = df_tagged.loc[
    alarms_mask,
    [
        "health_state_label",
        "fault_sensor",
        "fault_metric",
        "fault_detail",
        "active_violations_count",
    ],
].tail(5)
print(sample_alarms.to_string())
print("=" * 85)


# --------------------------------------------------------------
# 4. Root-Cause Timeline Visualization
# --------------------------------------------------------------
def plot_root_cause_timeline(df_plot):
    """Plots operational health states alongside originating sensor identifiers."""
    alarms = df_plot[df_plot["health_state"] > 0]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 2]},
    )

    # 1. Discrete Health State Status Bar
    state_numeric = df_plot["health_state"]
    ax1.step(df_plot.index, state_numeric, where="mid", color="#34495e", linewidth=1.2)
    ax1.fill_between(
        df_plot.index, state_numeric, step="mid", color="#34495e", alpha=0.15
    )
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["NORMAL", "WARNING", "CRITICAL"], fontweight="bold")
    ax1.set_ylim(-0.3, 2.3)
    ax1.set_ylabel("Health State", fontweight="bold")
    ax1.set_title(
        "Operational Machinery Health & Fault Origin Timeline",
        fontweight="bold",
        fontsize=12,
    )

    # 2. Scatter plot indicating which sensor node faulted
    if len(alarms) > 0:
        unique_sensors = sorted(alarms["fault_sensor"].unique())
        sensor_y_map = {s: idx for idx, s in enumerate(unique_sensors)}

        colors = alarms["health_state"].map({1: "#f39c12", 2: "#c0392b"})
        y_vals = alarms["fault_sensor"].map(sensor_y_map)

        ax2.scatter(alarms.index, y_vals, c=colors, s=45, alpha=0.75, edgecolors="none")
        ax2.set_yticks(range(len(unique_sensors)))
        ax2.set_yticklabels(unique_sensors, fontweight="semibold")
        ax2.set_ylabel("Faulting Sensor Node", fontweight="bold")
        ax2.set_xlabel("Timestamp", fontweight="bold")

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Warning Node",
                markerfacecolor="#f39c12",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Critical Trip Node",
                markerfacecolor="#c0392b",
                markersize=8,
            ),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No Warning or Critical Alarms Detected",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# 5. Export Enriched Dataset
# --------------------------------------------------------------
OUTPUT_DIR = "../../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PKL_EXPORT_PATH = os.path.join(OUTPUT_DIR, "04_labeled_health_states.pkl")
CSV_EXPORT_PATH = os.path.join(OUTPUT_DIR, "04_labeled_health_states.csv")

df_tagged.to_pickle(PKL_EXPORT_PATH)
df_tagged.to_csv(CSV_EXPORT_PATH)

print("\n✔ Stage 4 Complete: Dataset with Granular Root-Cause Diagnostics exported:")
print(f"   Pickle: {os.path.abspath(PKL_EXPORT_PATH)}")
print(f"   CSV:    {os.path.abspath(CSV_EXPORT_PATH)}")
