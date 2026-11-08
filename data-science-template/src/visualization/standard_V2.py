"""
================================================================================
BALL MILL CONDITION MONITORING & AUTOMATED DIAGNOSTIC PIPELINE (HOLCIM M-PREDICT)
Asset: 2300 kW SEC YRKK2300-6W Motor (988 RPM, NU1038M Cylindrical Bearings)
Reducer: RENK #2092421 Dual-Pinion Pure Mechanical Torque-Split Reducer
Standards: ISO 20816-3 Group 1 (Rigid Foundation) | AGMA 6014-B15 (±15% Mechanical)
================================================================================
"""

import json
import math
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from scipy import signal

# ==============================================================================
# 1. CALIBRATED ENGINEERING THRESHOLDS (2300 kW RIGID FOUNDATION)
# ==============================================================================
V_GOOD = 3.5  # mm/s (ISO 20816-3 Group 1 Rigid Normal Ceiling)
V_ALERT = 4.5  # mm/s (Advisory/Alert boundary)
V_CRITICAL = 7.1  # mm/s (Danger/Hard Trip boundary)

A_GOOD = 0.25  # g (Normal continuous meshing energy floor)
A_ALERT = 0.60  # g (Active tooth pitting / raceway flaking)
A_CRITICAL = 1.00  # g (Severe multi-flank spalling explosion)

VC_G_GOOD = 2.5  # g (Normal meshing + tumbling noise floor)
VC_G_ALERT = 5.0  # g (Incipient impact shock / microcrack)
VC_G_CRITICAL = 15.0  # g (Critical shock normalization denominator)

FC_CRACK_ALERT = 5.0  # Crest Factor alert limit
FC_NORMAL = 3.5  # Gaussian continuous meshing baseline
K_CRACK_ALERT = 5.0  # Kurtosis alert limit
K_NORMAL = 3.0  # Gaussian continuous rolling baseline

TEMP_WARNING = 65.0  # °C (Lubricant degradation / inspection advisory)
TEMP_TRIP = 80.0  # °C (Hard interlock OEM emergency shutdown)

TORQUE_SPLIT_IDEAL = 1.00  # Perfect 50/50 balance
TORQUE_SPLIT_TOL_LOWER = 0.85  # -15% AGMA 6014-B15 lower tolerance limit
TORQUE_SPLIT_TOL_UPPER = 1.15  # +15% AGMA 6014-B15 upper tolerance limit
SHAFT_SKEW_LIMIT = 0.30  # g (Maximum allowable Inboard - Outboard peak delta)
BACKLASH_CHATTER_LIMIT = 2.50  # Ratio (VCC_G / VC_G rebound threshold)

SENSOR_FROZEN_STD = 0.001  # mm/s (Standard deviation freeze limit)
SENSOR_DISCONNECT_G = 49.0  # g (Transducer rail voltage saturation limit)
SENSOR_LOOSE_A = 0.05  # g (Loose mounting stud indicator)

DEGRADATION_RATE = 0.020  # Daily wear exponent for RUL extrapolation

# Telemetry Position Mapping (Active 12 Sensors)
DRIVE_TRAIN_SENSOR_LABELS = {
    "1RV": "Motor NDE (1RV)",
    "2RV": "Motor DE - Main Coupling (2RV)",
    "3RV": "Splitter Input Inboard DE (3RV)",
    "4RV": "Splitter Input Outboard NDE (4RV)",
    "5RV": "Splitter Lower Int-1 Inboard (5RV)",
    "6RV": "Splitter Lower Int-1 Outboard (6RV)",
    "9RV": "Splitter Lower Output Inboard (9RV)",
    "10RV": "Splitter Lower Output Outboard DE (10RV)",
    "7RV": "Splitter Upper Int-2 Inboard (7RV)",
    "8RV": "Splitter Upper Int-2 Outboard (8RV)",
    "11RV": "Splitter Upper Output Inboard (11RV)",
    "12RV": "Splitter Upper Output Outboard DE (12RV)",
}

ACTIVE_SENSORS = list(DRIVE_TRAIN_SENSOR_LABELS.keys())

KINEMATIC_GROUPS = {
    "Drive Motor (2300 kW)": (["1RV", "2RV"], "#1f77b4", "s"),
    "Splitter Input Shaft": (["3RV", "4RV"], "#2ca02c", "D"),
    "Upper Branch (To Top Pinion)": (
        ["7RV", "8RV", "11RV", "12RV"],
        "#d62728",
        "^",
    ),
    "Lower Branch (To Bottom Pinion)": (
        ["5RV", "6RV", "9RV", "10RV"],
        "#ff7f0e",
        "o",
    ),
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
# 2. DATA INGESTION & DYNAMIC OPERATING REGIME CLUSTERING
# ==============================================================================
def load_and_preprocess_telemetry(
    data_path="../../data/interim/01_processed.pkl",
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source telemetry missing at: {data_path}")

    df = pd.read_pickle(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df = df.sort_index()
    df_resampled = df.resample(rule="10min").mean().dropna(how="all")

    if "Motor_Power_kW" not in df_resampled.columns:
        df_resampled["Motor_Power_kW"] = np.where(
            df_resampled["1RV_Acceleration (g)"] >= 0.10, 2250.0, 0.0
        )

    # Operational Regime Tagging (Prevents Stoppage Mathematical Singularities)
    def assign_regime(row):
        v = row.get("1RV_Velocity (mm/s)", 0.0)
        a = row.get("1RV_Acceleration (g)", 0.0)
        if v < 0.5 and a < 0.05:
            return "REGIME 0: STOPPED / IDLE"
        elif row.get("Motor_Power_kW", 0.0) < 1500:
            return "REGIME 1: STARTUP / EMPTY MILL"
        else:
            return "REGIME 2: FULL PRODUCTION (2300 kW)"

    df_resampled["Operating_Regime"] = df_resampled.apply(assign_regime, axis=1)
    df_operational = df_resampled[
        df_resampled["Operating_Regime"] == "REGIME 2: FULL PRODUCTION (2300 kW)"
    ].copy()

    return df_resampled, df_operational


# ==============================================================================
# 3. 4-ZONE CROSS-DOMAIN AUTOMATED DECISION & ALERT ENGINE
# ==============================================================================
def classify_point_with_prescriptive_rul(v_val, a_val, p_val, fc_val, t_val, is_spike):
    is_v_high = v_val > V_ALERT
    is_v_trip = v_val >= V_CRITICAL
    is_a_high = a_val > A_ALERT
    is_a_trip = a_val >= A_CRITICAL
    is_p_high = p_val > VC_G_ALERT
    is_p_trip = p_val >= VC_G_CRITICAL
    is_t_warning = t_val >= TEMP_WARNING
    is_t_trip = t_val >= TEMP_TRIP

    rul_days = "> 180"

    if (
        not is_v_high
        and not is_a_high
        and not is_p_high
        and not is_spike
        and not is_t_warning
    ):
        zone = "ZONE 1.good Operating State"
        status = "NORMAL"
        failure_mode = "Continuous hydrodynamic mesh / uniform roller track"
        action = "Maintain routine lubrication and standard PM sampling"
    elif not is_v_high and (is_a_high or is_p_high or is_spike):
        zone = "ZONE 2: Incipient Surface Fatigue"
        status = "ALERT" if (is_a_high or is_p_high) else "ADVISORY"
        failure_mode = "Raceway microcracks, tooth pitting, or roller spalling"
        action = (
            "PRESCRIPTIVE DERATE: Recommend -10% mill feed to suppress dynamic"
            " shock. Draw oil ferrography."
        )
        if a_val > 0:
            calc_days = max(0.0, math.log(A_CRITICAL / a_val) / DEGRADATION_RATE)
            rul_days = f"{int(calc_days)} Days" if calc_days < 180 else "> 180"
    elif is_v_high and not is_a_high and not is_p_high and not is_spike:
        zone = "ZONE 3: Macro-Structural Defect"
        status = "CRITICAL" if is_v_trip else "ALERT"
        failure_mode = "Coupling misalignment, foundation looseness, or shaft unbalance"
        action = (
            "Check precision laser coupling alignment & foundation hold-down torque"
        )
        if v_val > 0:
            calc_days = max(0.0, math.log(V_CRITICAL / v_val) / DEGRADATION_RATE)
            rul_days = f"{int(calc_days)} Days" if calc_days < 180 else "> 180"
    else:
        zone = "ZONE 4: Terminal Breakdown"
        status = (
            "EMERGENCY TRIP"
            if (is_v_trip or is_a_trip or is_p_trip or is_t_trip)
            else "CRITICAL"
        )
        failure_mode = "Catastrophic tooth flank shear or bearing clearance collapse"
        action = (
            "MANDATORY OUTAGE: Remove load. Internal borescope inspection required."
        )
        rul_days = "0 Days (IMMEDIATE)"

    if is_t_trip:
        status = "EMERGENCY TRIP"
        action = (
            "IMMEDIATE TRIP: Temperature breached OEM Limit"
            f" ({t_val:.1f}°C ≥ {TEMP_TRIP}°C)"
        )
        rul_days = "0 Days (IMMEDIATE)"
    elif is_t_warning and status in ["NORMAL", "ADVISORY"]:
        status = "ALERT"
        action += (
            f" | NOTE: High temperature advisory ({t_val:.1f}°C ≥ {TEMP_WARNING}°C)"
        )

    return zone, status, failure_mode, action, rul_days


def evaluate_cross_domain_alerts(df_op, export_dir=None, export_all_records=True):
    diagnostic_records = []
    cmms_dispatches = []
    mean_motor_power_mw = df_op.get("Motor_Power_kW", pd.Series(2250.0)).mean() / 1000.0

    # 1. Sensor-Level Summary Evaluation
    for s in ACTIVE_SENSORS:
        pos_name = DRIVE_TRAIN_SENSOR_LABELS[s]
        v_std = df_op[f"{s}_Velocity (mm/s)"].std()
        p_max = df_op[f"{s}_VC_G (g)"].max()

        v_val = df_op[f"{s}_Velocity (mm/s)"].mean()
        a_val = df_op[f"{s}_Acceleration (g)"].mean()
        p_val = df_op[f"{s}_VC_G (g)"].mean()
        fc_val = df_op[f"{s}_FC"].mean()
        k_val = df_op[f"{s}_K Factor"].mean()
        t_val = df_op[f"{s}_Temperature (°C)"].mean()

        # Zero-Trust Plausibility Checks
        is_frozen = v_std < SENSOR_FROZEN_STD
        is_disconnect = p_max >= SENSOR_DISCONNECT_G
        is_loose = (v_val > V_ALERT) and (a_val < SENSOR_LOOSE_A)

        if is_frozen or is_disconnect or is_loose:
            zone = "INVALID DATA"
            status = "SENSOR FAULT"
            fail_mode = "Hardware fault: " + (
                "Frozen Signal"
                if is_frozen
                else ("Transducer Saturation" if is_disconnect else "Loose Sensor")
            )
            action = "Dispatch I&E Technician to inspect cable/sensor."
            rul_days = "N/A"
            mhi_score = 0.0
        else:
            alert_exposure = (df_op[f"{s}_FC"] >= FC_CRACK_ALERT).mean() * 100.0
            is_spike = (fc_val >= FC_CRACK_ALERT) or (alert_exposure >= 10.0)

            zone, status, fail_mode, action, rul_days = (
                classify_point_with_prescriptive_rul(
                    v_val, a_val, p_val, fc_val, t_val, is_spike
                )
            )

            # Safety-Hardened Machine Health Index (MHI)
            norm_v = 0.0 if pd.isna(v_val) else max(0.0, float(v_val / V_CRITICAL))
            norm_fc = 0.0 if pd.isna(fc_val) else max(0.0, float(fc_val / 6.0))
            norm_peak = (
                0.0 if pd.isna(p_val) else max(0.0, float(p_val / VC_G_CRITICAL))
            )
            norm_temp = (
                0.0
                if pd.isna(t_val)
                else max(0.0, float((t_val - 35.0) / (TEMP_TRIP - 35.0)))
            )
            alert_rate = (
                0.0
                if pd.isna(alert_exposure)
                else max(0.0, float(alert_exposure / 100.0))
            )

            if (
                0.30 * norm_v >= 1.0
                or 0.25 * norm_fc >= 1.0
                or 0.20 * norm_peak >= 1.0
                or 0.15 * norm_temp >= 1.0
                or 0.10 * alert_rate >= 1.0
            ):
                mhi_score = 100.0
            else:
                mhi_score = (
                    min(1.0, 0.30 * norm_v)
                    + min(1.0, 0.25 * norm_fc)
                    + min(1.0, 0.20 * norm_peak)
                    + min(1.0, 0.15 * norm_temp)
                    + min(1.0, 0.10 * alert_rate)
                ) * 100.0
                mhi_score = min(100.0, mhi_score)

            # CMMS Payload Formatting
            if status in ["ALERT", "CRITICAL", "EMERGENCY TRIP"]:
                cmms_dispatches.append(
                    {
                        "Equipment_ID": "BALL_MILL_01_PRIMARY_DRIVE",
                        "Asset_Description": "2300kW SEC Motor / RENK Dual-Pinion Reducer",
                        "Trigger_Sensor": s,
                        "Location": pos_name,
                        "Fault_Classification": fail_mode,
                        "Priority": "P1" if "TRIP" in status else "P2",
                        "MHI_Severity": round(mhi_score, 1),
                        "Estimated_RUL": rul_days,
                        "Action_Required": action,
                    }
                )

        specific_severity = (
            v_val / mean_motor_power_mw if mean_motor_power_mw > 0 else 0.0
        )

        diagnostic_records.append(
            {
                "Sensor": s,
                "Designation": pos_name,
                "Zone": zone,
                "Severity Status": status,
                "v_RMS (mm/s)": round(v_val, 2),
                "a_RMS (g)": round(a_val, 2),
                "Peak VC_G (g)": round(p_val, 2),
                "Crest Factor": round(fc_val, 2),
                "Kurtosis": round(k_val, 2),
                "Temp (°C)": round(t_val, 1),
                "Crack Exposure (%)": (
                    round(alert_exposure, 1) if not is_frozen else np.nan
                ),
                "Specific Severity (mm/s/MW)": round(specific_severity, 2),
                "Est. RUL": rul_days,
                "MHI Score": round(mhi_score, 1),
                "Diagnosed Failure Mode": fail_mode,
                "Prescribed Action": action,
            }
        )

    df_alerts = pd.DataFrame(diagnostic_records)

    # 2. Time-Series Record Classification for the Whole Document
    df_all_records = None
    if export_all_records:
        all_rows = []
        for ts, row in df_op.iterrows():
            for s in ACTIVE_SENSORS:
                pos_name = DRIVE_TRAIN_SENSOR_LABELS[s]
                v_inst = row.get(f"{s}_Velocity (mm/s)", 0.0)
                a_inst = row.get(f"{s}_Acceleration (g)", 0.0)
                p_inst = row.get(f"{s}_VC_G (g)", 0.0)
                fc_inst = row.get(f"{s}_FC", 0.0)
                k_inst = row.get(f"{s}_K Factor", 0.0)
                t_inst = row.get(f"{s}_Temperature (°C)", 0.0)

                is_inst_spike = fc_inst >= FC_CRACK_ALERT
                zone_i, status_i, fail_i, act_i, rul_i = (
                    classify_point_with_prescriptive_rul(
                        v_inst, a_inst, p_inst, fc_inst, t_inst, is_inst_spike
                    )
                )

                norm_v = (
                    0.0 if pd.isna(v_inst) else max(0.0, float(v_inst / V_CRITICAL))
                )
                norm_fc = 0.0 if pd.isna(fc_inst) else max(0.0, float(fc_inst / 6.0))
                norm_peak = (
                    0.0 if pd.isna(p_inst) else max(0.0, float(p_inst / VC_G_CRITICAL))
                )
                norm_temp = (
                    0.0
                    if pd.isna(t_inst)
                    else max(0.0, float((t_inst - 35.0) / (TEMP_TRIP - 35.0)))
                )
                alert_rate = 1.0 if is_inst_spike else 0.0

                if (
                    0.30 * norm_v >= 1.0
                    or 0.25 * norm_fc >= 1.0
                    or 0.20 * norm_peak >= 1.0
                    or 0.15 * norm_temp >= 1.0
                    or 0.10 * alert_rate >= 1.0
                ):
                    mhi_inst = 100.0
                else:
                    mhi_inst = (
                        min(1.0, 0.30 * norm_v)
                        + min(1.0, 0.25 * norm_fc)
                        + min(1.0, 0.20 * norm_peak)
                        + min(1.0, 0.15 * norm_temp)
                        + min(1.0, 0.10 * alert_rate)
                    ) * 100.0
                    mhi_inst = min(100.0, mhi_inst)

                all_rows.append(
                    {
                        "Timestamp": ts,
                        "Sensor": s,
                        "Designation": pos_name,
                        "Zone": zone_i,
                        "Severity Status": status_i,
                        "v_RMS (mm/s)": round(v_inst, 2),
                        "a_RMS (g)": round(a_inst, 2),
                        "Peak VC_G (g)": round(p_inst, 2),
                        "Crest Factor": round(fc_inst, 2),
                        "Kurtosis": round(k_inst, 2),
                        "Temp (°C)": round(t_inst, 1),
                        "MHI Score": round(mhi_inst, 1),
                        "Est. RUL": rul_i,
                        "Diagnosed Failure Mode": fail_i,
                        "Prescribed Action": act_i,
                    }
                )
        df_all_records = pd.DataFrame(all_rows)

    # 3. Formatted Exports
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        df_alerts.to_csv(
            os.path.join(export_dir, "automated_diagnostic_summary.csv"),
            index=False,
        )

        if df_all_records is not None:
            df_all_records.to_csv(
                os.path.join(export_dir, "automated_diagnostic_all_records.csv"),
                index=False,
            )

        if cmms_dispatches:
            with open(os.path.join(export_dir, "cmms_sap_work_orders.json"), "w") as jf:
                json.dump(cmms_dispatches, jf, indent=4)

        try:
            import openpyxl
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
            from openpyxl.utils.dataframe import dataframe_to_rows

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "4-Zone Diagnostic Summary"
            ws.append(df_alerts.columns.tolist())

            header_fill = PatternFill(
                start_color="1A365D", end_color="1A365D", fill_type="solid"
            )
            header_font = Font(color="FFFFFF", bold=True)
            thin_border = Border(
                left=Side(style="thin", color="CCCCCC"),
                right=Side(style="thin", color="CCCCCC"),
                top=Side(style="thin", color="CCCCCC"),
                bottom=Side(style="thin", color="CCCCCC"),
            )

            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(
                    horizontal="center", vertical="center", wrap_text=True
                )
                cell.border = thin_border

            status_colors = {
                "NORMAL": "C6EFCE",
                "ADVISORY": "FFEB9C",
                "ALERT": "FFC7CE",
                "CRITICAL": "FF9999",
                "EMERGENCY TRIP": "FF0000",
                "SENSOR FAULT": "E0E0E0",
            }
            status_fonts = {
                "NORMAL": Font(color="006100"),
                "ADVISORY": Font(color="9C5700"),
                "ALERT": Font(color="9C0006"),
                "CRITICAL": Font(color="9C0006", bold=True),
                "EMERGENCY TRIP": Font(color="FFFFFF", bold=True),
                "SENSOR FAULT": Font(color="4A4A4A", bold=True, italic=True),
            }

            for r_idx, row in enumerate(
                dataframe_to_rows(df_alerts, index=False, header=False), 2
            ):
                for c_idx, value in enumerate(row, 1):
                    cell = ws.cell(row=r_idx, column=c_idx, value=value)
                    cell.border = thin_border
                    cell.alignment = Alignment(
                        horizontal=(
                            "center" if isinstance(value, (int, float)) else "left"
                        ),
                        vertical="center",
                    )
                    status_val = row[3]
                    if c_idx in (3, 4) and status_val in status_colors:
                        cell.fill = PatternFill(
                            start_color=status_colors[status_val],
                            end_color=status_colors[status_val],
                            fill_type="solid",
                        )
                        cell.font = status_fonts[status_val]

            col_widths = {
                "A": 10,
                "B": 35,
                "C": 30,
                "D": 20,
                "E": 15,
                "F": 15,
                "G": 15,
                "H": 15,
                "I": 15,
                "J": 15,
                "K": 18,
                "L": 25,
                "M": 15,
                "N": 12,
                "O": 45,
                "P": 50,
            }
            for col_letter, width in col_widths.items():
                if col_letter in ws.column_dimensions:
                    ws.column_dimensions[col_letter].width = width
            ws.freeze_panes = "A2"
            wb.save(os.path.join(export_dir, "4_Zone_Diagnostic_Alerts.xlsx"))
        except ImportError:
            pass

    return df_alerts, df_all_records


# ==============================================================================
# 4. EXPERT VISUALIZATION MODULES (MODULES 1 TO 8)
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
                ax.scatter(v_data, f_data, color=color, alpha=0.15, s=15)
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

    ax.axvline(
        V_GOOD,
        color="green",
        linestyle="--",
        linewidth=1.4,
        alpha=0.7,
        label=f"ISO Group 1 Acceptable ({V_GOOD} mm/s)",
    )
    ax.axvline(
        V_ALERT,
        color="darkorange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"ISO Group 1 Alert ({V_ALERT} mm/s)",
    )
    ax.axvline(
        V_CRITICAL,
        color="red",
        linestyle="-",
        linewidth=1.6,
        alpha=0.85,
        label=f"ISO Group 1 Danger/Trip ({V_CRITICAL} mm/s)",
    )
    ax.axhline(
        FC_CRACK_ALERT,
        color="crimson",
        linestyle="-.",
        linewidth=1.6,
        alpha=0.9,
        label=f"Incipient Microcrack Strike (FC = {FC_CRACK_ALERT})",
    )
    ax.axhline(
        FC_NORMAL,
        color="green",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8,
        label=f"Healthy Continuous Mesh (FC = {FC_NORMAL})",
    )

    ax.set_title(
        "Asset Quadrant Matrix (2300 kW Rigid Support): ISO 20816-3 Severity vs."
        " Impulsive Impacts",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "RMS Vibration Velocity [mm/s] (ISO Group 1 Macro Severity)",
        fontweight="semibold",
    )
    ax.set_ylabel(
        "Crest Factor [FC] (Peak / RMS Impact Concentration)",
        fontweight="semibold",
    )
    ax.set_xlim(0, 8.5)
    ax.set_ylim(1.8, 10.5)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "01_risk_quadrant_matrix_expert.png"), dpi=140)
    plt.close(fig)


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
                "Crack Alert Rate (%)": (df_op[f"{s}_FC"] >= FC_CRACK_ALERT).mean()
                * 100,
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
        cbar_kws={"label": "Normalized Stress Intensity"},
        linewidths=1.2,
        linecolor="#e0e0e0",
        mask=df_norm.isna(),
        ax=ax,
    )
    ax.text(
        0.5,
        len(df_heat_with_delta) - 0.5,
        f"Pure Mechanical Load Differential: ΔVel={branch_delta['Velocity (mm/s)']:.2f} mm/s | ΔPeak={branch_delta['Peak Shock VC_G (g)']:.2f}g | ΔTemp={branch_delta['Temperature (°C)']:.1f}°C",
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


def plot_maintenance_priority_and_balance(df_op, export_dir):
    fig, (ax_rank, ax_gauge) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.2, 0.8]}
    )
    fig.patch.set_facecolor("white")

    mhi_scores = {}
    for s in ACTIVE_SENSORS:
        norm_v = (
            0.0
            if pd.isna(df_op[f"{s}_Velocity (mm/s)"].mean())
            else max(0.0, df_op[f"{s}_Velocity (mm/s)"].mean() / V_CRITICAL)
        )
        norm_fc = (
            0.0
            if pd.isna(df_op[f"{s}_FC"].mean())
            else max(0.0, df_op[f"{s}_FC"].mean() / 6.0)
        )
        norm_peak = (
            0.0
            if pd.isna(df_op[f"{s}_VC_G (g)"].mean())
            else max(0.0, df_op[f"{s}_VC_G (g)"].mean() / 15.0)
        )
        t_val = df_op[f"{s}_Temperature (°C)"].mean()
        norm_temp = (
            0.0 if pd.isna(t_val) else max(0.0, (t_val - 35.0) / (TEMP_TRIP - 35.0))
        )
        alert_rate = (df_op[f"{s}_FC"] >= FC_CRACK_ALERT).mean()

        if (
            0.30 * norm_v >= 1.0
            or 0.25 * norm_fc >= 1.0
            or 0.20 * norm_peak >= 1.0
            or 0.15 * norm_temp >= 1.0
            or 0.10 * alert_rate >= 1.0
        ):
            mhi_scores[DRIVE_TRAIN_SENSOR_LABELS[s]] = 100.0
        else:
            mhi_scores[DRIVE_TRAIN_SENSOR_LABELS[s]] = min(
                100.0,
                (
                    min(1.0, 0.30 * norm_v)
                    + min(1.0, 0.25 * norm_fc)
                    + min(1.0, 0.20 * norm_peak)
                    + min(1.0, 0.15 * norm_temp)
                    + min(1.0, 0.10 * alert_rate)
                )
                * 100.0,
            )

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
        "Composite Severity Score (0 = Normal, 100 = ISO Trip / 80°C Boundary)",
        fontweight="semibold",
    )
    ax_rank.set_xlim(0, 85)
    ax_rank.legend(loc="lower right")

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
        "Kinematic Velocity Ratio (Upper / Lower)", fontweight="semibold"
    )
    ax_gauge.set_title(
        "2. Pure Mechanical Torque-Split Symmetry Gauge", fontsize=11, fontweight="bold"
    )
    ax_gauge.legend(loc="upper right")
    for bar in bars_g:
        ax_gauge.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}x",
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


def plot_thermal_mechanical_phase_space(df_op, export_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    # Panel C: Motor DE Bearing Thermal Stability Envelope
    ax3 = axes[0]
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
        V_ALERT,
        color="darkorange",
        linestyle="--",
        alpha=0.8,
        label=f"ISO Group 1 Alert ({V_ALERT} mm/s)",
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
    ax4 = axes[1]
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


def plot_defect_lifecycle_trajectory(
    df_op,
    export_dir,
    target_stages=["3RV", "7RV", "5RV", "8RV", "11RV", "12RV", "9RV", "10RV"],
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor("white")
    stage_markers = {
        "3RV": "D",
        "7RV": "^",
        "5RV": "o",
        "8RV": "s",
        "11RV": "P",
        "12RV": "X",
        "9RV": "*",
        "10RV": "H",
    }
    stage_colors = {
        "3RV": "#1f77b4",
        "7RV": "#d62728",
        "5RV": "#10df9a",
        "8RV": "#d5eb0b",
        "11RV": "#ff7f0e",
        "12RV": "#9467bd",
        "9RV": "#8c564b",
        "10RV": "#e377c2",
    }

    # PANEL A: Macro Velocity vs. Crest Factor
    ax1.axvspan(
        0.0, V_GOOD, color="green", alpha=0.08, label=f"Good / Normal (≤ {V_GOOD} mm/s)"
    )
    ax1.axvspan(
        V_GOOD,
        V_ALERT,
        color="orange",
        alpha=0.12,
        label=f"Alert / Advisory ({V_GOOD} - {V_ALERT} mm/s)",
    )
    ax1.axvspan(
        V_ALERT,
        9.0,
        color="red",
        alpha=0.12,
        label=f"Critical / Danger (> {V_ALERT} mm/s)",
    )
    ax1.axvline(V_GOOD, color="green", linestyle="--", linewidth=1.4)
    ax1.axvline(V_ALERT, color="darkorange", linestyle="--", linewidth=1.4)
    ax1.axvline(
        V_CRITICAL,
        color="red",
        linestyle="-",
        linewidth=1.6,
        label=f"Hard Trip ({V_CRITICAL} mm/s)",
    )
    ax1.axhline(
        FC_CRACK_ALERT,
        color="crimson",
        linestyle="-.",
        linewidth=1.5,
        label=f"Incipient Microcrack (FC = {FC_CRACK_ALERT})",
    )
    ax1.axhline(
        FC_NORMAL,
        color="darkgreen",
        linestyle=":",
        linewidth=1.4,
        label=f"Healthy Continuous Mesh (FC = {FC_NORMAL})",
    )

    for sensor in target_stages:
        v_rms = df_op[f"{sensor}_Velocity (mm/s)"]
        fc = df_op[f"{sensor}_FC"]
        color = stage_colors.get(sensor, "#333333")
        marker = stage_markers.get(sensor, "o")

        ax1.plot(
            v_rms,
            fc,
            linestyle="-",
            linewidth=1.2,
            alpha=0.55,
            color=color,
            label=f"Trajectory Path: {sensor}",
        )
        ax1.scatter(
            v_rms,
            fc,
            color=color,
            s=25,
            alpha=0.45,
            edgecolor="none",
        )
        ax1.scatter(
            v_rms.iloc[-1],
            fc.iloc[-1],
            s=170,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label=f"Current Operating Point ({sensor})",
        )

    ax1.set_title(
        "A. High-Frequency Crest Factor Trajectory vs. Velocity Severity",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel(
        "RMS Vibration Velocity [mm/s] (ISO Severity Band)", fontweight="semibold"
    )
    ax1.set_ylabel(
        "Primary Peakedness: Crest Factor [FC] (Peak / RMS)", fontweight="semibold"
    )
    ax1.set_xlim(0, 8.5)
    ax1.set_ylim(1.8, 10.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # PANEL B: Macro Velocity vs. Kurtosis
    ax2.axvspan(0.0, V_GOOD, color="green", alpha=0.08, label=f"Good (≤ {V_GOOD} mm/s)")
    ax2.axvspan(
        V_GOOD,
        V_ALERT,
        color="orange",
        alpha=0.12,
        label=f"Alert ({V_GOOD} - {V_ALERT} mm/s)",
    )
    ax2.axvspan(
        V_ALERT, 9.0, color="red", alpha=0.12, label=f"Critical (> {V_ALERT} mm/s)"
    )
    ax2.axvline(V_GOOD, color="green", linestyle="--", linewidth=1.4)
    ax2.axvline(V_ALERT, color="darkorange", linestyle="--", linewidth=1.4)
    ax2.axvline(
        V_CRITICAL,
        color="red",
        linestyle="-",
        linewidth=1.6,
        label=f"Hard Trip ({V_CRITICAL} mm/s)",
    )
    ax2.axhline(
        K_CRACK_ALERT,
        color="crimson",
        linestyle="-.",
        linewidth=1.5,
        label=f"Elevated Kurtosis Threshold (K = {K_CRACK_ALERT})",
    )
    ax2.axhline(
        K_NORMAL,
        color="darkgreen",
        linestyle=":",
        linewidth=1.4,
        label=f"Gaussian Baseline Contact (K = {K_NORMAL})",
    )

    for sensor in target_stages:
        v_rms = df_op[f"{sensor}_Velocity (mm/s)"]
        kurt = df_op[f"{sensor}_K Factor"]
        color = stage_colors.get(sensor, "#333333")
        marker = stage_markers.get(sensor, "o")

        ax2.plot(
            v_rms,
            kurt,
            linestyle="-",
            linewidth=1.2,
            alpha=0.55,
            color=color,
            label=f"Trajectory Path: {sensor}",
        )
        ax2.scatter(
            v_rms,
            kurt,
            color=color,
            s=25,
            alpha=0.45,
            edgecolor="none",
        )
        ax2.scatter(
            v_rms.iloc[-1],
            kurt.iloc[-1],
            s=170,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label=f"Current Operating Point ({sensor})",
        )

    ax2.set_title(
        "B. Statistical Kurtosis Trajectory vs. Velocity Severity",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel(
        "RMS Vibration Velocity [mm/s] (ISO Severity Band)", fontweight="semibold"
    )
    ax2.set_ylabel("Secondary Peakedness: Kurtosis [K-Factor]", fontweight="semibold")
    ax2.set_xlim(0, 8.5)
    ax2.set_ylim(1.8, 10.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.suptitle(
        "Stewart's Defect Life-Cycle Suite: Multi-Domain Impulsive Peakedness vs. Calibrated ISO Velocity",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    os.makedirs(export_dir, exist_ok=True)
    plt.savefig(
        os.path.join(export_dir, "05_gear_defect_lifecycle_trajectory.png"), dpi=140
    )
    plt.close(fig)


def plot_shaft_deflection_matrix(df_op, export_dir):
    skews = []
    for shaft, (ib, ob) in SHAFT_ALIGNED_PAIRS.items():
        peak_diff = df_op[f"{ib}_VC_G (g)"].mean() - df_op[f"{ob}_VC_G (g)"].mean()
        temp_diff = (
            df_op[f"{ib}_Temperature (°C)"].mean()
            - df_op[f"{ob}_Temperature (°C)"].mean()
        )
        skews.append(
            {
                "Shaft Line": shaft,
                "Δ Peak Shock (g)": peak_diff,
                "Δ Temperature (°C)": temp_diff,
            }
        )
    df_skew = pd.DataFrame(skews).set_index("Shaft Line")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor("white")
    colors1 = [
        "#d62728" if abs(x) > SHAFT_SKEW_LIMIT else "#2ca02c"
        for x in df_skew["Δ Peak Shock (g)"]
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
        -0.3, 0.3, color="green", alpha=0.15, label="Parallel Contact Band (±0.3g)"
    )
    ax1.set_title(
        "1. Dynamic Shaft Deflection: Inboard vs Outboard Peak Shock",
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

    colors2 = [
        "#ff0e0e" if abs(x) > 6.0 else "#1f77b4" for x in df_skew["Δ Temperature (°C)"]
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
        -4.0, 4.0, color="blue", alpha=0.1, label="Balanced Thermal Band (±4.0°C)"
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
        label="Normal Unidirectional Contact (1.60 - 2.10)",
    )
    ax.axhspan(
        2.1,
        BACKLASH_CHATTER_LIMIT,
        color="orange",
        alpha=0.15,
        label=f"Mild Flank Separation (2.10 - {BACKLASH_CHATTER_LIMIT})",
    )
    ax.axhspan(
        BACKLASH_CHATTER_LIMIT,
        4.0,
        color="red",
        alpha=0.15,
        label=f"Severe Backlash Chattering (> {BACKLASH_CHATTER_LIMIT})",
    )
    ax.axhline(BACKLASH_CHATTER_LIMIT, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(
        "Gear Mesh Dynamic Stability: Pure Mechanical Backlash & Rebound Ratio"
        " (VCC_G / VC_G)",
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


def plot_torque_split_covariance(df_op, export_dir):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    fig.patch.set_facecolor("white")

    v10 = df_op["10RV_Velocity (mm/s)"]
    v12 = df_op["12RV_Velocity (mm/s)"]
    fc12 = df_op["12RV_FC"]

    e_split = v12 - v10
    mean_v = 0.5 * (v10 + v12)

    # Normalized Mean Squared Error of Balance
    mse_balance = np.mean(e_split**2)
    var_norm = np.std(v10) * np.std(v12)
    nmse_balance = mse_balance / (var_norm + 1e-9)

    # Phase Coherence Pearson r & Cross-Correlation Lag
    pearson_r = np.corrcoef(v10, v12)[0, 1] if len(v10) > 1 else 1.0
    try:
        corr = signal.correlate(v12 - np.mean(v12), v10 - np.mean(v10), mode="full")
        lags = signal.correlation_lags(len(v12), len(v10))
        optimal_lag = lags[np.argmax(corr)]
    except Exception:
        optimal_lag = 0

    max_err = np.max(np.abs(e_split))
    mean_ratio = np.mean(v12 / (v10 + 1e-6))

    # Panel A: Co-Variance Scatter & 95% Confidence Ellipse
    max_val = max(v10.max(), v12.max()) * 1.15
    ax1.plot(
        [0, max_val],
        [0, max_val],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal 1:1 Balance (y = x)",
    )
    ax1.plot(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_UPPER],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"+15% AGMA Upper ({TORQUE_SPLIT_TOL_UPPER}x)",
    )
    ax1.plot(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_LOWER],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"-15% AGMA Lower ({TORQUE_SPLIT_TOL_LOWER}x)",
    )
    ax1.fill_between(
        [0, max_val],
        [0, max_val * TORQUE_SPLIT_TOL_LOWER],
        [0, max_val * TORQUE_SPLIT_TOL_UPPER],
        color="green",
        alpha=0.08,
    )

    sc = ax1.scatter(
        v10,
        v12,
        c=fc12,
        cmap="plasma",
        s=35,
        alpha=0.7,
        edgecolor="none",
        zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax1)
    cb.set_label("Upper Output Crest Factor [FC_12RV]", fontweight="semibold")

    if len(v10) > 5 and np.std(v10) > 1e-6 and np.std(v12) > 1e-6:
        cov = np.cov(v10, v12)
        pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-9)
        ell_radius_x = np.sqrt(max(0.0, 1 + pearson))
        ell_radius_y = np.sqrt(max(0.0, 1 - pearson))
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor="none",
            edgecolor="blue",
            linestyle="--",
            linewidth=1.8,
            label="95% Covariance Ellipse (Phase Orbit)",
        )
        scale_x = np.sqrt(cov[0, 0]) * 2.0
        scale_y = np.sqrt(cov[1, 1]) * 2.0
        mean_x, mean_y = np.mean(v10), np.mean(v12)
        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transf + ax1.transData)
        ax1.add_patch(ellipse)

    badge_text = "\n".join(
        [
            f"mean_ratio(12RV /10RV):{mean_ratio: .4f}",
            f"phase_lag: {optimal_lag} samples",
            f"Max |e_Split|: {max_err:.4f} mm/s",
        ]
    )
    ax1.text(
        0.04,
        0.96,
        badge_text,
        transform=ax1.transAxes,
        fontsize=9,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.2, alpha=0.9),
    )

    ax1.set_title(
        "A. Torque-Split Co-Variance & Phase Lag Ellipse",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Lower Branch Output Velocity [10RV] (mm/s)", fontweight="semibold")
    ax1.set_ylabel("Upper Branch Output Velocity [12RV] (mm/s)", fontweight="semibold")
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Dynamic Torque-Split Error Trajectory
    ax2.plot(
        df_op.index,
        e_split,
        color="#1f77b4",
        linewidth=1.3,
        label="e_Split(t) = v_12RV(t) - v_10RV(t)",
    )
    ax2.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1.2,
        label="Zero Mismatch Baseline (e = 0)",
    )

    rolling_mean_v = mean_v.rolling(window=12, min_periods=1).mean()
    upper_tol = 1 * rolling_mean_v / rolling_mean_v
    lower_tol = -1 * rolling_mean_v / rolling_mean_v
    ax2.plot(
        df_op.index,
        upper_tol,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="AGMA +1 Upper Limit",
    )
    ax2.plot(
        df_op.index,
        lower_tol,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="AGMA -1 Lower Limit",
    )
    ax2.fill_between(df_op.index, lower_tol, upper_tol, color="green", alpha=0.08)

    ax2.set_title(
        "B. Dynamic Torque-Split Error Trajectory: e_Split(t)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Operational Timestamp", fontweight="semibold")
    ax2.set_ylabel("Velocity Mismatch e_Split [mm/s]", fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.suptitle(
        "Module 8: Dual-Pinion Torque-Split Co-Variance, Phase Lag & Dynamic"
        " Balance Error",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "08_torque_split_covariance.png"), dpi=140)
    plt.close(fig)


# ==============================================================================
# 5. PIPELINE EXECUTION ORCHESTRATOR (SILENT RUN)
# ==============================================================================
def main():
    export_dir = "../../reports/figures/expert_analysis"
    os.makedirs(export_dir, exist_ok=True)
    data_source_path = "../../data/interim/01_processed.pkl"

    try:
        df_resampled, df_operational = load_and_preprocess_telemetry(data_source_path)
    except FileNotFoundError:
        return

    # 1. Execute Automated 4-Zone Cross-Domain Alert Engine
    df_alerts, df_all_records = evaluate_cross_domain_alerts(
        df_operational, export_dir=export_dir, export_all_records=True
    )

    # 2. Render Full Suite of 8 Calibrated Expert Visualizations
    plot_risk_quadrant_matrix(df_operational, export_dir)
    plot_bifurcated_heatmap(df_operational, export_dir)
    plot_maintenance_priority_and_balance(df_operational, export_dir)
    plot_thermal_mechanical_phase_space(df_operational, export_dir)
    plot_defect_lifecycle_trajectory(df_operational, export_dir)
    plot_shaft_deflection_matrix(df_operational, export_dir)
    plot_backlash_chatter_index(df_operational, export_dir)
    plot_torque_split_covariance(df_operational, export_dir)


if __name__ == "__main__":
    main()
