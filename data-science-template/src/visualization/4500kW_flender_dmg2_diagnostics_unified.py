"""
================================================================================
BALL MILL CONDITION MONITORING & EXPERT PREDICTIVE PIPELINE (FLENDER DMG2 22)
Asset: 4500 kW Cantarey Reinosa SL 800 - KA6 Motor (995 RPM, 6 kV, 513 A)
Reducer: FLENDER DMG2 22 Heavy-Duty Helical Gearbox (4150 kW, i = 6.779:1)
Input Speed: 990 RPM | Output Speed: 146.04 RPM | Lube: ISO VG 460 Mineral Oil
Standards: ISO 20816-3 Group 1 (Rigid Foundation) | AGMA 6000-B96 | API 670
Architecture: Single-Path 4-Stage Reduction with 11-Channel Telemetry Grid
================================================================================
"""

import os
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# 1. CALIBRATED ENGINEERING THRESHOLDS (4500 kW RIGID FOUNDATION)
# ==============================================================================
# ISO 20816-3 Group 1 Rigid Support (Macro Velocity Thresholds)
V_GOOD = 3.5  # mm/s (Zone A/B Good/Normal operational ceiling)
V_ALERT = 4.5  # mm/s (Zone B/C Advisory/Alert boundary)
V_CRITICAL = 7.1  # mm/s (Zone C/D Danger/Trip boundary)

# Continuous High-Frequency Energy (RMS Acceleration Thresholds - ISO 20816-8 / AGMA)
A_GOOD = 0.25  # g (Normal continuous meshing energy floor)
A_ALERT = 0.60  # g (Active tooth pitting / raceway flaking)
A_CRITICAL = 1.00  # g (Severe multi-flank spalling explosion)

# True Peak Shock Acceleration (VC_G Impact Metric)
VC_G_GOOD = 2.5  # g (Normal meshing + ball charge tumbling noise floor)
VC_G_ALERT = 5.0  # g (Incipient impact shock / tooth root microcrack)
VC_G_CRITICAL = 15.0  # g (Critical shock saturation / ISO 7.1 mm/s equivalent at GMF)

# Impulsive Peakedness & Surface Microcrack Thresholds
FC_CRACK_ALERT = 5.0  # Crest Factor alert limit
FC_NORMAL = 3.5  # Gaussian continuous meshing baseline
K_CRACK_ALERT = 5.0  # Kurtosis alert limit
K_NORMAL = 3.0  # Gaussian continuous rolling baseline

# Thermal Operating Limits (FLENDER BA 4218309-040 - ISO VG 460 Mineral Oil)
TEMP_WARNING = 65.0  # °C (Lubricant oxidation / film thinning advisory)
TEMP_TRIP = 75.0  # °C (OEM Interlock emergency shutdown limit)

# Kinematic Mechanical Constraints
SKEW_PEAK_NORMAL = 0.20  # g (Parallel contact band: [-0.20g, +0.20g])
SKEW_PEAK_ALERT = 0.30  # g (AGMA KH_beta > 1.35 edge-loading risk threshold)
SKEW_PEAK_TRIP = 0.50  # g (Critical casing cocking / structural tilt)

SKEW_TEMP_NORMAL = 2.0  # °C (Balanced thermal dissipation band: [-2.0°C, +2.0°C])
SKEW_TEMP_ALERT = 4.0  # °C (Bearing load concentration advisory)
SKEW_TEMP_TRIP = 8.0  # °C (Severe thrust overload / EHD film collapse)

# Dynamic Chattering Index (R_Chatter = VCC_G / VC_G)
R_CHATTER_NORMAL = 2.10  # Stable unidirectional contact ceiling (1.60 - 2.10)
R_CHATTER_ALERT = 2.50  # Active backlash chattering / tooth rattle threshold

# Sensor Plausibility Limits ("Zero-Trust" Telemetry Ingestion)
SENSOR_FROZEN_STD = 0.001  # mm/s (Flatline standard deviation)
SENSOR_DISCONNECT_G = 49.0  # g (Transducer rail voltage saturation)
SENSOR_LOOSE_A = 0.05  # g (Abnormally low high-frequency content)

# Machine Nameplate Baselines (Cantarey Reinosa SL 800 - KA6 & Flender DMG2 22)
RATED_MOTOR_POWER_KW = 4500.0  # 4.5 MW
RATED_GEARBOX_POWER_KW = 4150.0  # 4.15 MW
MOTOR_RATED_RPM = 995.0
GEARBOX_INPUT_RPM = 990.0
GEARBOX_OUTPUT_RPM = 146.04
GEARBOX_RATIO = 6.779
DEGRADATION_RATE = 0.035  # Daily exponential wear exponent for RUL

# Telemetry Position Mapping (11 Active Channels on Single-Path Grid)
DRIVE_TRAIN_SENSOR_LABELS = {
    "1RV": "Motor NDE Bearing (1RV)",
    "2RV": "Motor DE Bearing [Coupled] (2RV)",
    "3RV": "Gearbox Input Inboard DE (3RV)",
    "4RV": "Gearbox Input Outboard NDE (4RV)",
    "5RV": "Intermediate-1 Inboard (5RV)",
    "6RV": "Intermediate-1 Outboard (6RV)",
    "7RV": "Intermediate-2 Inboard (7RV)",
    "8RV": "Intermediate-2 Outboard (8RV)",
    "9RV": "Output Shaft Inboard (9RV)",
    "10RV": "Output Shaft Outboard (10RV)",
    "11RO": "Top Casing Radial-Axial / Sump Channel (11RO)",
}

ACTIVE_SENSORS = list(DRIVE_TRAIN_SENSOR_LABELS.keys())

# Channel Subsets based on physical RTD installations
CHANNELS_WITH_TEMP = ["2RV", "4RV", "5RV", "6RV", "7RV", "11RO"]
CHANNELS_NO_TEMP = ["1RV", "3RV", "8RV", "9RV", "10RV"]

# Kinematic Power Path Sequence (Motor -> Input -> Int-1 -> Int-2 -> Output)
KINEMATIC_CASCADE_SEQUENCE = [
    ("1RV", "1. Motor NDE (1RV)"),
    ("2RV", "2. Motor DE [Coupled] (2RV)"),
    ("3RV", "3. Input Inboard DE (3RV)"),
    ("4RV", "3. Input Outboard NDE (4RV)"),
    ("5RV", "4. Int-1 Inboard (5RV)"),
    ("6RV", "4. Int-1 Outboard (6RV)"),
    ("7RV", "5. Int-2 Inboard (7RV)"),
    ("8RV", "5. Int-2 Outboard (8RV)"),
    ("9RV", "6. Output Inboard (9RV)"),
    ("10RV", "6. Output Outboard (10RV)"),
    ("11RO", "7. Top Casing / Axial (11RO)"),
]

# Parallel Shaft Lines for Dynamic Skew & Alignment Audit
SHAFT_ALIGNED_PAIRS = {
    "Input Shaft Line (~990 RPM)": ("3RV", "4RV"),
    "Intermediate-1 Shaft Line": ("5RV", "6RV"),
    "Intermediate-2 Shaft Line": ("7RV", "8RV"),
    "Low-Speed Output Line (~146 RPM)": ("9RV", "10RV"),
}

KINEMATIC_GROUPS = {
    "Cantarey Motor (4500 kW)": (["1RV", "2RV"], "#1f77b4", "s"),
    "Flender Input Shaft": (["3RV", "4RV"], "#2ca02c", "D"),
    "Intermediate Stages": (["5RV", "6RV", "7RV", "8RV"], "#ff7f0e", "^"),
    "Output Stage & Casing": (["9RV", "10RV", "11RO"], "#d62728", "o"),
}

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
# 2. DATA INGESTION & DYNAMIC OPERATING REGIME CLUSTERING
# ==============================================================================
def load_and_preprocess_telemetry(data_path):
    """
    Ingests 4.5 MW Cantarey / Flender DMG2 22 mill telemetry.
    Applies M-Predict Zero-Trust data protocols, sensor health auditing,
    and Dynamic Operating Regime Clustering without requiring electrical power.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source telemetry missing at: {data_path}")

    if data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported data format. Provide .pkl, .csv, or .xlsx file.")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
    df = df.sort_index()

    # Resample to 10-minute industrial standard
    df_resampled = df.resample(rule="10min").mean().dropna(how="all")

    # 1. Zero-Trust Sensor Plausibility Validation
    sensor_health_flags = {}
    for s in ACTIVE_SENSORS:
        acc_col = f"{s}_Acceleration (g)"
        vel_col = f"{s}_Velocity (mm/s)"
        if acc_col in df_resampled.columns and vel_col in df_resampled.columns:
            std_dev = df_resampled[vel_col].std()
            max_p = df_resampled.get(f"{s}_VC_G (g)", df_resampled[acc_col]).max()
            if std_dev < SENSOR_FROZEN_STD:
                sensor_health_flags[s] = "FAULT: SENSOR_FROZEN"
            elif max_p >= SENSOR_DISCONNECT_G:
                sensor_health_flags[s] = "FAULT: ADC_SATURATION / DISCONNECTED"
            else:
                sensor_health_flags[s] = "HEALTHY"
        else:
            sensor_health_flags[s] = "NOT_INSTALLED"

    # 2. Dynamic Operating Regime Clustering (Kinetic Gating)
    def assign_regime(row):
        a_1rv = row.get("1RV_Acceleration (g)", 0.0)
        v_1rv = row.get("1RV_Velocity (mm/s)", 0.0)
        v_2rv = row.get("2RV_Velocity (mm/s)", 0.0)

        if a_1rv < 0.01 and v_1rv < 0.03:
            return "REGIME 0: STOPPED / INCHING"
        elif a_1rv < 0.012 or v_1rv < 0.08:
            return "REGIME 1: STARTUP / TRANSIENT"
        else:
            return "REGIME 2: FULL PRODUCTION (4.5 MW)"

    df_resampled["Operating_Regime"] = df_resampled.apply(assign_regime, axis=1)

    # Isolate active steady-state grinding for fatigue evaluation
    df_operational = df_resampled[
        df_resampled["Operating_Regime"] == "REGIME 2: FULL PRODUCTION (4.5 MW)"
    ].copy()

    return df_resampled, df_operational, sensor_health_flags


def calculate_expert_mhi(v_val, fc_val, p_val, t_val, alert_exposure, sensor_name):
    """
    Computes MHI on a true 0-100 scale.
    If temperature is N/A, the 15% thermal weight is redistributed
    proportionally across the 4 mechanical vibration terms.
    """
    norm_v = 0.0 if pd.isna(v_val) else max(0.0, float(v_val / V_CRITICAL))
    norm_fc = 0.0 if pd.isna(fc_val) else max(0.0, float(fc_val / 6.0))
    norm_peak = 0.0 if pd.isna(p_val) else max(0.0, float(p_val / VC_G_CRITICAL))
    alert_rate = (
        0.0 if pd.isna(alert_exposure) else max(0.0, float(alert_exposure / 100.0))
    )

    has_temp = (sensor_name in CHANNELS_WITH_TEMP) and (not pd.isna(t_val))

    if has_temp:
        norm_temp = max(0.0, float((t_val - 35.0) / (TEMP_TRIP - 35.0)))
        w_v, w_fc, w_pk, w_temp, w_alt = 0.30, 0.25, 0.20, 0.15, 0.10
    else:
        norm_temp = 0.0
        # Re-scale weights over 0.85 total mechanical allocation
        scale = 1.0 / 0.85
        w_v = 0.30 * scale
        w_fc = 0.25 * scale
        w_pk = 0.20 * scale
        w_temp = 0.00
        w_alt = 0.10 * scale

    if (
        w_v * norm_v >= 1.0
        or w_fc * norm_fc >= 1.0
        or w_pk * norm_peak >= 1.0
        or (w_temp * norm_temp >= 1.0 if has_temp else False)
        or w_alt * alert_rate >= 1.0
    ):
        return 100.0
    else:
        score = (
            min(1.0, w_v * norm_v)
            + min(1.0, w_fc * norm_fc)
            + min(1.0, w_pk * norm_peak)
            + min(1.0, w_temp * norm_temp)
            + min(1.0, w_alt * alert_rate)
        ) * 100.0
        return min(100.0, score)


# ==============================================================================
# 3. 4-ZONE CROSS-DOMAIN M-PREDICT DECISION & CMMS ENGINE
# ==============================================================================
def evaluate_cross_domain_alerts(
    df_op, sensor_health_flags, export_dir=None, export_all_records=True
):
    """
    Automated Decision Logic Engine with Prescriptive RUL and CMMS formatting.
    Calibrated for the single-path 4500 kW / Flender DMG2 22 mill drive.
    """
    diagnostic_records = []
    cmms_dispatch_payloads = []

    def estimate_rul_days(a_val, v_val):
        if a_val < A_GOOD and v_val < V_GOOD:
            return "> 90"
        elif a_val >= A_CRITICAL or v_val >= V_CRITICAL:
            return "0 (Immediate Outage)"
        else:
            calc_rul = min(
                np.log(A_CRITICAL / max(a_val, 1e-4)) / DEGRADATION_RATE
                if a_val > 0
                else 90,
                np.log(V_CRITICAL / max(v_val, 1e-4)) / DEGRADATION_RATE
                if v_val > 0
                else 90,
            )
            return f"{max(1, int(calc_rul))} Days" if calc_rul < 90 else "> 90"

    for s in ACTIVE_SENSORS:
        pos_name = DRIVE_TRAIN_SENSOR_LABELS[s]

        if sensor_health_flags.get(s) != "HEALTHY":
            diagnostic_records.append(
                {
                    "Sensor": s,
                    "Designation": pos_name,
                    "Zone": "SENSOR FAULT",
                    "Severity Status": "MAINTENANCE REQUIRED",
                    "v_RMS (mm/s)": np.nan,
                    "a_RMS (g)": np.nan,
                    "Peak VC_G (g)": np.nan,
                    "Crest Factor": np.nan,
                    "Kurtosis": np.nan,
                    "Temp (°C)": np.nan,
                    "Crack Exposure (%)": np.nan,
                    "Relative Severity (v/v_Good)": np.nan,
                    "MHI Score": np.nan,
                    "Est. RUL": "N/A",
                    "Diagnosed Failure Mode": sensor_health_flags.get(
                        s, "SENSOR FAULT"
                    ),
                    "Prescribed Action": "Dispatch I&E Technician: inspect accelerometer stud and cabling.",
                }
            )
            continue

        v_val = (
            df_op[f"{s}_Velocity (mm/s)"].mean()
            if f"{s}_Velocity (mm/s)" in df_op.columns
            else 0.0
        )
        a_val = (
            df_op[f"{s}_Acceleration (g)"].mean()
            if f"{s}_Acceleration (g)" in df_op.columns
            else 0.0
        )
        p_val = (
            df_op[f"{s}_VC_G (g)"].mean() if f"{s}_VC_G (g)" in df_op.columns else 0.0
        )
        fc_val = df_op[f"{s}_FC"].mean() if f"{s}_FC" in df_op.columns else 0.0
        k_val = (
            df_op[f"{s}_K Factor"].mean() if f"{s}_K Factor" in df_op.columns else 3.0
        )

        # Temp handling for uninstrumented channels
        if s in CHANNELS_WITH_TEMP and f"{s}_Temperature (°C)" in df_op.columns:
            t_val = df_op[f"{s}_Temperature (°C)"].mean()
        else:
            t_val = np.nan

        alert_exposure = (
            (df_op[f"{s}_FC"] >= FC_CRACK_ALERT).mean() * 100
            if f"{s}_FC" in df_op.columns
            else 0.0
        )
        is_spike = (fc_val >= FC_CRACK_ALERT) or (alert_exposure >= 10.0)

        # 4-Zone Cross-Domain Logic Evaluation
        is_v_high = v_val > V_ALERT
        is_v_trip = v_val >= V_CRITICAL
        is_a_high = a_val > A_ALERT
        is_a_trip = a_val >= A_CRITICAL
        is_p_high = p_val > VC_G_ALERT
        is_p_trip = p_val >= VC_G_CRITICAL
        is_t_warning = t_val >= TEMP_WARNING if not pd.isna(t_val) else False
        is_t_trip = t_val >= TEMP_TRIP if not pd.isna(t_val) else False

        rul_days = estimate_rul_days(a_val, v_val)
        prescriptive_derate = "None required."

        if not is_v_high and not is_a_high and not is_p_high and not is_spike:
            zone = "ZONE 1.good Operating State"
            status = "NORMAL"
            failure_mode = "Continuous hydrodynamic mesh / uniform roller track"
            action = "Maintain routine lubrication and standard PM sampling"
        elif not is_v_high and (is_a_high or is_p_high or is_spike):
            zone = "ZONE 2: Incipient Surface Fatigue"
            status = "ALERT" if (is_a_high or is_p_high) else "ADVISORY"
            failure_mode = "Raceway microcracks, tooth pitting, or roller spalling"
            action = "PRESCRIPTIVE DERATE: Recommend -10% mill feed to suppress dynamic shock. Draw oil ferrography."
            prescriptive_derate = "-10% Feed Derate recommended."
        elif is_v_high and not is_a_high and not is_p_high and not is_spike:
            zone = "ZONE 3: Macro-Structural Defect"
            status = "CRITICAL" if is_v_trip else "ALERT"
            failure_mode = (
                "Coupling misalignment, foundation looseness, or shaft unbalance"
            )
            action = (
                "Check precision laser coupling alignment & foundation hold-down torque"
            )
            prescriptive_derate = "Re-align coupling at next opportunity."
        else:
            zone = "ZONE 4: Terminal Breakdown"
            status = (
                "EMERGENCY TRIP"
                if (is_v_trip or is_a_trip or is_p_trip or is_t_trip)
                else "CRITICAL"
            )
            failure_mode = (
                "Catastrophic tooth flank shear or bearing clearance collapse"
            )
            action = (
                "MANDATORY OUTAGE: Remove load. Internal borescope inspection required."
            )
            prescriptive_derate = "Immediate shutdown required."

        if is_t_trip:
            status = "EMERGENCY TRIP"
            action = f"IMMEDIATE TRIP: Temperature breached OEM Limit ({t_val:.1f}°C ≥ {TEMP_TRIP}°C)"
            rul_days = "0 (Immediate Outage)"
        elif is_t_warning and status in ["NORMAL", "ADVISORY"]:
            status = "ALERT"
            action += (
                f" | NOTE: High temperature advisory ({t_val:.1f}°C ≥ {TEMP_WARNING}°C)"
            )

        # Safety-Hardened Composite Machine Health Index (MHI)
        mhi_score = calculate_expert_mhi(v_val, fc_val, p_val, t_val, alert_exposure, s)

        # Replace specific electrical severity with relative velocity severity
        rel_severity = (v_val / V_GOOD) if V_GOOD > 0 else 0.0

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
                "Temp (°C)": round(t_val, 1) if not pd.isna(t_val) else "N/A",
                "Crack Exposure (%)": round(alert_exposure, 1),
                "Relative Severity (v/v_Good)": round(rel_severity, 2),
                "MHI Score": round(mhi_score, 1),
                "Est. RUL": rul_days,
                "Diagnosed Failure Mode": failure_mode,
                "Prescribed Action": action,
            }
        )

        if status in ["CRITICAL", "EMERGENCY TRIP", "ALERT"]:
            cmms_dispatch_payloads.append(
                {
                    "Work_Order_Trigger": f"{datetime.now().strftime('%Y-%m-%d %H:%M')}_{s}",
                    "Equipment_ID": "BALL_MILL_02_PRIMARY_DRIVE",
                    "Asset_Description": "4500kW Cantarey SL800 Motor / FLENDER DMG2 22 Reducer",
                    "Trigger_Sensor": s,
                    "Position": pos_name,
                    "Fault_Classification": failure_mode,
                    "Priority_Level": "P1" if "TRIP" in status else "P2",
                    "Severity_Score_MHI": round(mhi_score, 1),
                    "Est_RUL": rul_days,
                    "Prescriptive_Derate": prescriptive_derate,
                    "Required_Spare_Parts": [
                        {"Description": f"Replacement Bearings/Seals for {pos_name}"}
                    ],
                    "Prescribed_Field_Action": action,
                }
            )

    df_alerts = pd.DataFrame(diagnostic_records)

    # Export Handlers
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        summary_csv_path = os.path.join(
            export_dir, "flender_dmg2_diagnostic_summary.csv"
        )
        df_alerts.to_csv(summary_csv_path, index=False)

        if cmms_dispatch_payloads:
            cmms_json_path = os.path.join(export_dir, "cmms_sap_dispatch_flender.json")
            with open(cmms_json_path, "w") as f:
                json.dump(cmms_dispatch_payloads, f, indent=4)

        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils.dataframe import dataframe_to_rows

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Flender DMG2 Diagnostic Summary"
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
                "MAINTENANCE REQUIRED": "E0E0E0",
            }
            status_fonts = {
                "NORMAL": Font(color="006100"),
                "ADVISORY": Font(color="9C5700"),
                "ALERT": Font(color="9C0006"),
                "CRITICAL": Font(color="9C0006", bold=True),
                "EMERGENCY TRIP": Font(color="FFFFFF", bold=True),
                "MAINTENANCE REQUIRED": Font(color="000000", italic=True),
            }

            for r_idx, row in enumerate(
                dataframe_to_rows(df_alerts, index=False, header=False), 2
            ):
                for c_idx, value in enumerate(row, 1):
                    cell = ws.cell(row=r_idx, column=c_idx, value=value)
                    cell.border = thin_border
                    cell.alignment = Alignment(
                        horizontal="center"
                        if isinstance(value, (int, float))
                        else "left",
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
                "M": 12,
                "N": 15,
                "O": 45,
                "P": 50,
            }
            for col_letter, width in col_widths.items():
                if col_letter in ws.column_dimensions:
                    ws.column_dimensions[col_letter].width = width
            ws.freeze_panes = "A2"
            wb.save(os.path.join(export_dir, "Flender_DMG2_Diagnostic_Alerts.xlsx"))
        except ImportError:
            pass

    # ROW-LEVEL EXPORT
    df_all_records = None
    if export_all_records:
        all_records = []
        for ts, row in df_op.iterrows():
            for s in ACTIVE_SENSORS:
                if sensor_health_flags.get(s) != "HEALTHY":
                    continue

                v_inst = row.get(f"{s}_Velocity (mm/s)", 0.0)
                a_inst = row.get(f"{s}_Acceleration (g)", 0.0)
                p_inst = row.get(f"{s}_VC_G (g)", 0.0)
                fc_inst = row.get(f"{s}_FC", 0.0)

                is_inst_spike = fc_inst >= FC_CRACK_ALERT

                t_inst = (
                    row.get(f"{s}_Temperature (°C)", np.nan)
                    if s in CHANNELS_WITH_TEMP
                    else np.nan
                )

                mhi_inst = calculate_expert_mhi(
                    v_inst, fc_inst, p_inst, t_inst, 100.0 if is_inst_spike else 0.0, s
                )

                all_records.append(
                    {
                        "Timestamp": ts,
                        "Sensor": s,
                        "v_RMS (mm/s)": round(v_inst, 2),
                        "a_RMS (g)": round(a_inst, 2),
                        "Peak VC_G (g)": round(p_inst, 2),
                        "Crest Factor": round(fc_inst, 2),
                        "Temp (°C)": round(t_inst, 1) if not pd.isna(t_inst) else "N/A",
                        "MHI Score": round(mhi_inst, 1),
                    }
                )

        df_all_records = pd.DataFrame(all_records)
        if export_dir:
            df_all_records.to_csv(
                os.path.join(export_dir, "automated_diagnostic_all_records.csv"),
                index=False,
            )

    return df_alerts, df_all_records


# ==============================================================================
# 4. EXPERT VISUALIZATION SUITE (SINGLE-PATH 4-STAGE REDUCTION)
# ==============================================================================


# Module 1: Asset Risk Quadrant Matrix
def plot_risk_quadrant_matrix(df_op, export_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    for group_name, (sensors, color, marker) in KINEMATIC_GROUPS.items():
        first = True
        for s in sensors:
            vel_col, fc_col = f"{s}_Velocity (mm/s)", f"{s}_FC"
            if vel_col in df_op.columns and fc_col in df_op.columns:
                v_data, f_data = df_op[vel_col], df_op[fc_col]
                v_mean, f_mean = v_data.mean(), f_data.mean()

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
        "Asset Quadrant Matrix (4500 kW Rigid Support): ISO 20816-3 Severity vs. Impulsive Impacts",
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
    plt.savefig(
        os.path.join(export_dir, "01_risk_quadrant_matrix_flender.png"), dpi=140
    )
    plt.close(fig)


# Module 2: Multi-Stage Kinematic Cascade
def plot_multi_stage_kinematic_cascade(df_op, export_dir):
    heatmap_rows = []
    for s, label in KINEMATIC_CASCADE_SEQUENCE:
        if f"{s}_Velocity (mm/s)" in df_op.columns:
            temp_val = (
                df_op[f"{s}_Temperature (°C)"].mean()
                if s in CHANNELS_WITH_TEMP and f"{s}_Temperature (°C)" in df_op.columns
                else np.nan
            )
            heatmap_rows.append(
                {
                    "Component Stage": label,
                    "Velocity (mm/s)": df_op[f"{s}_Velocity (mm/s)"].mean(),
                    "Acceleration (g)": df_op[f"{s}_Acceleration (g)"].mean(),
                    "Peak Shock VC_G (g)": df_op[f"{s}_VC_G (g)"].mean()
                    if f"{s}_VC_G (g)" in df_op.columns
                    else 0.0,
                    "Crest Factor (FC)": df_op[f"{s}_FC"].mean()
                    if f"{s}_FC" in df_op.columns
                    else 0.0,
                    "Temperature (°C)": temp_val,
                }
            )

    df_cascade = pd.DataFrame(heatmap_rows).set_index("Component Stage")

    # We create custom annotations to show "N/A" where temperature is missing
    annot_array = np.empty(df_cascade.shape, dtype=object)
    for i in range(df_cascade.shape[0]):
        for j in range(df_cascade.shape[1]):
            val = df_cascade.iloc[i, j]
            if pd.isna(val):
                annot_array[i, j] = "N/A"
            else:
                annot_array[i, j] = f"{val:.2f}"

    # Fill NA with -1 for norm mapping so they appear as a blank/grey color
    df_filled = df_cascade.fillna(-1)
    df_norm = (df_filled - df_filled.min()) / (df_filled.max() - df_filled.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    sns.heatmap(
        df_norm,
        annot=annot_array,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Stress Intensity"},
        linewidths=1.2,
        linecolor="#e0e0e0",
        mask=df_cascade.isna(),
        ax=ax,
    )

    ax.set_title(
        "FLENDER DMG2 22 Multi-Stage Kinematic Cascade Heatmap: 4500 kW Power Transmission Flow",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "02_multi_stage_kinematic_cascade.png"), dpi=140
    )
    plt.close(fig)


# Module 3: Maintenance Priority Ranking & Coupling Dynamics
def plot_maintenance_priority_and_coupling(df_op, export_dir):
    fig, (ax_rank, ax_couple) = plt.subplots(
        1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.1, 0.9]}
    )
    fig.patch.set_facecolor("white")

    # Panel 1: Composite MHI Ranking
    mhi_scores = {}
    for s in ACTIVE_SENSORS:
        if f"{s}_Velocity (mm/s)" in df_op.columns:
            v_val = df_op[f"{s}_Velocity (mm/s)"].mean()
            fc_val = df_op.get(f"{s}_FC", pd.Series(0.0)).mean()
            p_val = df_op.get(f"{s}_VC_G (g)", pd.Series(0.0)).mean()
            alert_val = (
                (df_op[f"{s}_FC"] >= FC_CRACK_ALERT).mean() * 100
                if f"{s}_FC" in df_op.columns
                else 0.0
            )

            t_val = (
                df_op[f"{s}_Temperature (°C)"].mean()
                if s in CHANNELS_WITH_TEMP and f"{s}_Temperature (°C)" in df_op.columns
                else np.nan
            )

            score = calculate_expert_mhi(v_val, fc_val, p_val, t_val, alert_val, s)
            mhi_scores[DRIVE_TRAIN_SENSOR_LABELS[s]] = score

    df_mhi = pd.Series(mhi_scores).sort_values(ascending=True)
    colors_mhi = [
        "#d62728" if score >= 45 else ("#ff7f0e" if score >= 25 else "#2ca02c")
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
            if w >= 45
            else ("Field Inspection" if w >= 25 else "Routine Maintenance")
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
        45,
        color="red",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-1 Alarm (Score ≥ 45)",
    )
    ax_rank.axvline(
        25,
        color="orange",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label="Tier-2 Advisory (Score ≥ 25)",
    )
    ax_rank.set_title(
        "1. Maintenance Priority: Dynamic MHI (Calibrated to 4500 kW)",
        fontsize=11,
        fontweight="bold",
    )
    ax_rank.set_xlabel(
        "Composite Severity Score (0 = Normal, 100 = ISO Trip / 75°C Boundary)",
        fontweight="semibold",
    )
    ax_rank.set_xlim(0, 105)
    ax_rank.legend(loc="lower right")

    # Panel 2: 4.5 MW Flexible Main Coupling Transmission
    if (
        "2RV_Velocity (mm/s)" in df_op.columns
        and "3RV_Velocity (mm/s)" in df_op.columns
    ):
        sc = ax_couple.scatter(
            df_op["2RV_Velocity (mm/s)"],
            df_op["3RV_Velocity (mm/s)"],
            c=df_op.get("2RV_FC", pd.Series(3.0)),
            cmap="viridis",
            s=35,
            alpha=0.75,
        )
        cb = plt.colorbar(sc, ax=ax_couple)
        cb.set_label("Motor DE Crest Factor (FC)", fontweight="semibold")
        ax_couple.axvline(
            V_GOOD,
            color="green",
            linestyle=":",
            alpha=0.6,
            label=f"ISO Good ({V_GOOD} mm/s)",
        )
        ax_couple.axhline(V_GOOD, color="green", linestyle=":", alpha=0.6)
        ax_couple.set_title(
            "2. 4.5 MW Main Coupling Transmission (Motor 2RV vs Flender 3RV)",
            fontsize=11,
            fontweight="bold",
        )
        ax_couple.set_xlabel(
            "Motor DE Velocity (mm/s) [Driving Side]", fontweight="semibold"
        )
        ax_couple.set_ylabel(
            "Gearbox Input DE Velocity (mm/s) [Driven Side]", fontweight="semibold"
        )
        ax_couple.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "03_vulnerability_and_coupling_flender.png"), dpi=140
    )
    plt.close(fig)


# Module 4: Thermal-Mechanical Phase Space Envelopes (Switched D to 11RO)
def plot_thermal_mechanical_phase_space(df_op, export_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    # Panel C: Motor DE Bearing (2RV) - Has Temp
    ax3 = axes[0]
    sc3 = ax3.scatter(
        df_op.get("2RV_Temperature (°C)", pd.Series(42.0)),
        df_op.get("2RV_Velocity (mm/s)", pd.Series(1.5)),
        c=df_op.get("2RV_FC", pd.Series(3.2)),
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
        "C. 4500 kW Motor DE: Hydrodynamic Film vs Binding Runaway",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Bearing Temperature (°C)", fontweight="semibold")
    ax3.set_ylabel("RMS Velocity (mm/s)", fontweight="semibold")
    ax3.set_xlim(30, 80)
    ax3.legend(loc="upper left")

    # Panel D: FLENDER Sump Equilibrium using 11RO (Top Casing/Return Oil)
    ax4 = axes[1]
    if (
        "11RO_Temperature (°C)" in df_op.columns
        and "3RV_Acceleration (g)" in df_op.columns
    ):
        sc4 = ax4.scatter(
            df_op["11RO_Temperature (°C)"],
            df_op["3RV_Acceleration (g)"],
            c=df_op.get("3RV_FC", pd.Series(3.1)),
            cmap="magma",
            s=35,
            alpha=0.75,
        )
        cb4 = plt.colorbar(sc4, ax=ax4)
        cb4.set_label("Input DE Crest Factor (FC)", fontweight="semibold")
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
            "D. Flender Input Pinion: Mesh Friction vs Oil Sump Temperature",
            fontsize=11,
            fontweight="bold",
        )
        ax4.set_xlabel(
            "Return Oil Sump Temp [11RO] (°C) [VG 460]", fontweight="semibold"
        )
        ax4.set_ylabel("Input Shaft RMS Acceleration [3RV] (g)", fontweight="semibold")
        ax4.set_xlim(30, 80)
        ax4.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "04_thermal_mechanical_envelope_flender.png"), dpi=140
    )
    plt.close(fig)


# Module 5: Stewart's Gear Defect Life-Cycle Trajectory
def plot_defect_lifecycle_trajectory(
    df_op, export_dir, target_stages=["3RV", "5RV", "7RV", "9RV"]
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor("white")
    stage_markers = {"3RV": "D", "5RV": "o", "7RV": "^", "9RV": "s"}
    stage_colors = {
        "3RV": "#1f77b4",
        "5RV": "#2ca02c",
        "7RV": "#ff7f0e",
        "9RV": "#d62728",
    }

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
        if (
            f"{sensor}_Velocity (mm/s)" in df_op.columns
            and f"{sensor}_FC" in df_op.columns
        ):
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
                label=f"Trajectory: {sensor}",
            )
            ax1.scatter(v_rms, fc, color=color, s=25, alpha=0.45, edgecolor="none")
            ax1.scatter(
                v_rms.iloc[-1],
                fc.iloc[-1],
                s=170,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
                label=f"Current: {sensor}",
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
        if (
            f"{sensor}_Velocity (mm/s)" in df_op.columns
            and f"{sensor}_K Factor" in df_op.columns
        ):
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
                label=f"Trajectory: {sensor}",
            )
            ax2.scatter(v_rms, kurt, color=color, s=25, alpha=0.45, edgecolor="none")
            ax2.scatter(
                v_rms.iloc[-1],
                kurt.iloc[-1],
                s=170,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
                label=f"Current: {sensor}",
            )

    ax2.set_title(
        "B. Statistical Kurtosis Trajectory vs. Velocity Severity",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel(
        "RMS Vibration Velocity [mm/s] (ISO Severity Band)", fontweight="semibold"
    )
    ax2.set_ylabel(
        "Secondary Peakedness: Kurtosis [K-Factor] (μ₄ / σ⁴)", fontweight="semibold"
    )
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
    plt.savefig(
        os.path.join(export_dir, "05_gear_defect_lifecycle_trajectory.png"), dpi=140
    )
    plt.close(fig)


# Module 6: Shaft Dynamic Deflection & Casing Skew Matrix (4 Parallel Shaft Lines)
def plot_shaft_deflection_matrix(df_op, export_dir):
    skews = []
    for shaft, (ib, ob) in SHAFT_ALIGNED_PAIRS.items():
        if f"{ib}_VC_G (g)" in df_op.columns and f"{ob}_VC_G (g)" in df_op.columns:
            peak_diff = df_op[f"{ib}_VC_G (g)"].mean() - df_op[f"{ob}_VC_G (g)"].mean()

            if (
                ib in CHANNELS_WITH_TEMP
                and ob in CHANNELS_WITH_TEMP
                and f"{ib}_Temperature (°C)" in df_op.columns
                and f"{ob}_Temperature (°C)" in df_op.columns
            ):
                temp_diff = (
                    df_op[f"{ib}_Temperature (°C)"].mean()
                    - df_op[f"{ob}_Temperature (°C)"].mean()
                )
            else:
                temp_diff = np.nan

            skews.append(
                {
                    "Shaft Line": shaft,
                    "Δ Peak Shock (g)": peak_diff,
                    "Δ Temperature (°C)": temp_diff,
                }
            )

    if not skews:
        return

    df_skew = pd.DataFrame(skews).set_index("Shaft Line")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    colors1 = [
        "#d62728" if abs(x) > SKEW_PEAK_ALERT else "#2ca02c"
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
        -SKEW_PEAK_NORMAL,
        SKEW_PEAK_NORMAL,
        color="green",
        alpha=0.15,
        label=f"Parallel Contact Band (±{SKEW_PEAK_NORMAL}g)",
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

    # Handle N/A temps visually
    colors2 = []
    for x in df_skew["Δ Temperature (°C)"]:
        if pd.isna(x):
            colors2.append("#cccccc")  # grey for N/A
        elif abs(x) > SKEW_TEMP_ALERT:
            colors2.append("#ff7f0e")
        else:
            colors2.append("#1f77b4")

    df_temp_plot = df_skew["Δ Temperature (°C)"].fillna(0)
    bars2 = ax2.barh(
        df_skew.index, df_temp_plot, color=colors2, edgecolor="black", linewidth=0.8
    )
    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.axvspan(
        -SKEW_TEMP_NORMAL,
        SKEW_TEMP_NORMAL,
        color="blue",
        alpha=0.1,
        label=f"Balanced Thermal Band (±{SKEW_TEMP_NORMAL}°C)",
    )
    ax2.set_title("2. Thermal Skew: Differential Bearing Loading", fontweight="bold")
    ax2.set_xlabel("Δ Temperature (Inboard - Outboard) [°C]", fontweight="semibold")
    ax2.legend(loc="lower right")

    for idx, bar in enumerate(bars2):
        w = bar.get_width()
        val = df_skew["Δ Temperature (°C)"].iloc[idx]
        if pd.isna(val):
            ax2.text(
                0.1,
                bar.get_y() + bar.get_height() / 2,
                "N/A (Single RTD)",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#555",
            )
        else:
            ax2.text(
                w + (0.2 if w >= 0 else -0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}°C",
                va="center",
                ha="left" if w >= 0 else "right",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle(
        "Shaft Alignment & Gear Taper Contact Matrix (FLENDER DMG2 22 Casing Rigidity)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "06_shaft_deflection_matrix_flender.png"), dpi=140
    )
    plt.close(fig)


# Module 7: Backlash & Dynamic Chattering Index
def plot_backlash_chatter_index(df_op, export_dir, target_stages=["3RV", "4RV", "9RV"]):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    for sensor in target_stages:
        if (
            f"{sensor}_VCC_G (g)" in df_op.columns
            and f"{sensor}_VC_G (g)" in df_op.columns
        ):
            ratio = df_op[f"{sensor}_VCC_G (g)"] / (df_op[f"{sensor}_VC_G (g)"] + 1e-6)
            ax.plot(
                df_op.index,
                ratio,
                linewidth=1.3,
                label=f"Chatter Ratio: {DRIVE_TRAIN_SENSOR_LABELS[sensor]}",
            )

    ax.axhspan(
        1.6,
        R_CHATTER_NORMAL,
        color="green",
        alpha=0.12,
        label=f"Normal Unidirectional Contact (1.60 - {R_CHATTER_NORMAL:.2f})",
    )
    ax.axhspan(
        R_CHATTER_NORMAL,
        R_CHATTER_ALERT,
        color="orange",
        alpha=0.15,
        label=f"Mild Flank Separation ({R_CHATTER_NORMAL:.2f} - {R_CHATTER_ALERT:.2f})",
    )
    ax.axhspan(
        R_CHATTER_ALERT,
        4.0,
        color="red",
        alpha=0.15,
        label=f"Severe Backlash Chattering (> {R_CHATTER_ALERT:.2f})",
    )
    ax.axhline(R_CHATTER_ALERT, color="red", linestyle="--", linewidth=1.5)

    ax.set_title(
        "Gear Mesh Dynamic Stability: Pure Mechanical Backlash & Rebound Ratio (VCC_G / VC_G)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Impact Rebound Ratio (Pk-to-Pk / Peak)", fontweight="semibold")
    ax.set_xlabel("Acquisition Timestamp", fontweight="semibold")
    ax.set_ylim(1.4, 3.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "07_tooth_backlash_chatter_flender.png"), dpi=140
    )
    plt.close(fig)


# Module 8: Output Stage Radial-to-Axial Load Correlation
def plot_output_radial_axial_correlation(df_op, export_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # Panel A: Low-Speed Output Line Radial vs. Axial Reaction (10RV vs 11RO)
    if (
        "10RV_Velocity (mm/s)" in df_op.columns
        and "11RO_Velocity (mm/s)" in df_op.columns
    ):
        sc1 = ax1.scatter(
            df_op["10RV_Velocity (mm/s)"],
            df_op["11RO_Velocity (mm/s)"],
            c=df_op.get("10RV_FC", pd.Series(3.0)),
            cmap="plasma",
            s=35,
            alpha=0.75,
        )
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("Output OB Crest Factor (FC)", fontweight="semibold")
        ax1.axvline(
            V_GOOD,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Output Radial Good ({V_GOOD} mm/s)",
        )
        ax1.axhline(
            V_GOOD,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Casing Axial Good ({V_GOOD} mm/s)",
        )
        ax1.set_title(
            "A. Output Shaft Radial vs. Casing Axial Thrust Reaction (10RV vs 11RO)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel(
            "Output Outboard Radial Velocity [10RV] (mm/s)", fontweight="semibold"
        )
        ax1.set_ylabel(
            "Top Casing Axial / Reaction Velocity [11RO] (mm/s)", fontweight="semibold"
        )
        ax1.legend(loc="upper left")

    # Panel B: Low-Speed Output Line IB vs. OB Differential (9RV vs 10RV)
    if (
        "9RV_Velocity (mm/s)" in df_op.columns
        and "10RV_Velocity (mm/s)" in df_op.columns
    ):
        # Changed C to purely kinematic Kurtosis since 9RV has no temp
        sc2 = ax2.scatter(
            df_op["9RV_Velocity (mm/s)"],
            df_op["10RV_Velocity (mm/s)"],
            c=df_op.get("9RV_K Factor", pd.Series(3.0)),
            cmap="coolwarm",
            s=35,
            alpha=0.75,
        )
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Output IB Kurtosis (K-Factor)", fontweight="semibold")
        max_val = (
            max(df_op["9RV_Velocity (mm/s)"].max(), df_op["10RV_Velocity (mm/s)"].max())
            * 1.15
        )
        ax2.plot(
            [0, max_val],
            [0, max_val],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Symmetric Radial Line (y = x)",
        )
        ax2.set_title(
            "B. Output Shaft Radial Symmetry (Inboard 9RV vs Outboard 10RV)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Output Inboard Velocity [9RV] (mm/s)", fontweight="semibold")
        ax2.set_ylabel("Output Outboard Velocity [10RV] (mm/s)", fontweight="semibold")
        ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(export_dir, "08_output_radial_axial_correlation.png"), dpi=140
    )
    plt.close(fig)


# ==============================================================================
# 5. PIPELINE EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    export_dir = "../../reports/figures/expert_analysis_BM6"
    os.makedirs(export_dir, exist_ok=True)
    data_source_path = "../../data/interim/processed_BM6.pkl"

    try:
        df_resampled, df_operational, sensor_health_flags = (
            load_and_preprocess_telemetry(data_source_path)
        )
    except FileNotFoundError:
        return

    # 1. 4-Zone Cross-Domain Alert Engine & SAP PM Dispatch
    df_alerts, df_all_records = evaluate_cross_domain_alerts(
        df_operational,
        sensor_health_flags,
        export_dir=export_dir,
        export_all_records=True,
    )

    # 2. Render Full Suite of 8 Calibrated Expert Visualizations
    plot_risk_quadrant_matrix(df_operational, export_dir)
    plot_multi_stage_kinematic_cascade(df_operational, export_dir)
    plot_maintenance_priority_and_coupling(df_operational, export_dir)
    plot_thermal_mechanical_phase_space(df_operational, export_dir)
    plot_defect_lifecycle_trajectory(df_operational, export_dir)
    plot_shaft_deflection_matrix(df_operational, export_dir)
    plot_backlash_chatter_index(df_operational, export_dir)
    plot_output_radial_axial_correlation(df_operational, export_dir)


if __name__ == "__main__":
    main()
