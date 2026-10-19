import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


files = glob("../../data/raw/*.csv")


def read_data_from_files(files):
    Temp_Vib = pd.DataFrame()
    for f in files:
        participant = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(f)

        Temp_Vib = pd.concat([Temp_Vib, df], axis=0)

    Temp_Vib.index = pd.to_datetime(Temp_Vib["timestamp"], unit="ms")

    del Temp_Vib["timestamp"]
    del Temp_Vib["asset_id"]

    return Temp_Vib


Temp_Vib = read_data_from_files(files)
Temp_Vib.info()
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# not doing this because i only have one dataset, but if i had multiple datasets, i could merge them here.

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    0.000833 set to 0.001667HZ
# thermal probe:    0.000833 set to 0.001667HZ
# Gyroscope:        xxx.000Hz
####
wide_df = Temp_Vib.pivot(
    columns="sensor_position",
    values=[
        "Acceleration (g)",
        "Temperature (°C)",
        "Velocity (mm/s)",
        "K Factor",
        "VC_G (g)",
        "VCC_G (g)",
        "FC",
    ],
)

wide_df.columns = [f"{sensor}_{feat}" for feat, sensor in wide_df.columns]

# 1. Convert index to DatetimeIndex and sort chronologically
wide_df.index = pd.to_datetime(wide_df.index)
wide_df = wide_df.sort_index()

# 2. Resample into uniform 10-minute intervals
data_resample = wide_df.resample(rule="10min").mean().dropna().reset_index()


wide_df.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resample.to_csv("../../data/processed/resampled_data.csv", index=False)
# done
