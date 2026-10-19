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
sampling = {
    "sensor_position": "last",
    "Acceleration (g)": "mean",
    "Velocity (mm/s)": "mean",
    "K Factor": "mean",
    "Temperature (°C)": "mean",
    "VC_G (g)": "mean",
    "FC": "mean",
    "VCC_G (g)": "mean",
}


Temp_Vib[:1000].resample(rule="10min").apply(sampling)

# split to days
days = [g for n, g in Temp_Vib.groupby(pd.Grouper(freq="D"))]

data_resample = pd.concat(
    [df.resample(rule="10min").apply(sampling).dropna() for df in days]
)

data_resample.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resample.to_csv("../../data/processed/resampled_data.csv")
# done
