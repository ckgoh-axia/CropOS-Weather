"""Quality control flags for METAR sensor data."""
import pandas as pd

PRECIP_MAX_MM_HR = 200.0


def flag_metar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qc_flag"] = "ok"
    df.loc[df["precip_mm"] > PRECIP_MAX_MM_HR, "qc_flag"] = "outlier_high"
    df.loc[df["precip_mm"] < 0, "qc_flag"] = "outlier_low"
    return df
