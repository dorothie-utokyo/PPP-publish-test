"""Cleanup steps mirrored from the export notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _negative_to_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.nan if pd.notna(x) and x < 0 else x)


def cleanup_reference_ages(df: pd.DataFrame) -> pd.DataFrame:
    if "work_reference_age_days_mean" in df.columns:
        df["work_reference_age_days_mean"] = _negative_to_nan(df["work_reference_age_days_mean"])
    if "mean_age_of_work_references" in df.columns:
        df["mean_age_of_work_references"] = _negative_to_nan(df["mean_age_of_work_references"])
    if "patent_reference_age_days_mean" in df.columns:
        df["patent_reference_age_days_mean"] = _negative_to_nan(df["patent_reference_age_days_mean"])
    if "mean_age_of_patent_references" in df.columns:
        df["mean_age_of_patent_references"] = _negative_to_nan(df["mean_age_of_patent_references"])
    return df
