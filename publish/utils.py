"""Shared helpers for the publish pipeline."""
from __future__ import annotations

import ast
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

_ILLEGAL_EXCEL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def require_columns(df: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    """Fail fast if any required columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {', '.join(missing)}")


def decode_list(value):
    """Decode a stringified list into a Python list when possible."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    if not isinstance(value, str):
        return pd.NA

    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return pd.NA

    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else pd.NA
    except Exception:
        return pd.NA


def normalize_list_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].apply(decode_list)
    return df


def normalize_list_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        normalize_list_column(df, column)
    return df


def safe_len(value) -> Optional[int]:
    if isinstance(value, list):
        return len(value)
    return pd.NA


def mean_or_nan(values: Optional[Iterable]) -> float:
    if not isinstance(values, list) or len(values) == 0:
        return np.nan
    numeric = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not numeric:
        return np.nan
    return float(np.mean(numeric))


def strip_illegal_excel_chars(value):
    if isinstance(value, str):
        return _ILLEGAL_EXCEL_RE.sub("", value)
    return value


def ensure_datetime(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df
