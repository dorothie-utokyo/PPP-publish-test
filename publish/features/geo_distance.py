"""Geographic distance features."""
from __future__ import annotations

from math import radians
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from publish.utils import require_columns


def _latlon_array(latlon_list: Optional[Iterable]) -> Optional[np.ndarray]:
    if not isinstance(latlon_list, list):
        return None
    coords = []
    for entry in latlon_list:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and entry[0] is not None
            and entry[1] is not None
        ):
            try:
                coords.append((float(entry[0]), float(entry[1])))
            except (TypeError, ValueError):
                continue
    return np.array(coords, dtype=float) if coords else None


def _haversine_km_vectorized(pat_latlon_list, work_latlon_list) -> Optional[np.ndarray]:
    coords1 = _latlon_array(pat_latlon_list)
    coords2 = _latlon_array(work_latlon_list)
    if coords1 is None or coords2 is None:
        return None

    lon1 = np.radians(coords1[:, 1])[:, None]
    lat1 = np.radians(coords1[:, 0])[:, None]
    lon2 = np.radians(coords2[:, 1])[None, :]
    lat2 = np.radians(coords2[:, 0])[None, :]

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def avg_haversine_km(pat_latlon_list, work_latlon_list) -> float:
    km = _haversine_km_vectorized(pat_latlon_list, work_latlon_list)
    if km is None or km.size == 0:
        return np.nan
    return float(np.mean(km))


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        ["patent_assignee_latlon_list", "work_latlon_list"],
        context="geo_distance",
    )
    df["avg_haversine_km"] = df.apply(
        lambda row: avg_haversine_km(
            row.get("patent_assignee_latlon_list"), row.get("work_latlon_list")
        ),
        axis=1,
    )
    return df
