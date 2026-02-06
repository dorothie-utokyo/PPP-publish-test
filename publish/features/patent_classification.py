"""Patent classification features (WIPO/IPC)."""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from publish.utils import require_columns


def _to_underscore(class_code: str) -> str:
    # "C12" -> "C_12"
    return f"{class_code[0]}_{class_code[1:]}"


@lru_cache(maxsize=8)
def _ipc_code_to_sector_map(ipc_technology_xlsx_path: str) -> dict[str, str]:
    """Build IPC class -> sector mapping from an external WIPO IPC technology file."""
    path = Path(ipc_technology_xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"ipc_sectors: mapping file not found: {path}")

    ipc = pd.read_excel(path, skiprows=6, dtype={"IPC_code": str})
    required = {"IPC_code", "Sector_en"}
    missing = required - set(ipc.columns)
    if missing:
        raise ValueError(
            "ipc_sectors: mapping file missing required columns: "
            + ", ".join(sorted(missing))
        )

    ipc = ipc.dropna(subset=["IPC_code", "Sector_en"]).copy()
    # Extract IPC class like "C12" from strings like "C12N%"
    ipc["class_code"] = (
        ipc["IPC_code"].astype(str).str.extract(r"^([A-H]\d{2})", expand=False)
    )
    ipc = ipc.dropna(subset=["class_code"]).copy()

    grouped = ipc.groupby("class_code")["Sector_en"].apply(lambda s: " / ".join(sorted(set(s))))
    return {_to_underscore(cc): sector for cc, sector in grouped.items()}


def _normalize_ipc_codes(xs) -> list[str]:
    """Normalize IPC codes into a list of strings.

    `xs` may be: list, numpy array, NA/NaN, None, a single code, or a stringified list.
    """
    if xs is None or xs is pd.NA:
        return []

    if isinstance(xs, (float, np.floating)) and pd.isna(xs):
        return []

    if isinstance(xs, np.ndarray):
        xs = xs.tolist()

    if isinstance(xs, str):
        s = xs.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                xs = ast.literal_eval(s)
            except Exception:
                xs = [s]
        else:
            xs = [s]

    try:
        if len(xs) == 0:
            return []
    except TypeError:
        xs = [xs]

    out: list[str] = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, (float, np.floating)) and pd.isna(x):
            continue
        out.append(str(x))
    return out


def _map_ipc_sectors(xs, mapping: dict[str, str]) -> list[Optional[str]]:
    codes = _normalize_ipc_codes(xs)
    return [mapping.get(code) for code in codes]


def add_patent_classification(
    df: pd.DataFrame, *, ipc_technology_xlsx_path: str | Path
) -> pd.DataFrame:
    # `wipo_fields` and `ipc_codes` are expected to be prepared upstream from USPTO data.
    require_columns(df, ["wipo_fields", "ipc_codes"], context="patent_classification")

    mapping = _ipc_code_to_sector_map(str(ipc_technology_xlsx_path))
    df["ipc_sectors"] = df["ipc_codes"].apply(lambda xs: _map_ipc_sectors(xs, mapping))
    return df
