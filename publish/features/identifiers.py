"""Identifiers and pair metadata."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def add_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["paper_id", "work_doi", "pair_source"], context="identifiers")

    # We accept either patent_id_us or patent_id as input, but we always produce both.
    if "patent_id_us" in df.columns:
        df["patent_id"] = (
            df["patent_id_us"].astype(str).str.replace("US-", "", regex=False)
        )
    elif "patent_id" in df.columns:
        df["patent_id_us"] = "US-" + df["patent_id"].astype(str)
        df["patent_id"] = df["patent_id"].astype(str)
    else:
        raise ValueError("identifiers: missing required columns: patent_id or patent_id_us")

    df["pair_id"] = df["paper_id"].astype(str) + "|" + df["patent_id_us"].astype(str)
    return df
