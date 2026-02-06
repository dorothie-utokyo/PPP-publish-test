"""Date-based features."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["work_publication_date", "patent_filing_date"], context="dates")

    df["publication_year"] = df["work_publication_date"].dt.year.astype("Int64")
    df["patent_priority_year"] = df["patent_filing_date"].dt.year.astype("Int64")
    df["lag_days"] = (df["work_publication_date"] - df["patent_filing_date"]).dt.days

    return df
