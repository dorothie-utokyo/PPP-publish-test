"""Team size metrics."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns, safe_len


def add_team_size_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["work_author_ids", "patent_inventor_ids"], context="team_size")

    df["author_team_size"] = df["work_author_ids"].apply(safe_len)
    df["inventor_team_size"] = df["patent_inventor_ids"].apply(safe_len)
    df["team_size_difference"] = df["author_team_size"] - df["inventor_team_size"]

    return df
