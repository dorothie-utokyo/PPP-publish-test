"""Citation overlap feature."""
from __future__ import annotations

import pandas as pd

from publish.scores import citation_overlap_score
from publish.utils import require_columns


def _compute_overlap(row):
    return citation_overlap_score(
        row.get("patent_cited_works"), row.get("work_referenced_works")
    )


def add_citation_overlap(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        ["patent_cited_works", "work_referenced_works"],
        context="citation_overlap",
    )
    df["citation_overlap_score"] = df.apply(_compute_overlap, axis=1)
    return df
