"""Reference age and impact features."""
from __future__ import annotations

import pandas as pd

from publish.utils import mean_or_nan, require_columns, safe_len


def add_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "work_referenced_works",
            "patent_doi_references",
            "work_reference_age_days",
            "patent_reference_age_days",
            "work_reference_cited_by_counts",
            "patent_reference_cited_by_counts",
        ],
        context="references",
    )

    df["num_work_references"] = df["work_referenced_works"].apply(safe_len)
    df["patent_num_references"] = df["patent_doi_references"].apply(safe_len)
    df["work_reference_age_days_mean"] = df["work_reference_age_days"].apply(mean_or_nan)
    df["patent_reference_age_days_mean"] = df["patent_reference_age_days"].apply(mean_or_nan)
    df["work_reference_cited_by_counts_mean"] = df["work_reference_cited_by_counts"].apply(
        mean_or_nan
    )
    df["patent_reference_cited_by_counts_mean"] = df[
        "patent_reference_cited_by_counts"
    ].apply(mean_or_nan)

    return df
