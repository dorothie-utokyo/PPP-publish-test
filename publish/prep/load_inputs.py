"""Load and normalize inputs for the publish pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from publish.utils import ensure_datetime, normalize_list_columns

DEFAULT_LIST_COLUMNS = [
    "work_author_ids",
    "work_author_positions",
    "work_author_positions_list",
    "patent_inventor_ids",
    "patent_assignee_names",
    "patent_assignee_types",
    "ipc_codes",
    "work_institution_names",
    "work_institution_types",
    "work_institution_country_codes",
    "institution_country_codes",
    "work_latlon_list",
    "patent_assignee_latlon_list",
    "work_referenced_works",
    "patent_cited_works",
    "work_reference_dates",
    "patent_reference_dates",
    "work_reference_age_days",
    "patent_reference_age_days",
    "work_reference_cited_by_counts",
    "patent_reference_cited_by_counts",
    "patent_doi_references",
]

DEFAULT_DATE_COLUMNS = [
    "work_publication_date",
    "patent_filing_date",
    "patent_date",
]

REQUIRED_COLUMNS = [
    "paper_id",
    "pair_source",
    "work_doi",
]


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_inputs(
    df: pd.DataFrame,
    list_columns: Optional[Sequence[str]] = None,
    date_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    list_columns = list_columns or DEFAULT_LIST_COLUMNS
    date_columns = date_columns or DEFAULT_DATE_COLUMNS

    df = normalize_list_columns(df, list_columns)
    df = ensure_datetime(df, date_columns)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"prepare_inputs: missing required columns: {', '.join(missing)}")

    if "patent_id_us" not in df.columns and "patent_id" not in df.columns:
        raise ValueError("prepare_inputs: missing required columns: patent_id or patent_id_us")

    return df
