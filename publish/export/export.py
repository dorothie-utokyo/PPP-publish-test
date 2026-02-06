"""Prepare and export the final feature set."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from publish.utils import strip_illegal_excel_chars

RENAME_MAP = {
    "patent_num_references": "patent_reference_list_length",
    "num_work_references": "work_reference_list_length",
    "work_reference_age_days_mean": "mean_age_of_work_references",
    "patent_reference_age_days_mean": "mean_age_of_patent_references",
    "relative_filing_date": "lag_days",
    "avg_haversine_km": "geographical_distance",
}

FINAL_COLUMNS = [
    "patent_id",
    "patent_id_us",
    "paper_id",
    "work_doi",
    "pair_id",
    "pair_source",
    "author_team_size",
    "inventor_team_size",
    "team_size_difference",
    "multiple_assignee",
    "multiple_author_institution",
    "assignee_type",
    "author_type",
    "journal_impact",
    "word_overlap_score",
    "semantic_similarity_score",
    "citation_overlap_score",
    "previous_experience",
    "previous_experience_first_last",
    "primary_topic_display_name",
    "primary_subfield_display_name",
    "primary_field_display_name",
    "primary_domain_display_name",
    "wipo_fields",
    "ipc_codes",
    "ipc_sectors",
    "international_collab",
    "geographical_distance",
    "publication_year",
    "patent_priority_year",
    "lag_days",
    "mean_age_of_work_references",
    "mean_age_of_patent_references",
    "work_reference_cited_by_counts_mean",
    "patent_reference_cited_by_counts_mean",
    "work_reference_list_length",
    "patent_reference_list_length",
    "patent_num_claims",
    "patent_first_claim_length",
]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    for old, new in RENAME_MAP.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
        elif old in df.columns and new in df.columns:
            df = df.drop(columns=[old])
    return df


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_columns(df)
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "prepare_export: missing required final columns: " + ", ".join(missing)
        )

    df = df[FINAL_COLUMNS]
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(object_cols):
        df.loc[:, object_cols] = df[object_cols].apply(lambda col: col.map(strip_illegal_excel_chars))
    return df


def export_to_excel(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    df.to_excel(output_path, index=False)
    return output_path


def export_to_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    return output_path
