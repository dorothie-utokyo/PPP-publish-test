"""Organization and collaboration type features."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


_COMPANY_CODES = {"2", "2.0", 2, 2.0, "3", "3.0", 3, 3.0}


def _unique_count(value):
    if isinstance(value, list):
        return len({v for v in value if v is not None})
    return pd.NA


def _assignee_type(value):
    if not isinstance(value, list):
        return pd.NA
    if all(v in _COMPANY_CODES for v in value):
        return "company"
    return "non-company"


def _normalize_institution_types(types_list):
    if not isinstance(types_list, list):
        return pd.NA
    normalized = []
    for t in types_list:
        if t == "company":
            normalized.append("company")
        elif t == "education":
            normalized.append("education")
        else:
            normalized.append("other")
    return normalized


def _author_type(types_list):
    normalized = _normalize_institution_types(types_list)
    if normalized is pd.NA:
        return pd.NA
    if all(t == "company" for t in normalized):
        return "company"
    if all(t == "education" for t in normalized):
        return "education"
    return "other"


def add_org_collab_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "patent_assignee_names",
            "work_institution_names",
            "patent_assignee_types",
            "work_institution_types",
        ],
        context="org_collab",
    )

    df["multiple_assignee"] = df["patent_assignee_names"].apply(
        lambda x: _unique_count(x) > 1 if isinstance(x, list) else pd.NA
    )
    df["multiple_author_institution"] = df["work_institution_names"].apply(
        lambda x: _unique_count(x) > 1 if isinstance(x, list) else pd.NA
    )
    df["assignee_type"] = df["patent_assignee_types"].apply(_assignee_type)
    df["author_type"] = df["work_institution_types"].apply(_author_type)

    return df
