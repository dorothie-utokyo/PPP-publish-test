"""International collaboration feature."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def _has_multiple_countries(countries):
    if not isinstance(countries, list):
        return pd.NA
    unique = {c for c in countries if c}
    return len(unique) > 1 if unique else pd.NA


def add_international_collab(df: pd.DataFrame) -> pd.DataFrame:
    if "collab_countries" in df.columns:
        df["international_collab"] = df["collab_countries"].apply(_has_multiple_countries)
        return df

    if {"work_institution_country_codes", "patent_assignee_country"} <= set(df.columns):
        require_columns(
            df,
            ["work_institution_country_codes", "patent_assignee_country"],
            context="international_collab",
        )

        def _compute(row):
            work_countries = row.get("work_institution_country_codes")
            patent_country = row.get("patent_assignee_country")
            work_countries = work_countries if isinstance(work_countries, list) else []
            countries = {c for c in work_countries if c}
            if patent_country:
                countries.add(patent_country)
            return len(countries) > 1 if countries else pd.NA

        df["international_collab"] = df.apply(_compute, axis=1)
        return df

    if {"institution_country_codes", "patent_assignee_country"} <= set(df.columns):
        require_columns(
            df,
            ["institution_country_codes", "patent_assignee_country"],
            context="international_collab",
        )

        def _compute_alt(row):
            work_countries = row.get("institution_country_codes")
            patent_country = row.get("patent_assignee_country")
            work_countries = work_countries if isinstance(work_countries, list) else []
            countries = {c for c in work_countries if c}
            if patent_country:
                countries.add(patent_country)
            return len(countries) > 1 if countries else pd.NA

        df["international_collab"] = df.apply(_compute_alt, axis=1)
        return df

    raise ValueError(
        "international_collab: missing required columns: "
        "collab_countries OR (work_institution_country_codes+patent_assignee_country) "
        "OR (institution_country_codes+patent_assignee_country)"
    )

    return df
