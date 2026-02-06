"""Prior author experience features."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def _first_last_authors(authors, positions=None):
    if not isinstance(authors, list) or len(authors) == 0:
        return []
    if isinstance(positions, list) and len(positions) == len(authors):
        return [a for a, p in zip(authors, positions) if p in {"first", "last"}]
    if len(authors) == 1:
        return [authors[0]]
    return [authors[0], authors[-1]]


def _experience_by_order(df: pd.DataFrame, author_ids_col: str, positions_col: str | None):
    order_cols = [c for c in ["work_publication_date", "patent_date", "patent_filing_date"] if c in df.columns]
    if not order_cols:
        return None, None

    ordered = df.sort_values(order_cols).index
    seen_all = set()
    seen_fl = set()
    prev = {}
    prev_fl = {}

    for idx in ordered:
        authors = df.at[idx, author_ids_col] if author_ids_col in df.columns else []
        positions = df.at[idx, positions_col] if positions_col and positions_col in df.columns else None
        authors = authors if isinstance(authors, list) else []

        prev[idx] = any(a in seen_all for a in authors) if authors else pd.NA
        seen_all.update(authors)

        fl = _first_last_authors(authors, positions)
        prev_fl[idx] = any(a in seen_fl for a in fl) if fl else pd.NA
        seen_fl.update(fl)

    return pd.Series(prev), pd.Series(prev_fl)


def add_author_experience(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["work_author_ids"], context="author_experience")

    positions_col = None
    if "work_author_positions" in df.columns:
        positions_col = "work_author_positions"
    elif "work_author_positions_list" in df.columns:
        positions_col = "work_author_positions_list"

    prev, prev_fl = _experience_by_order(df, "work_author_ids", positions_col)
    if prev is None:
        raise ValueError(
            "author_experience: missing required columns: "
            "need at least one ordering column among "
            "work_publication_date, patent_date, patent_filing_date"
        )

    df["previous_experience"] = prev
    df["previous_experience_first_last"] = prev_fl
    return df
