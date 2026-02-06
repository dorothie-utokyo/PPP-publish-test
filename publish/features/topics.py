"""OpenAlex topic fields."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


TOPIC_COLUMNS = [
    "primary_topic_display_name",
    "primary_subfield_display_name",
    "primary_field_display_name",
    "primary_domain_display_name",
]


def add_topics(df: pd.DataFrame) -> pd.DataFrame:
    # Topics are expected to be prepared upstream (e.g., extracted from OpenAlex work topics).
    require_columns(df, TOPIC_COLUMNS, context="topics")
    return df
