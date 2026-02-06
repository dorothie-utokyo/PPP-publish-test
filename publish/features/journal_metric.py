"""Journal impact feature."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def add_journal_metric(df: pd.DataFrame) -> pd.DataFrame:
    # Journal impact is expected to be prepared upstream (e.g., OpenAlex source 2yr_mean_citedness).
    require_columns(df, ["journal_impact"], context="journal_metric")
    return df
