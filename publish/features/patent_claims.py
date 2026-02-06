"""Patent claims features."""
from __future__ import annotations

import pandas as pd

from publish.utils import require_columns


def add_patent_claims(df: pd.DataFrame) -> pd.DataFrame:
    # Claims features are expected to be prepared upstream from USPTO claim-level data.
    require_columns(
        df,
        ["patent_num_claims", "patent_first_claim_length"],
        context="patent_claims",
    )
    return df
