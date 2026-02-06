"""Text similarity features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from publish.scores import semantic_similarity_score_sbert, semantic_similarity_score_word_overlap


def _score_word_overlap(row, col_a, col_b):
    a = row.get(col_a)
    b = row.get(col_b)
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return semantic_similarity_score_word_overlap(str(a), str(b))


_SBERT_ENABLED = True
_SBERT_WARNED = False


def _score_semantic(row, col_a, col_b):
    global _SBERT_ENABLED, _SBERT_WARNED

    if not _SBERT_ENABLED:
        return np.nan

    a = row.get(col_a)
    b = row.get(col_b)
    if pd.isna(a) or pd.isna(b):
        return np.nan
    try:
        score = semantic_similarity_score_sbert(str(a), str(b))
    except ImportError as exc:
        # Allow the pipeline to run without SBERT dependencies installed.
        _SBERT_ENABLED = False
        if not _SBERT_WARNED:
            print(f"SBERT similarity disabled (missing dependency): {exc}")
            _SBERT_WARNED = True
        return np.nan

    return np.nan if score is None else score


def add_text_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    has_title = {"work_title", "patent_title"} <= set(df.columns)
    has_abstract = {"work_abstract", "patent_abstract"} <= set(df.columns)

    if not (has_title or has_abstract):
        raise ValueError(
            "text_similarity: missing required columns: "
            "(work_title+patent_title) or (work_abstract+patent_abstract)"
        )

    # Word overlap
    if has_title:
        df["title_word_overlap_score"] = df.apply(
            lambda row: _score_word_overlap(row, "work_title", "patent_title"), axis=1
        )
    else:
        df["title_word_overlap_score"] = np.nan

    if has_abstract:
        df["abstract_word_overlap_score"] = df.apply(
            lambda row: _score_word_overlap(row, "work_abstract", "patent_abstract"), axis=1
        )
    else:
        df["abstract_word_overlap_score"] = np.nan

    df["word_overlap_score"] = df[
        ["title_word_overlap_score", "abstract_word_overlap_score"]
    ].mean(axis=1)

    # SBERT similarity
    if has_title:
        df["title_semantic_similarity"] = df.apply(
            lambda row: _score_semantic(row, "work_title", "patent_title"), axis=1
        )
    else:
        df["title_semantic_similarity"] = np.nan

    if has_abstract:
        df["abstract_semantic_similarity"] = df.apply(
            lambda row: _score_semantic(row, "work_abstract", "patent_abstract"), axis=1
        )
    else:
        df["abstract_semantic_similarity"] = np.nan

    df["semantic_similarity_score"] = df[
        ["title_semantic_similarity", "abstract_semantic_similarity"]
    ].mean(axis=1)

    return df
