"""Helpers to merge compact outputs with Pierre control datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from publish.utils import require_columns


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing control file: {path}")
    return pd.read_csv(path)


def _prepare_merged_control(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    require_columns(df, ["paperid", "patent"], context=context)
    out = df.copy()
    out["pair_id"] = out["paperid"].astype(str) + "|" + out["patent"].astype(str)
    out = out.drop(columns=["paperid", "patent"], errors="ignore")
    return out


def _prepare_true_merged_control(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    require_columns(df, ["work_id", "patent_id_us"], context=context)
    out = df.copy()
    out["pair_id"] = (
        out["work_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)
        + "|"
        + out["patent_id_us"].astype(str)
    )
    out = out.drop(columns=["work_id", "patent_id_us", "patent_id"], errors="ignore")
    return out


def _combine_controls(merged_df: pd.DataFrame, true_merged_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([merged_df, true_merged_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pair_id"], keep="last")
    return combined


def load_and_prepare_control_frames(control_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load and normalize available Pierre control datasets."""
    control_root = Path(control_root)
    pierre_dir = control_root / "pierre_data"
    noselfcite_dir = control_root / "pierre_data_noselfcite"

    frames: dict[str, pd.DataFrame] = {}

    required_pairs = {
        "control_combined_y0": ("merged_PPP.csv", "true_merged_PPP.csv"),
        "control_combined_y5": ("merged_PPP_y5.csv", "true_merged_PPP_y5.csv"),
    }
    for key, (merged_name, true_merged_name) in required_pairs.items():
        merged_df = _prepare_merged_control(
            _read_csv(pierre_dir / merged_name),
            context=f"{key}:{merged_name}",
        )
        true_merged_df = _prepare_true_merged_control(
            _read_csv(pierre_dir / true_merged_name),
            context=f"{key}:{true_merged_name}",
        )
        frames[key] = _combine_controls(merged_df, true_merged_df)

    optional_pairs = {
        "control_noselfcite_combined_y0": (
            "merged_PPP_Y0_no_selfcite.csv",
            "true_merged_PPP_Y0_no_selfcite.csv",
        ),
        "control_noselfcite_combined_y5": (
            "merged_PPP_Y5_no_selfcite.csv",
            "true_merged_PPP_Y5_no_selfcite.csv",
        ),
    }
    for key, (merged_name, true_merged_name) in optional_pairs.items():
        merged_path = noselfcite_dir / merged_name
        true_merged_path = noselfcite_dir / true_merged_name
        if not merged_path.exists() or not true_merged_path.exists():
            print(f"Skipping optional control dataset {key}; missing files in {noselfcite_dir}")
            continue

        merged_df = _prepare_merged_control(
            _read_csv(merged_path),
            context=f"{key}:{merged_name}",
        )
        true_merged_df = _prepare_true_merged_control(
            _read_csv(true_merged_path),
            context=f"{key}:{true_merged_name}",
        )
        frames[key] = _combine_controls(merged_df, true_merged_df)

    return frames


def merge_compact_with_controls(
    compact_df: pd.DataFrame, control_frames: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Left-join compact output with each prepared control frame by pair_id."""
    require_columns(compact_df, ["pair_id"], context="merge_compact_with_controls:compact_df")

    merged_outputs: dict[str, pd.DataFrame] = {}
    for key, control_df in control_frames.items():
        require_columns(control_df, ["pair_id"], context=f"merge_compact_with_controls:{key}")
        overlap = [c for c in control_df.columns if c != "pair_id" and c in compact_df.columns]

        merged = compact_df.merge(
            control_df,
            on="pair_id",
            how="left",
            suffixes=("_compact", "_control"),
        )
        if len(merged) != len(compact_df):
            raise ValueError(
                f"merge_compact_with_controls:{key} changed row count "
                f"from {len(compact_df)} to {len(merged)}"
            )

        control_columns = [c for c in control_df.columns if c != "pair_id"]
        # When there is overlap, right-side columns receive the _control suffix.
        if overlap:
            control_columns = [
                f"{c}_control" if c in overlap else c for c in control_columns
            ]
        present_control_columns = [c for c in control_columns if c in merged.columns]
        match_rate = (
            merged[present_control_columns].notna().any(axis=1).mean()
            if present_control_columns
            else 0.0
        )
        print(
            f"{key}: rows={len(merged)} controls={len(present_control_columns)} "
            f"match_rate={match_rate:.2%}"
        )
        merged_outputs[key] = merged

    return merged_outputs
