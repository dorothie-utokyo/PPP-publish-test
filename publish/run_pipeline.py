"""Run the publish-ready feature pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from publish.export.export import export_to_csv, export_to_excel, prepare_export
from publish.features.author_experience import add_author_experience
from publish.features.citation_overlap import add_citation_overlap
from publish.features.dates import add_date_features
from publish.features.geo_distance import add_geo_distance
from publish.features.identifiers import add_identifiers
from publish.features.international_collab import add_international_collab
from publish.features.journal_metric import add_journal_metric
from publish.features.org_collab import add_org_collab_features
from publish.features.patent_claims import add_patent_claims
from publish.features.patent_classification import add_patent_classification
from publish.features.references import add_reference_features
from publish.features.team_size import add_team_size_features
from publish.features.text_similarity import add_text_similarity_features
from publish.features.topics import add_topics
from publish.prep.cleanup import cleanup_reference_ages
from publish.prep.load_inputs import load_parquet, prepare_inputs


def build_features(df: pd.DataFrame, *, ipc_technology_xlsx: str | Path) -> pd.DataFrame:
    df = add_identifiers(df)
    df = add_team_size_features(df)
    df = add_org_collab_features(df)
    df = add_journal_metric(df)
    df = add_text_similarity_features(df)
    df = add_citation_overlap(df)
    df = add_author_experience(df)
    df = add_topics(df)
    df = add_patent_classification(df, ipc_technology_xlsx_path=ipc_technology_xlsx)
    df = add_international_collab(df)
    df = add_geo_distance(df)
    df = add_date_features(df)
    df = add_reference_features(df)
    df = add_patent_claims(df)
    return df


def run_pipeline(
    input_path: str | Path, output_dir: str | Path, *, ipc_technology_xlsx: str | Path
) -> dict:
    df = load_parquet(input_path)
    df = prepare_inputs(df)

    df = cleanup_reference_ages(df)
    df = build_features(df, ipc_technology_xlsx=ipc_technology_xlsx)
    export_df = prepare_export(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "final_features.parquet"
    csv_path = output_dir / "final_features.csv"
    excel_path = output_dir / "final_features.xlsx"

    export_df.to_parquet(parquet_path, index=False)
    export_to_csv(export_df, csv_path)
    try:
        export_to_excel(export_df, excel_path)
    except ImportError as exc:
        print(f"Excel export failed (missing dependency): {exc}")
        excel_path = None

    return {
        "parquet": parquet_path,
        "csv": csv_path,
        "excel": excel_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish-ready feature pipeline")
    parser.add_argument("--input", required=True, help="Path to input parquet")
    parser.add_argument(
        "--output-dir",
        default="publish_outputs",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--ipc-technology-xlsx",
        required=True,
        help="Path to external ipc_technology.xlsx mapping file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = run_pipeline(
        args.input, args.output_dir, ipc_technology_xlsx=args.ipc_technology_xlsx
    )
    print("Wrote outputs:", outputs)
