# Pipeline

## Run

From the repo root, using the project’s `uv` environment:

```bash
uv run python -m publish.run_pipeline \
  --input /path/to/input_pairs.parquet \
  --output-dir publish_outputs \
  --ipc-technology-xlsx /path/to/ipc_technology.xlsx \
  --control-root /path/to/control_root
```

Outputs written to `--output-dir`:

- Always written:
  - `final_features.parquet`
  - `final_features.csv`
  - `final_features.xlsx` (requires `openpyxl`)
- Also written when `--control-root` is provided:
  - `final_features_control_combined_y0.{parquet,csv,xlsx}`
  - `final_features_control_combined_y5.{parquet,csv,xlsx}`
  - `final_features_control_noselfcite_combined_y0.{parquet,csv,xlsx}` (if available)
  - `final_features_control_noselfcite_combined_y5.{parquet,csv,xlsx}` (if available)

## Optional control merge inputs

When `--control-root` is passed, the pipeline loads control CSVs from this structure:

```text
control_root/
  pierre_data/
    merged_PPP.csv
    true_merged_PPP.csv
    merged_PPP_y5.csv
    true_merged_PPP_y5.csv
  pierre_data_noselfcite/                 # optional directory
    merged_PPP_Y0_no_selfcite.csv
    true_merged_PPP_Y0_no_selfcite.csv
    merged_PPP_Y5_no_selfcite.csv
    true_merged_PPP_Y5_no_selfcite.csv
```

Behavior:

- Files in `pierre_data/` are required when `--control-root` is set.
- `pierre_data_noselfcite/` and its files are optional; missing optional files are skipped with a warning.
- Compact features are left-joined to each combined control table by `pair_id`.

## Standalone repo (using `uv`)

If you publish this pipeline into its own repository, `uv` will work fine **as long as the new repo root contains a `pyproject.toml`** next to the `publish/` package directory, like:

```text
PPP-publish-test/
  pyproject.toml
  publish/
    run_pipeline.py
    features/
    prep/
    export/
```

Suggested minimal `pyproject.toml` for a standalone repo:

```toml
[project]
name = "ppp-publish"
version = "0.1.0"
description = "Publish-ready feature pipeline"
readme = "publish/README.md"
requires-python = ">=3.13"
dependencies = [
  "numpy>=2.3.4",
  "pandas>=2.3.3",
  "pyarrow>=22.0.0",
  "spacy[lookups]>=3.8.0",
]

[project.optional-dependencies]
# Needed only if you want .xlsx output
excel = ["openpyxl>=3.1.5"]

# Needed only if you want SBERT-based semantic similarity features.
# If not installed, the pipeline will continue and fill those columns with NaN.
sbert = [
  "sentence-transformers>=5.1.2",
  "torch",
]
```

Then in the standalone repo root:

```bash
uv sync --extra excel --extra sbert
uv run python -m publish.run_pipeline --help
```

## Required input columns

The input parquet must contain **at least** the following columns (types are shown as “expected shape”; the loader will also attempt to decode *stringified lists* like `"['A_01', 'C_12']"` into Python lists):

### Identifiers

- `paper_id` (string)
- `work_doi` (string; can be null)
- `pair_source` (string; e.g. `our_data`, `marx_data`, `both`)
- `patent_id_us` (string like `US-1234567`) **or** `patent_id` (digits-only string)

### Team size

- `work_author_ids` (list)
- `patent_inventor_ids` (list)

### Organization / collaboration types

- `patent_assignee_names` (list)
- `patent_assignee_types` (list; USPTO type codes)
- `work_institution_names` (list)
- `work_institution_types` (list; OpenAlex institution types like `company` / `education` / other)

### Journal impact (upstream dependency)

- `journal_impact` (float; OpenAlex source `summary_stats.2yr_mean_citedness`)

### Text similarity

At least one of these pairs must exist:

- `work_title` + `patent_title` (strings)
- `work_abstract` + `patent_abstract` (strings)

### Citation overlap

- `patent_cited_works` (list; OpenAlex work IDs cited by the patent)
- `work_referenced_works` (list; OpenAlex referenced works for the paper)

### Prior author experience

- `work_author_ids` (list)
- At least one ordering column: `work_publication_date` and/or `patent_filing_date` and/or `patent_date` (datetime-like)

### Topics (upstream dependency)

- `primary_topic_display_name` (string)
- `primary_subfield_display_name` (string)
- `primary_field_display_name` (string)
- `primary_domain_display_name` (string)

### Patent classification + IPC sectors

Upstream dependencies:

- `wipo_fields` (string; semicolon-separated)
- `ipc_codes` (list of normalized IPC *section_class* codes like `G_01`, `C_12`)

External file dependency (required at runtime):

- `ipc_technology.xlsx` (WIPO IPC-to-technology concordance)
  - expected columns: `IPC_code`, `Sector_en`
  - the pipeline extracts IPC **classes** (e.g. `C12` from `C12N%`) and maps each normalized `C_12`/`G_01` code to a sector string.

### International collaboration

Provide one of:

- `collab_countries` (list of country codes), **or**
- `work_institution_country_codes` (list) + `patent_assignee_country` (string), **or**
- `institution_country_codes` (list) + `patent_assignee_country` (string)

### Geographic distance

- `patent_assignee_latlon_list` (list of `[lat, lon]` pairs)
- `work_latlon_list` (list of `[lat, lon]` pairs)

### Dates

- `work_publication_date` (datetime-like)
- `patent_filing_date` (datetime-like)

### Reference age & impact (OpenAlex referenced works)

- `work_referenced_works` (list)
- `work_reference_age_days` (list of numbers; relative to `work_publication_date`)
- `work_reference_cited_by_counts` (list of numbers)

### Patent references (DOI-mapped OpenAlex works)

- `patent_doi_references` (list)
- `patent_reference_age_days` (list of numbers; relative to `patent_filing_date`)
- `patent_reference_cited_by_counts` (list of numbers)

### Patent claims (upstream dependency)

- `patent_num_claims` (int)
- `patent_first_claim_length` (int; word count of the first independent claim)

## Final exported columns

The authoritative final column list is in `publish/export/export.py` as `FINAL_COLUMNS`.

