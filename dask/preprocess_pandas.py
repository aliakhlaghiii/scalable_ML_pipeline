"""
preprocess_pandas.py

Pandas baseline for Task 2 (Initial Data Preprocessing).

This mirrors the logic of preprocess_delayed.py but runs with a
traditional single-machine pandas workflow.

Steps:
- Load radiomics features (per-patient Excel files) into a pandas DataFrame
- Load clinical table (Excel) into a pandas DataFrame
- Normalize/standardize patient IDs and join radiomics + clinical
- Drop sparse columns (high missingness)
- Impute numeric columns with the median

Output:
- A parquet dataset directory (or a single parquet file path).

Notes:
- This script exists to support Task 6 performance comparisons between
  Dask and pandas for preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

from load_and_benchmark import load_yaml_config, get_paths
from preprocess_delayed import (
    Timepoint,
    normalize_id_value,
    normalize_id_series_pdf,
    patient_to_row,
    load_clinical_pandas,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Task 2 pandas baseline preprocessing.")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--which", choices=["A", "B", "DELTA"], default="DELTA")
    p.add_argument("--id-col", default="id", help="Unified ID column name used after rename")
    p.add_argument("--clinical-id-col", default="", help="Exact ID column name in clinical file (optional)")
    p.add_argument("--max-missing-frac", type=float, default=0.3)
    p.add_argument("--out", required=True, help="Output path (.parquet recommended)")
    return p.parse_args()


def make_unique_columns(cols: List[str]) -> List[str]:
    """Make column names unique by appending '__dupN' suffixes."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        c0 = str(c)
        if c0 not in seen:
            seen[c0] = 0
            out.append(c0)
        else:
            seen[c0] += 1
            out.append(f"{c0}__dup{seen[c0]}")
    return out


def ensure_unique_columns_pdf(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pandas DataFrame has unique column names."""
    cols = list(map(str, df.columns))
    if len(cols) == len(set(cols)):
        return df
    df = df.copy()
    df.columns = make_unique_columns(cols)
    return df


def load_radiomics_pandas(data_folder: str | Path, which: Timepoint, id_col: str = "id") -> pd.DataFrame:
    """Load radiomics into a single pandas DataFrame (one row per patient)."""
    root = Path(data_folder)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    rows: List[pd.DataFrame] = []
    for pdir in patient_dirs:
        try:
            one = patient_to_row(pdir, which=which, id_col=id_col)
        except Exception:
            continue
        if one is None or one.empty:
            continue
        one = one.copy()
        one[id_col] = normalize_id_series_pdf(one[id_col])
        rows.append(one)

    if not rows:
        return pd.DataFrame({id_col: pd.Series(dtype="string")})

    df = pd.concat(rows, axis=0, ignore_index=True)
    df = ensure_unique_columns_pdf(df)
    df[id_col] = normalize_id_series_pdf(df[id_col])
    return df


def drop_sparse_columns_pdf(df: pd.DataFrame, id_col: str, max_missing_frac: float) -> pd.DataFrame:
    """Drop columns whose missing fraction is above the threshold."""
    cols = [c for c in df.columns if c != id_col]
    if not cols:
        return df
    miss = df[cols].isna().mean()
    keep = miss[miss <= float(max_missing_frac)].index.tolist()
    return df[[id_col] + keep]


def impute_numeric_median_pdf(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Impute numeric columns with median (pandas baseline)."""
    df = df.copy()
    cols = [c for c in df.columns if c != id_col]
    if not cols:
        return df

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df

    med = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(med)
    return df


def main() -> None:
    """Run the pandas preprocessing pipeline."""
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_folder, clinical_path = get_paths(cfg)

    id_col = args.id_col

    rad = load_radiomics_pandas(data_folder, which=args.which, id_col=id_col)
    clin = load_clinical_pandas(
        clinical_path,
        id_col_out=id_col,
        clinical_id_col=args.clinical_id_col,
    )

    rad = ensure_unique_columns_pdf(rad)
    clin = ensure_unique_columns_pdf(clin)

    if id_col not in rad.columns:
        raise SystemExit(f"Radiomics table has no '{id_col}' column.")
    if id_col not in clin.columns:
        raise SystemExit(f"Clinical table has no '{id_col}' column.")

    rad[id_col] = normalize_id_series_pdf(rad[id_col])
    clin[id_col] = normalize_id_series_pdf(clin[id_col])

    df = rad.merge(clin, on=id_col, how="inner")
    if len(df) == 0:
        raise SystemExit("Join produced 0 rows. Use --clinical-id-col to set the correct clinical ID column.")

    df = drop_sparse_columns_pdf(df, id_col=id_col, max_missing_frac=args.max_missing_frac)
    df = impute_numeric_median_pdf(df, id_col=id_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet" or out_path.name.endswith(".parquet"):
        df.to_parquet(out_path.as_posix(), index=False)
    else:
        df.to_csv(out_path.as_posix(), index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()