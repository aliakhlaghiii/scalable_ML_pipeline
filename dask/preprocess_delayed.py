"""
preprocess_delayed.py

Task 2:
- Initial preprocessing with Dask Delayed
  - handle missing values (impute)
  - quality filter (drop sparse columns)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from load_and_benchmark import load_yaml_config, get_paths, find_ab_files

Timepoint = Literal["A", "B", "DELTA"]

# ID normalization
_DIGITS = re.compile(r"\d+")

def normalize_id_value(x: Any) -> str:
    """Normalize an ID value by extracting digits and stripping noise."""
    s = "" if x is None else str(x).strip()
    parts = _DIGITS.findall(s)
    if parts:
        try:
            return str(int("".join(parts)))
        except Exception:
            return "".join(parts)
    return s


def normalize_id_series_pdf(s: pd.Series) -> pd.Series:
    """Normalize all ID values in a pandas Series."""
    return s.map(normalize_id_value).astype("string")


def normalize_id_ddf(ddf: dd.DataFrame, id_col: str) -> dd.DataFrame:
    """Normalize ID values in a Dask DataFrame column (partition-wise)."""
    if ddf is None:
        raise ValueError("normalize_id_ddf received None. Upstream loader returned None.")
    meta = ddf._meta.copy()
    meta[id_col] = pd.Series([], dtype="string")
    return ddf.map_partitions(
        lambda pdf: pdf.assign(**{id_col: normalize_id_series_pdf(pdf[id_col])}),
        meta=meta,
    )


# -------------------------
# Radiomics loader
# -------------------------

def extract_feature_vector(
    xlsx_path: Path,
    segmentation_col: str = "Segmentation",
    segmentation_match: str = "suv2.5",
    feature_start_col: int = 23,
) -> pd.Series:
    """Extract the radiomics feature vector from one Excel file."""
    df = pd.read_excel(xlsx_path)
    mask = df[segmentation_col].astype(str).str.contains(segmentation_match, na=False)
    row = df.loc[mask].iloc[0, feature_start_col:]
    numeric = pd.to_numeric(row, errors="coerce")
    numeric.index = numeric.index.astype(str)
    return numeric


def patient_to_row(
    patient_dir: Path,
    which: Timepoint,
    id_col: str = "id",
) -> pd.DataFrame:
    """Convert one patient folder into a single-row feature DataFrame."""
    a_path, b_path = find_ab_files(patient_dir)
    if a_path is None or b_path is None:
        return pd.DataFrame(columns=[id_col])

    a_feat = extract_feature_vector(a_path)
    b_feat = extract_feature_vector(b_path)

    if which == "A":
        feat = a_feat
    elif which == "B":
        feat = b_feat
    else:
        feat = (b_feat - a_feat)

    one = feat.to_frame().T
    one.insert(0, id_col, patient_dir.name)
    one[id_col] = one[id_col].astype("string")
    return one


def build_meta_from_sample(
    data_folder: str | Path, which: Timepoint, id_col: str = "id"
) -> pd.DataFrame:
    """Infer a stable Dask schema (meta) by scanning the first valid patient."""
    root = Path(data_folder)
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        try:
            one = patient_to_row(p, which=which, id_col=id_col)
            if not one.empty:
                meta = one.iloc[0:0].copy()
                meta[id_col] = pd.Series([], dtype="string")
                return meta
        except Exception:
            continue
    return pd.DataFrame({id_col: pd.Series(dtype="string")})


def load_radiomics_dask(
    data_folder: str | Path, which: Timepoint, id_col: str = "id"
) -> dd.DataFrame:
    """Load radiomics into a Dask DataFrame (one partition per patient)."""
    root = Path(data_folder)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not patient_dirs:
        meta = build_meta_from_sample(root, which=which, id_col=id_col)
        return dd.from_pandas(meta.copy(), npartitions=1)

    meta = build_meta_from_sample(data_folder, which=which, id_col=id_col)
    expected_cols = list(meta.columns)

    def patient_to_row_fixed(patient_dir: Path) -> pd.DataFrame:
        try:
            one = patient_to_row(patient_dir, which=which, id_col=id_col)
        except Exception:
            return meta.copy()
        if one.empty:
            return meta.copy()
        one = one.reindex(columns=expected_cols)
        one[id_col] = one[id_col].map(normalize_id_value).astype("string")
        return one

    parts = [delayed(patient_to_row_fixed)(p) for p in patient_dirs]
    return dd.from_delayed(parts, meta=meta)


# -------------------------
# Clinical loader
# -------------------------

def _pick_clinical_id_column(df: pd.DataFrame) -> str:
    """Heuristically pick the most likely clinical ID column name."""
    candidates = []
    for c in df.columns:
        s = str(c).strip().lower()
        score = 0
        if s in {"id", "patient_id", "patientid", "subject_id", "subjectid", "sample_id", "sampleid"}:
            score += 100
        if "patient" in s:
            score += 50
        if s.endswith("id") or " id" in s or "_id" in s:
            score += 40
        if "subject" in s:
            score += 20
        if "sample" in s:
            score += 10
        if score > 0:
            candidates.append((score, c))
    if not candidates:
        raise KeyError("Could not infer an ID column from the clinical file. Pass --clinical-id-col.")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return str(candidates[0][1])


def load_clinical_pandas(
    clinical_path: str | Path,
    id_col_out: str = "id",
    clinical_id_col: str = "",
) -> pd.DataFrame:
    """Load clinical Excel data and standardize the ID column name."""
    df = pd.read_excel(clinical_path)

    if clinical_id_col:
        if clinical_id_col not in df.columns:
            raise KeyError(f"Clinical file has no column named '{clinical_id_col}'.")
        df = df.rename(columns={clinical_id_col: id_col_out})
    else:
        inferred = _pick_clinical_id_column(df)
        df = df.rename(columns={inferred: id_col_out})

    df[id_col_out] = normalize_id_series_pdf(df[id_col_out])
    return df


def clinical_to_dask(df: pd.DataFrame) -> dd.DataFrame:
    """Convert a pandas clinical DataFrame to Dask with a small partition count."""
    nparts = max(1, min(8, len(df) // 50 or 1))
    return dd.from_pandas(df, npartitions=nparts)


# Preprocessing (Task 2)
def drop_sparse_columns(ddf: dd.DataFrame, id_col: str, max_missing_frac: float) -> dd.DataFrame:
    """Drop feature columns with missing fraction above the threshold."""
    cols = [c for c in ddf.columns if c != id_col]
    if not cols:
        return ddf

    missing_frac = ddf[cols].isna().mean().compute()
    keep = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    return ddf[[id_col] + keep]


def impute_numeric_approx_median(ddf: dd.DataFrame, id_col: str) -> dd.DataFrame:
    """Impute numeric columns with approximate median (Dask)."""
    cols = [c for c in ddf.columns if c != id_col]
    numeric_cols = [c for c in cols if str(ddf[c].dtype) not in ("object", "string")]
    if not numeric_cols:
        return ddf

    med = ddf[numeric_cols].median_approximate().compute()
    return ddf.assign(**{c: ddf[c].fillna(med[c]) for c in numeric_cols})


# CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Task 2: preprocessing with Dask Delayed.")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--which", choices=["A", "B", "DELTA"], default="DELTA")
    p.add_argument("--id-col", default="id", help="Unified ID column name used after rename")
    p.add_argument("--clinical-id-col", default="", help="Exact ID column name in clinical file (optional)")
    p.add_argument("--max-missing-frac", type=float, default=0.3)
    p.add_argument("--out", required=True, help="Output path (.parquet recommended)")
    return p.parse_args()


def main() -> None:
    """Run the end-to-end preprocessing pipeline."""
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_folder, clinical_path = get_paths(cfg)

    id_col = args.id_col

    rad = load_radiomics_dask(data_folder, which=args.which, id_col=id_col)
    rad = normalize_id_ddf(rad, id_col=id_col)

    clin_pdf = load_clinical_pandas(
        clinical_path,
        id_col_out=id_col,
        clinical_id_col=args.clinical_id_col,
    )
    clin = clinical_to_dask(clin_pdf)
    clin = normalize_id_ddf(clin, id_col=id_col)

    ddf = rad.merge(clin, on=id_col, how="inner")

    n_rows = ddf.shape[0].compute()
    if n_rows == 0:
        raise SystemExit("Join produced 0 rows. Use --clinical-id-col to set the correct clinical ID column.")

    ddf = drop_sparse_columns(ddf, id_col=id_col, max_missing_frac=args.max_missing_frac)
    ddf = impute_numeric_approx_median(ddf, id_col=id_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        ddf.to_parquet(out_path, write_index=False)
    else:
        ddf.to_csv(out_path, single_file=True, index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()