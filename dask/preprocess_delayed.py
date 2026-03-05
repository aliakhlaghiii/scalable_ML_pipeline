# preprocess_delayed.py
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
    meta = ddf._meta.copy()
    meta[id_col] = pd.Series(dtype="string")

    def _norm(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.copy()
        pdf[id_col] = normalize_id_series_pdf(pdf[id_col])
        return pdf

    return ddf.map_partitions(_norm, meta=meta)


# Radiomics loading (Delayed -> Dask DataFrame)
def extract_feature_vector(
    xlsx_path: Path,
    segmentation_col: str = "Segmentation",
    segmentation_match: str = "suv2.5",
    feature_start_col: int = 23,
) -> pd.Series:
    """Extract numeric radiomics feature vector from a given Excel file."""
    df = pd.read_excel(xlsx_path)
    if segmentation_col not in df.columns:
        raise KeyError(f"Missing '{segmentation_col}' in {xlsx_path.name}")

    mask = df[segmentation_col].astype(str).str.contains(segmentation_match, na=False)
    row = df.loc[mask].iloc[0, feature_start_col:]
    numeric = pd.to_numeric(row, errors="coerce").dropna()
    numeric.index = numeric.index.astype(str)
    return numeric


def patient_to_row(patient_dir: Path, which: Timepoint, id_col: str) -> pd.DataFrame:
    """Convert a patient's A/B radiomics files into a single feature row."""
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
        feat = (b_feat - a_feat).dropna()

    one = feat.to_frame().T
    one.insert(0, id_col, patient_dir.name)
    return one


def build_meta_from_sample(data_folder: str | Path, which: Timepoint, id_col: str) -> pd.DataFrame:
    """Build a Dask meta DataFrame by scanning for the first valid patient row."""
    root = Path(data_folder)
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        try:
            one = patient_to_row(p, which=which, id_col=id_col)
            if not one.empty:
                return one.iloc[0:0].copy()
        except Exception:
            continue
    return pd.DataFrame({id_col: pd.Series(dtype="object")})


def load_radiomics_dask(data_folder: str | Path, which: Timepoint, id_col: str) -> dd.DataFrame:
    """Load radiomics as a Dask DataFrame using delayed patient tasks."""
    root = Path(data_folder)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    meta = build_meta_from_sample(root, which=which, id_col=id_col)
    expected_cols = list(meta.columns)

    def _patient_fixed(pdir: Path) -> pd.DataFrame:
        try:
            one = patient_to_row(pdir, which=which, id_col=id_col)
            if one.empty:
                return meta.copy()
        except Exception:
            return meta.copy()
        return one.reindex(columns=expected_cols)

    delayed_parts = [delayed(_patient_fixed)(p) for p in patient_dirs]
    return dd.from_delayed(delayed_parts, meta=meta)


# Clinical loading
def load_clinical_pandas(
    clinical_path: str | Path,
    id_col_out: str,
    clinical_id_col: str = "",
) -> pd.DataFrame:
    """Load clinical spreadsheet as pandas and standardize ID column name."""
    clinical_path = Path(clinical_path)
    pdf = pd.read_excel(clinical_path)

    if clinical_id_col and clinical_id_col in pdf.columns:
        id_src = clinical_id_col
    else:
        # heuristic: pick first column containing 'id'
        cand = [c for c in pdf.columns if "id" in str(c).lower()]
        if not cand:
            raise KeyError("Could not infer clinical ID column. Provide --clinical-id-col")
        id_src = cand[0]

    pdf = pdf.rename(columns={id_src: id_col_out}).copy()
    return pdf


def clinical_to_dask(pdf: pd.DataFrame) -> dd.DataFrame:
    """Convert clinical pandas DataFrame to a Dask DataFrame with a reasonable partitioning."""
    if len(pdf) == 0:
        return dd.from_pandas(pdf, npartitions=1)

    # simple heuristic: up to 8 partitions
    nparts = max(1, min(8, len(pdf) // 50 or 1))
    return dd.from_pandas(pdf, npartitions=nparts)


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
 
 
def drop_sparse_columns_delayed(ddf: dd.DataFrame, id_col: str, max_missing_frac: float) -> dd.DataFrame:
    """Drop feature columns with missing fraction above the threshold using Dask Delayed."""
    cols = [c for c in ddf.columns if c != id_col]
    if not cols:
        return ddf

    def _missing_counts(pdf: pd.DataFrame, cols_: List[str]) -> Tuple[int, Dict[str, int]]:
        """Partition-level missing-value counts."""
        n = int(len(pdf))
        miss = pdf[cols_].isna().sum().to_dict()
        return n, {k: int(v) for k, v in miss.items()}

    parts = ddf.to_delayed()
    counted = [delayed(_missing_counts)(p, cols) for p in parts]

    def _aggregate(counts: List[Tuple[int, Dict[str, int]]], cols_: List[str]) -> pd.Series:
        """Aggregate missing counts into missing fractions."""
        total_n = 0
        miss_totals = {c: 0 for c in cols_}
        for n, miss in counts:
            total_n += int(n)
            for c in cols_:
                miss_totals[c] += int(miss.get(c, 0))
        if total_n <= 0:
            return pd.Series({c: 1.0 for c in cols_})
        return pd.Series({c: miss_totals[c] / float(total_n) for c in cols_})

    missing_frac = delayed(_aggregate)(counted, cols).compute()
    keep = missing_frac[missing_frac <= float(max_missing_frac)].index.tolist()
    return ddf[[id_col] + keep]


def impute_numeric_approx_median_delayed(ddf: dd.DataFrame, id_col: str) -> dd.DataFrame:
    """Impute numeric columns with approximate median using Dask Delayed for partition transforms."""
    cols = [c for c in ddf.columns if c != id_col]
    numeric_cols = [c for c in cols if str(ddf[c].dtype) not in ("object", "string")]
    if not numeric_cols:
        return ddf

    med = ddf[numeric_cols].median_approximate().compute()
    med_map = {c: float(med[c]) for c in numeric_cols}

    def _impute(pdf: pd.DataFrame, numeric_cols_: List[str], med_map_: Dict[str, float]) -> pd.DataFrame:
        """Partition-level imputation."""
        pdf = pdf.copy()
        for c in numeric_cols_:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(med_map_[c])
        return pdf

    meta = ddf._meta.copy()
    for c in numeric_cols:
        meta[c] = pd.Series(dtype="float64")

    parts = ddf.to_delayed()
    imputed = [delayed(_impute)(p, numeric_cols, med_map) for p in parts]
    return dd.from_delayed(imputed, meta=meta)


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

    ddf = drop_sparse_columns_delayed(ddf, id_col=id_col, max_missing_frac=args.max_missing_frac)
    ddf = impute_numeric_approx_median_delayed(ddf, id_col=id_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        ddf.to_parquet(out_path, write_index=False)
    else:
        ddf.to_csv(out_path, single_file=True, index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()