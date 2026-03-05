# prepare_ml_dataset.py
"""
prepare_ml_dataset.py

Builds a train/test ML dataset from a preprocessed parquet directory using Dask.

Covers requirements:
- Normalize numerical features using z-score scaling with Dask (fit on TRAIN only)
- Encode the target variable
- Split the dataset into training and test sets using Dask

Outputs:
  outputs/ml/train.parquet
  outputs/ml/test.parquet
  outputs/ml/target_mapping.json
  outputs/ml/feature_cols.json
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import dask.dataframe as dd

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet dataset dir (from Task 2)")
    p.add_argument("--outdir", default="outputs/ml", help="Output folder")
    p.add_argument("--target-col", default="surv_bestresponse_car.factor", help="Target column (categorical)")
    p.add_argument("--id-col", default="id", help="ID column name")
    p.add_argument("--test-frac", type=float, default=0.2, help="Fraction for test set")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# Target encoding
def encode_target(ddf: dd.DataFrame, target_col: str) -> Tuple[dd.DataFrame, Dict[str, int]]:
    """Encode a categorical target column to integer labels."""
    # compute unique values (drops NaN)
    uniq = ddf[target_col].dropna().unique().compute().tolist()
    uniq = sorted([str(u) for u in uniq])

    mapping = {u: i for i, u in enumerate(uniq)}

    def _encode(pdf: pd.DataFrame) -> pd.DataFrame:
        """Partition-level target encoding."""
        pdf = pdf.copy()
        pdf["target"] = pdf[target_col].astype(str).map(mapping).astype("int64")
        return pdf

    meta = ddf._meta.copy()
    meta["target"] = pd.Series(dtype="int64")
    return ddf.map_partitions(_encode, meta=meta), mapping


# Leakage removal
_LEAK_PAT = re.compile(r"^(surv_|post_cart_)", flags=re.IGNORECASE)

def leak_columns(cols: List[str], target_col: str) -> List[str]:
    """Detect columns that could leak label information (heuristic)."""
    leak = []
    for c in cols:
        if c == target_col or c == "target":
            continue
        if _LEAK_PAT.search(str(c)):
            leak.append(c)
    # also drop any near-duplicates of target col
    for c in ("bestresponse", "response", "survival"):
        if c in cols and c not in leak:
            leak.append(c)
    return leak


def build_mapping_for_series(s: dd.Series) -> Dict[str, int]:
    """Build a value->code mapping for a Dask Series (stringified)."""
    uniq = s.dropna().astype(str).unique().compute().tolist()
    uniq = sorted(uniq)
    return {v: i for i, v in enumerate(uniq)}


def encode_object_features(ddf: dd.DataFrame, exclude: List[str]) -> dd.DataFrame:
    """Encode object/string columns to numeric codes (float64) via per-column mapping."""
    obj_cols = [
        c for c, dt in ddf.dtypes.items()
        if str(dt) in ("object", "string") and c not in exclude
    ]
    if not obj_cols:
        return ddf

    mappings: Dict[str, Dict[str, int]] = {}
    for c in obj_cols:
        mappings[c] = build_mapping_for_series(ddf[c])  # compute uniques once per col

    def _encode(pdf: pd.DataFrame) -> pd.DataFrame:
        """Partition-level object encoding; unmapped values become NaN."""
        pdf = pdf.copy()
        for c in obj_cols:
            # Using float64 so NaN is representable (int dtype can't hold NaN).
            pdf[c] = pdf[c].astype(str).map(mappings[c]).astype("float64")
        return pdf

    meta = ddf._meta.copy()
    for c in obj_cols:
        meta[c] = pd.Series(dtype="float64")  # required for Dask type inference stability

    return ddf.map_partitions(_encode, meta=meta)


def encode_object_features_train_test(
    train: dd.DataFrame,
    test: dd.DataFrame,
    exclude: List[str],
) -> Tuple[dd.DataFrame, dd.DataFrame]:
    """Fit object/string encoders on TRAIN only and apply to both train and test."""
    obj_cols = [
        c for c, dt in train.dtypes.items()
        if str(dt) in ("object", "string") and c not in exclude
    ]
    if not obj_cols:
        return train, test

    mappings: Dict[str, Dict[str, int]] = {}
    for c in obj_cols:
        mappings[c] = build_mapping_for_series(train[c])  # compute uniques on TRAIN only

    def _encode_with_map(pdf: pd.DataFrame) -> pd.DataFrame:
        """Partition-level object encoding; unmapped values become NaN."""
        pdf = pdf.copy()
        for c in obj_cols:
            pdf[c] = pdf[c].astype(str).map(mappings[c]).astype("float64")
        return pdf

    meta_train = train._meta.copy()
    meta_test = test._meta.copy()
    for c in obj_cols:
        meta_train[c] = pd.Series(dtype="float64")
        meta_test[c] = pd.Series(dtype="float64")

    train_enc = train.map_partitions(_encode_with_map, meta=meta_train)
    test_enc = test.map_partitions(_encode_with_map, meta=meta_test)
    return train_enc, test_enc


def coerce_numeric_features(ddf: dd.DataFrame, cols: List[str]) -> dd.DataFrame:
    """Force selected columns to float64, coercing non-numeric tokens to NaN."""
    def _coerce(pdf: pd.DataFrame) -> pd.DataFrame:
        """Partition-level numeric coercion (e.g., 'NE' -> NaN)."""
        pdf = pdf.copy()
        for c in cols:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float64")
        return pdf

    meta = ddf._meta.copy()
    for c in cols:
        meta[c] = pd.Series(dtype="float64")  # tells Dask the output dtype

    return ddf.map_partitions(_coerce, meta=meta)


def drop_all_nan_columns(ddf: dd.DataFrame, cols: List[str]) -> List[str]:
    """Return subset of cols that have at least one non-null value in the full dataset."""
    # This computes a boolean mask across the cluster (can be expensive for huge wide tables).
    nonnull_any = (~ddf[cols].isna()).any().compute()
    return nonnull_any[nonnull_any].index.tolist()


def zscore_fit_stats(train: dd.DataFrame, cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    """Compute mean/std on TRAIN only for z-score scaling."""
    mu = train[cols].mean().compute()
    # ddof=0 => population std; avoids tiny-sample behavior differences vs ddof=1
    sd = train[cols].std(ddof=0).compute().replace(0.0, np.nan)  # prevent divide-by-zero
    return mu, sd


def zscore_apply(ddf: dd.DataFrame, cols: List[str], mu: pd.Series, sd: pd.Series) -> dd.DataFrame:
    """Apply z-score scaling using provided train-fitted mean/std."""
    def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
        """Partition-level scaling; fill NaNs with train mean before scaling."""
        pdf = pdf.copy()
        for c in cols:
            m = float(mu[c])
            s = float(sd[c]) if pd.notna(sd[c]) else np.nan
            # fill missing with train mean so scaling doesn't inject NaNs downstream
            x = pd.to_numeric(pdf[c], errors="coerce").fillna(m)
            if not np.isfinite(s) or s == 0.0:
                # If feature has zero variance on train, it carries no signal -> set to 0
                pdf[c] = 0.0
            else:
                pdf[c] = (x - m) / s
        return pdf

    meta = ddf._meta.copy()
    for c in cols:
        meta[c] = pd.Series(dtype="float64")  # ensure stable output schema

    return ddf.map_partitions(_apply, meta=meta)


def main() -> None:
    """Run end-to-end dataset preparation and write train/test artifacts."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ddf = dd.read_parquet(args.data)

    if args.target_col not in ddf.columns:
        raise SystemExit(f"Target column not found: {args.target_col}")

    # drop NA target rows (can't train supervised model without labels)
    ddf = ddf.dropna(subset=[args.target_col])

    # encode target
    ddf, target_mapping = encode_target(ddf, args.target_col)

    # drop leakage columns
    cols_all = list(ddf.columns)
    leaks = leak_columns(cols_all, target_col=args.target_col)
    if leaks:
        ddf = ddf.drop(columns=leaks, errors="ignore")

    # build feature set
    id_present = args.id_col in ddf.columns
    exclude_for_features = {args.target_col, "target"}
    feature_cols = [c for c in ddf.columns if c not in exclude_for_features]

    # restrict to the columns we actually need downstream
    base_cols = ([args.id_col] if id_present else []) + feature_cols + ["target"]
    ddf_base = ddf[base_cols]

    # split with Dask (random_split is approximate by partition; good enough for most ML baselines)
    train_frac = 1.0 - float(args.test_frac)
    train_ddf, test_ddf = ddf_base.random_split([train_frac, float(args.test_frac)], random_state=args.seed)

    # encode object/string features -> numeric codes (as float64), fitted on TRAIN only
    train_ddf, test_ddf = encode_object_features_train_test(
        train_ddf,
        test_ddf,
        exclude=[args.id_col] if id_present else [],
    )

    # coerce all features (except id) to float64 (kills 'NE' etc.)
    scale_candidates = [c for c in feature_cols if (not id_present or c != args.id_col)]
    train_ddf = coerce_numeric_features(train_ddf, scale_candidates)
    test_ddf = coerce_numeric_features(test_ddf, scale_candidates)

    # drop features that are all-NaN (decide on TRAIN only, then apply to both)
    kept_scale = drop_all_nan_columns(train_ddf, scale_candidates)
    final_feature_cols = ([args.id_col] if id_present else []) + kept_scale

    # final tables: features + target
    train_final = train_ddf[final_feature_cols + ["target"]]
    test_final = test_ddf[final_feature_cols + ["target"]]

    # z-score scaling (fit on TRAIN only)
    mu, sd = zscore_fit_stats(train_final, kept_scale)
    train_final = zscore_apply(train_final, kept_scale, mu, sd)
    test_final = zscore_apply(test_final, kept_scale, mu, sd)

    # write artifacts
    train_path = outdir / "train.parquet"
    test_path = outdir / "test.parquet"

    train_final.to_parquet(train_path, write_index=False)
    test_final.to_parquet(test_path, write_index=False)

    (outdir / "target_mapping.json").write_text(
        json.dumps(target_mapping, indent=2),
        encoding="utf-8",
    )

    (outdir / "feature_cols.json").write_text(
        json.dumps(final_feature_cols, indent=2),
        encoding="utf-8",
    )

    n_train = int(train_final.shape[0].compute())
    n_test = int(test_final.shape[0].compute())

    print(f"Wrote: {train_path} (rows={n_train})")
    print(f"Wrote: {test_path} (rows={n_test})")
    print(f"Leakage columns removed: {len(leaks)}")
    print(f"Final feature cols (excluding target): {len(final_feature_cols)}")


if __name__ == "__main__":
    main()