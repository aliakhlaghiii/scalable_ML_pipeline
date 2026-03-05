"""
prepare_ml_dataset_pandas.py

Pandas baseline for Task 4 (Further Preprocessing).

This mirrors prepare_ml_dataset.py but runs with a traditional pandas workflow.

Steps:
- Read the preprocessed parquet dataset (from Task 2 pandas or dask output)
- Ensure unique column names
- Drop rows with missing target
- Encode target to integer labels
- Remove likely leakage columns (heuristic)
- Train/test split (reproducible)
- Encode object/string features (fit on TRAIN only)
- Coerce numeric features and drop all-NaN features (based on TRAIN)
- Z-score scaling (fit on TRAIN only, apply to TEST)

Outputs:
  <outdir>/train.parquet
  <outdir>/test.parquet
  <outdir>/target_mapping.json
  <outdir>/feature_cols.json
  """

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_LEAK_PAT = re.compile(r"^(surv_|post_cart_)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet dataset path (Task 2 output)")
    p.add_argument("--outdir", default="outputs/ml_pandas", help="Output folder")
    p.add_argument("--target-col", default="surv_bestresponse_car.factor", help="Target column")
    p.add_argument("--id-col", default="id", help="ID column")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
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


def leak_columns(cols: List[str], target_col: str) -> List[str]:
    """Detect columns that could leak label information (heuristic)."""
    leak: List[str] = []
    for c in cols:
        if c == target_col or c == "target":
            continue
        if _LEAK_PAT.search(str(c)):
            leak.append(c)
    for c in ("bestresponse", "response", "survival"):
        if c in cols and c not in leak:
            leak.append(c)
    return leak


def encode_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Encode a categorical target column to integer labels."""
    uniq = sorted(df[target_col].dropna().astype(str).unique().tolist())
    mapping = {u: i for i, u in enumerate(uniq)}
    out = df.copy()
    out["target"] = out[target_col].astype(str).map(mapping).astype("int64")
    return out, mapping


def fit_object_mappings(train: pd.DataFrame, obj_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """Build per-column value->code mappings from TRAIN only."""
    mappings: Dict[str, Dict[str, int]] = {}
    for c in obj_cols:
        uniq = sorted(train[c].dropna().astype(str).unique().tolist())
        mappings[c] = {v: i for i, v in enumerate(uniq)}
    return mappings


def apply_object_mappings(df: pd.DataFrame, obj_cols: List[str], mappings: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Apply train-fitted mappings; unseen values become NaN."""
    out = df.copy()
    for c in obj_cols:
        out[c] = out[c].astype(str).map(mappings[c]).astype("float64")
    return out


def main() -> None:
    """Run end-to-end pandas dataset preparation and write artifacts."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    df = ensure_unique_columns_pdf(df)

    if args.target_col not in df.columns:
        raise SystemExit(f"Target column not found: {args.target_col}")

    df = df.dropna(subset=[args.target_col]).copy()

    df, target_mapping = encode_target(df, args.target_col)

    leaks = leak_columns(list(df.columns), target_col=args.target_col)
    if leaks:
        df = df.drop(columns=leaks, errors="ignore")

    df = ensure_unique_columns_pdf(df)

    id_present = args.id_col in df.columns

    exclude_for_features = {args.target_col, "target"}
    feature_cols = [c for c in df.columns if c not in exclude_for_features]
    if id_present:
        feature_cols = [c for c in feature_cols if c != args.id_col]

    base_cols = ([args.id_col] if id_present else []) + feature_cols + ["target"]
    base_cols = [c for c in base_cols if c in df.columns]
    df_base = df[base_cols].copy()
    df_base = ensure_unique_columns_pdf(df_base)

    rng = np.random.default_rng(int(args.seed))
    n = len(df_base)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(float(args.test_frac) * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train = df_base.iloc[train_idx].copy()
    test = df_base.iloc[test_idx].copy()

    obj_cols = [c for c in feature_cols if c in train.columns and train[c].dtype == object]
    mappings = fit_object_mappings(train, obj_cols) if obj_cols else {}
    if obj_cols:
        train = apply_object_mappings(train, obj_cols, mappings)
        test = apply_object_mappings(test, obj_cols, mappings)

    scale_candidates = [c for c in feature_cols if c in train.columns]
    for c in scale_candidates:
        train[c] = pd.to_numeric(train[c], errors="coerce").astype("float64")
        test[c] = pd.to_numeric(test[c], errors="coerce").astype("float64")

    kept_scale = [c for c in scale_candidates if not train[c].isna().all()]
    final_feature_cols = ([args.id_col] if id_present else []) + kept_scale

    train_final = train[final_feature_cols + ["target"]].copy()
    test_final = test[final_feature_cols + ["target"]].copy()

    mu = train_final[kept_scale].mean(numeric_only=True)
    sd = train_final[kept_scale].std(ddof=0, numeric_only=True).replace(0.0, np.nan)

    for c in kept_scale:
        m = float(mu[c]) if pd.notna(mu[c]) else 0.0
        s = float(sd[c]) if pd.notna(sd[c]) else np.nan
        train_final[c] = pd.to_numeric(train_final[c], errors="coerce").fillna(m)
        test_final[c] = pd.to_numeric(test_final[c], errors="coerce").fillna(m)
        if not np.isfinite(s) or s == 0.0:
            train_final[c] = 0.0
            test_final[c] = 0.0
        else:
            train_final[c] = (train_final[c] - m) / s
            test_final[c] = (test_final[c] - m) / s

    train_path = outdir / "train.parquet"
    test_path = outdir / "test.parquet"

    train_final.to_parquet(train_path.as_posix(), index=False)
    test_final.to_parquet(test_path.as_posix(), index=False)

    (outdir / "target_mapping.json").write_text(json.dumps(target_mapping, indent=2), encoding="utf-8")
    (outdir / "feature_cols.json").write_text(json.dumps(final_feature_cols, indent=2), encoding="utf-8")

    print(f"Wrote: {train_path} (rows={len(train_final)})")
    print(f"Wrote: {test_path} (rows={len(test_final)})")
    print(f"Leakage columns removed: {len(leaks)}")
    print(f"Final feature cols (excluding target): {len(final_feature_cols)}")


if __name__ == "__main__":
    main()