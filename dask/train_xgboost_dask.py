"""
train_xgboost_dask.py

Train and evaluate an XGBoost classifier using Dask-distributed data.

Pipeline:
- Load train/test parquet datasets produced by the preprocessing step
- Train a multiclass XGBoost model using xgboost.dask
- Run distributed inference
- Compute evaluation metrics
- Save model, metrics, predictions, and benchmark timings

Outputs:
  outputs/xgb/model.json
  outputs/xgb/metrics.json
  outputs/xgb/predictions_test.csv
  outputs/xgb/benchmark_report.json
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import xgboost as xgb
from xgboost import dask as dxgb

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Train parquet directory")
    p.add_argument("--test", required=True, help="Test parquet directory")
    p.add_argument("--outdir", default="outputs/xgb", help="Output directory")
    p.add_argument("--target-col", default="target")
    p.add_argument("--id-col", default="id")
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-round", type=int, default=100)
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


def ensure_unique_ddf_columns(ddf: dd.DataFrame) -> dd.DataFrame:
    """Ensure Dask DataFrame has unique column names."""
    cols = list(map(str, ddf.columns))
    if len(cols) == len(set(cols)):
        return ddf
    ddf = ddf.copy()
    ddf.columns = make_unique_columns(cols)
    return ddf


def feature_cols(cols: List[str], target_col: str, id_col: str) -> List[str]:
    """Return feature columns excluding target and id."""
    return [c for c in cols if c not in (target_col, id_col)]


def prf_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall and F1 scores (macro and weighted)."""
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
    }


def auc_multiclass_ovr(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
) -> Dict[str, Any]:
    """
    Compute multiclass AUC-ROC using one-vs-rest strategy.

    If sklearn fails (for example when some classes are missing
    in the test split), the function falls back to computing AUC
    per class and averaging the results.
    """
    try:
        auc = roc_auc_score(
            y_true,
            proba,
            multi_class="ovr",
            average="macro",
        )
        return {"auc_ovr_macro": float(auc), "auc_fallback_used": False}
    except Exception:
        aucs: List[float] = []
        for k in range(int(n_classes)):
            yk = (y_true == k).astype(int)
            if yk.sum() == 0 or yk.sum() == len(yk):
                continue
            try:
                aucs.append(float(roc_auc_score(yk, proba[:, k])))
            except Exception:
                continue
        auc_mean = float(np.mean(aucs)) if aucs else float("nan")
        return {"auc_ovr_macro": auc_mean, "auc_fallback_used": True, "auc_classes_used": len(aucs)}


def main() -> None:
    """Train model, evaluate and write artifacts."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=int(args.n_workers),
        threads_per_worker=int(args.threads_per_worker),
        processes=False,
        dashboard_address=None,
    )
    client = Client(cluster)

    dtrain = dd.read_parquet(args.train)
    dtest = dd.read_parquet(args.test)

    dtrain = ensure_unique_ddf_columns(dtrain)
    dtest = ensure_unique_ddf_columns(dtest)

    if args.target_col not in dtrain.columns:
        raise SystemExit(f"Target column not found in train: {args.target_col}")

    feats = feature_cols(list(dtrain.columns), target_col=args.target_col, id_col=args.id_col)
    feats = list(dict.fromkeys(feats))

    train_arr = dtrain[feats + [args.target_col]].to_dask_array(lengths=True)
    X_train = train_arr[:, :-1].astype(np.float32)
    y_train = train_arr[:, -1].astype(np.int64)

    test_arr = dtest[feats + [args.target_col]].to_dask_array(lengths=True)
    X_test = test_arr[:, :-1].astype(np.float32)
    y_test = test_arr[:, -1].astype(np.int64)

    X_train, y_train, X_test, y_test = client.persist([X_train, y_train, X_test, y_test])

    n_classes = int(dtrain[args.target_col].nunique().compute())
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "seed": int(args.seed),
    }

    dtrain_mat = dxgb.DaskDMatrix(client, X_train, y_train)

    t0 = time.perf_counter()
    out = dxgb.train(
        client,
        params,
        dtrain_mat,
        num_boost_round=int(args.num_round),
    )
    train_seconds = time.perf_counter() - t0

    booster: xgb.Booster = out["booster"]

    t1 = time.perf_counter()
    dtest_mat = dxgb.DaskDMatrix(client, X_test)
    proba_d = dxgb.predict(client, booster, dtest_mat)
    proba = proba_d.compute()
    infer_seconds = time.perf_counter() - t1

    y_true = y_test.compute()
    y_pred = proba.argmax(axis=1)

    metrics: Dict[str, Any] = {
        "n_classes": int(n_classes),
        "n_train": int(dtrain.shape[0].compute()),
        "n_test": int(dtest.shape[0].compute()),
        "n_features": int(len(feats)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **prf_scores(y_true, y_pred),
        **auc_multiclass_ovr(y_true, proba, n_classes),
    }

    model_path = outdir / "model.json"
    booster.save_model(model_path.as_posix())

    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    bench = {
        "workflow": "dask_xgboost",
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }

    (outdir / "benchmark_report.json").write_text(
        json.dumps(bench, indent=2),
        encoding="utf-8",
    )

    test_pdf = dtest[[args.id_col]].compute() if args.id_col in dtest.columns else None

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if test_pdf is not None:
        pred_df.insert(0, "id", test_pdf[args.id_col].astype(str).to_numpy())

    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {outdir / 'metrics.json'}")
    print(f"Wrote: {outdir / 'benchmark_report.json'}")
    print(f"Wrote: {outdir / 'predictions_test.csv'}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()