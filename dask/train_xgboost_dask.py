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

    out: Dict[str, Any] = {
        "auc_macro_ovr": None,
        "auc_weighted_ovr": None,
        "auc_per_class": {},
    }

    try:
        out["auc_macro_ovr"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
        out["auc_weighted_ovr"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
        )
        return out
    except Exception:
        # sklearn can fail when some classes are missing in the test set
        pass

    # fallback: compute one-vs-rest AUC manually
    y_true = y_true.astype(int)
    present = set(np.unique(y_true).tolist())

    aucs = []
    weights = []

    for k in range(n_classes):

        # skip classes not present in test labels
        if k not in present:
            continue

        y_bin = (y_true == k).astype(int)

        # skip degenerate cases (only one label)
        if y_bin.min() == y_bin.max():
            continue

        try:
            a = float(roc_auc_score(y_bin, proba[:, k]))
            out["auc_per_class"][str(k)] = a
            aucs.append(a)
            weights.append(int(y_bin.sum()))
        except Exception:
            continue

    if aucs:
        out["auc_macro_ovr"] = float(np.mean(aucs))
        out["auc_weighted_ovr"] = float(
            np.average(aucs, weights=weights)
        ) if sum(weights) > 0 else float(np.mean(aucs))

    return out


def main() -> None:
    """Train XGBoost with Dask, evaluate performance, and save artifacts."""

    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Start local distributed Dask cluster
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        dashboard_address=None,  # disable dashboard to avoid port conflicts
    )

    client = Client(cluster)

    # Load train/test datasets
    dtrain = dd.read_parquet(args.train).repartition(
        npartitions=max(1, args.n_workers)
    )  # ensure partitions match worker count

    dtest = dd.read_parquet(args.test).repartition(
        npartitions=max(1, args.n_workers)
    )

    if args.target_col not in dtrain.columns or args.target_col not in dtest.columns:
        raise SystemExit(f"Missing target column '{args.target_col}' in train/test.")

    feats = feature_cols(list(dtrain.columns), args.target_col, args.id_col)

    # Convert Dask DataFrame
    X_train = dtrain[feats].to_dask_array(lengths=True)
    y_train = dtrain[args.target_col].astype("int64").to_dask_array(lengths=True)

    X_test = dtest[feats].to_dask_array(lengths=True)
    y_test = dtest[args.target_col].astype("int64").to_dask_array(lengths=True)

    # Compute number of classes from both datasets
    n_classes = int(
        dd.concat([dtrain[args.target_col], dtest[args.target_col]])
        .nunique()
        .compute()
    )

    # Minimal parameters for multiclass classification
    params = {
        "objective": "multi:softprob",  # return class probabilities
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "seed": args.seed,
    }

    # Dask-aware data matrix
    dtrain_mat = dxgb.DaskDMatrix(client, X_train, y_train)

    # Training
    t0 = time.perf_counter()

    out = dxgb.train(
        client,
        params,
        dtrain_mat,
        num_boost_round=args.num_round,
    )

    train_seconds = time.perf_counter() - t0

    booster: xgb.Booster = out["booster"]

    # Inference
    t1 = time.perf_counter()

    proba_d = dxgb.predict(
        client,
        booster,
        X_test,
    )  # returns Dask array

    proba = proba_d.compute()  # collect predictions locally
    infer_seconds = time.perf_counter() - t1

    y_true = y_test.compute()
    y_pred = proba.argmax(axis=1)

    # Metrics
    metrics: Dict[str, Any] = {
        "n_classes": int(n_classes),
        "n_train": int(dtrain.shape[0].compute()),
        "n_test": int(dtest.shape[0].compute()),
        "n_features": int(len(feats)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **prf_scores(y_true, y_pred),
        **auc_multiclass_ovr(y_true, proba, n_classes),
    }

    # Save model
    model_path = outdir / "model.json"
    booster.save_model(model_path.as_posix())

    # Save metrics
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save benchmark info
    bench = {
        "workflow": "dask_xgboost",
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }

    (outdir / "benchmark_report.json").write_text(
        json.dumps(bench, indent=2),
        encoding="utf-8",
    )

    # Save predictions
    test_pdf = (
        dtest[[args.id_col]].compute()
        if args.id_col in dtest.columns
        else None
    )

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    # include patient id if available
    if test_pdf is not None:
        pred_df.insert(
            0,
            "id",
            test_pdf[args.id_col].astype(str).to_numpy(),
        )

    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {outdir / 'metrics.json'}")
    print(f"Wrote: {outdir / 'benchmark_report.json'}")
    print(f"Wrote: {outdir / 'predictions_test.csv'}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()