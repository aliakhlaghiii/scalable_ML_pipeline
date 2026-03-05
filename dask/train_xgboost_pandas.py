# train_xgboost_pandas.py
"""
train_xgboost_pandas.py

Train and evaluate an XGBoost classifier using a pandas workflow.

Task 5 (pandas workflow):
- Load the same train/test parquet produced by the Dask preprocessing pipeline
- Train XGBoost with minimal required multiclass params
- Metrics: Accuracy, Precision/Recall/F1, AUC-ROC (OVR with fallback)
- Runtime: train + inference
- Save: model.json, metrics.json, benchmark_report.json, predictions_test.csv
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pandas training workflow."""
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Train parquet dir")
    p.add_argument("--test", required=True, help="Test parquet dir")
    p.add_argument("--outdir", default="outputs/xgb_pandas")
    p.add_argument("--target-col", default="target")
    p.add_argument("--id-col", default="id")
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


def auc_multiclass_ovr(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Compute multiclass AUC-ROC (OVR) with a robust fallback."""
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
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

    train = pd.read_parquet(args.train)
    test = pd.read_parquet(args.test)

    if args.target_col not in train.columns:
        raise SystemExit(f"Target column not found in train: {args.target_col}")

    feats = feature_cols(list(train.columns), target_col=args.target_col, id_col=args.id_col)

    X_train = train[feats].to_numpy()
    y_train = train[args.target_col].to_numpy()

    X_test = test[feats].to_numpy()
    y_test = test[args.target_col].to_numpy()

    n_classes = int(pd.Series(y_train).nunique())

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "seed": int(args.seed),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=int(args.num_round))
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    proba = booster.predict(dtest)
    infer_seconds = time.perf_counter() - t1

    y_pred = proba.argmax(axis=1)

    metrics: Dict[str, Any] = {
        "n_classes": int(n_classes),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feats)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        **prf_scores(y_test, y_pred),
        **auc_multiclass_ovr(y_test, proba, n_classes),
    }

    model_path = outdir / "model.json"
    booster.save_model(model_path.as_posix())

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bench = {
        "workflow": "pandas_xgboost",
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
    }
    (outdir / "benchmark_report.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if args.id_col in test.columns:
        pred_df.insert(0, "id", test[args.id_col].astype(str).to_numpy())
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {outdir / 'metrics.json'}")
    print(f"Wrote: {outdir / 'benchmark_report.json'}")
    print(f"Wrote: {outdir / 'predictions_test.csv'}")


if __name__ == "__main__":
    main()