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
    p.add_argument("--num-round", type=int, default=100)  # keep simple / default-ish baseline
    return p.parse_args()


def feature_cols(cols: List[str], target_col: str, id_col: str) -> List[str]:
    """Return feature columns excluding target and id."""
    return [c for c in cols if c not in (target_col, id_col)]


def prf_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro + weighted precision/recall/F1."""
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
    """
    Compute multiclass AUC-ROC using one-vs-rest (OVR).

    If sklearn multiclass AUC fails (often due to missing classes in the test split),
    fall back to per-class OVR AUC for computable classes and average.
    """
    out: Dict[str, Any] = {"auc_macro_ovr": None, "auc_weighted_ovr": None, "auc_per_class": {}}
    try:
        out["auc_macro_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        out["auc_weighted_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
        return out
    except Exception:
        # This commonly happens when some classes are absent from y_true in the test set.
        pass

    # fallback per-class for classes that exist in test with both pos/neg
    present = set(np.unique(y_true).tolist())
    aucs, weights = [], []
    for k in range(n_classes):
        if k not in present:
            continue
        y_bin = (y_true == k).astype(int)

        # If all labels are the same, AUC is undefined.
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
        out["auc_weighted_ovr"] = float(np.average(aucs, weights=weights)) if sum(weights) > 0 else float(np.mean(aucs))
    return out


def main() -> None:
    """Run pandas-based XGBoost training, evaluation, and artifact export."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # pandas loads everything into memory (this is the point of the baseline comparison)
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)

    feats = feature_cols(list(train_df.columns), args.target_col, args.id_col)
    X_train = train_df[feats]
    y_train = train_df[args.target_col].astype(int).to_numpy()
    X_test = test_df[feats]
    y_test = test_df[args.target_col].astype(int).to_numpy()

    # Using both train and test to infer number of classes is robust to missing classes in either split.
    n_classes = int(pd.Series(np.concatenate([y_train, y_test])).nunique())

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_estimators=args.num_round,
        # Everything else left as default to keep this a "baseline-ish" run.
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    infer_seconds = time.perf_counter() - t1

    y_pred = proba.argmax(axis=1)

    metrics: Dict[str, Any] = {
        "n_classes": int(n_classes),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feats)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        **prf_scores(y_test, y_pred),
        **auc_multiclass_ovr(y_test, proba, n_classes),
    }

    # Save model as JSON text (portable, easy to diff in a repo)
    booster = clf.get_booster()
    (outdir / "model.json").write_text(
        booster.save_raw(raw_format="json").decode("utf-8"),
        encoding="utf-8",
    )

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (outdir / "benchmark_report.json").write_text(
        json.dumps(
            {"workflow": "pandas_xgboost", "train_seconds": train_seconds, "infer_seconds": infer_seconds},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save predictions (optionally with id for traceability)
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if args.id_col in test_df.columns:
        pred_df.insert(0, "id", test_df[args.id_col].astype(str).to_numpy())
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    print(f"Wrote: {outdir / 'benchmark_report.json'}")
    print(f"Wrote: {outdir / 'metrics.json'}")
    print(f"Wrote: {outdir / 'predictions_test.csv'}")
    print(f"Wrote: {outdir / 'model.json'}")


if __name__ == "__main__":
    main()