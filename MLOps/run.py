"""
run.py

End-to-end pipeline for delta radiomics + clinical response prediction.

Reproduces the full analysis from the notebook:
  1. Load radiomics data and compute delta features
  2. Load and clean clinical data
  3. Align cohort (intersection of patient IDs)
  4. Build four feature sets: clinical, clin+A, clin+B, clin+delta
  5. Stratified train/test split (80/20, seed=42)
  6. SVM (RBF) with inner 5-fold GridSearchCV
  7. Random Forest with inner 5-fold GridSearchCV
  8. Print metrics and save plots

Usage
-----
    python run.py
    python run.py --config path/to/config.yaml --outdir outputs --verbose
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split

from config.config import load_paths
from data_io import calculate_delta_radiomics
from preprocessing import clean_radiomics, load_and_clean_clinical, align_cohort, build_feature_sets
from models import (
    make_svm_pipeline,
    make_rf_pipeline,
    default_svm_param_grid,
    default_rf_param_grid,
    make_inner_cv,
    fit_gridsearch_and_evaluate,
)
from plots import (
    plot_metric_heatmap_from_results,
    plot_all_metric_heatmaps,
    plot_confusion_matrices_strip,
)


def main():
    parser = argparse.ArgumentParser(
        description="Delta radiomics + clinical response prediction pipeline."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--outdir", default=None, help="Directory to save plots (default: no saving)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed model fitting output"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading config from:", args.config)
    data_folder, clinical_path = load_paths(args.config)

    print("\n[1/7] Loading radiomics data...")
    delta_df, a_df, b_df = calculate_delta_radiomics(data_folder)
    print(f"  Raw shapes — delta: {delta_df.shape}, A: {a_df.shape}, B: {b_df.shape}")

    # ------------------------------------------------------------------
    # 2. Clean radiomics
    # ------------------------------------------------------------------
    print("\n[2/7] Cleaning radiomics DataFrames...")
    delta_df, a_df, b_df = clean_radiomics(delta_df, a_df, b_df)
    print(f"  After cleaning — delta: {delta_df.shape}, A: {a_df.shape}, B: {b_df.shape}")

    # ------------------------------------------------------------------
    # 3. Load and clean clinical data
    # ------------------------------------------------------------------
    print("\n[3/7] Loading and cleaning clinical data...")
    clinic = load_and_clean_clinical(clinical_path)
    print(f"  Clinical shape after cleaning: {clinic.shape}")

    # ------------------------------------------------------------------
    # 4. Align cohort
    # ------------------------------------------------------------------
    print("\n[4/7] Aligning cohort across all data sources...")
    clinic_c, delta_c, a_c, b_c = align_cohort(clinic, delta_df, a_df, b_df)

    # ------------------------------------------------------------------
    # 5. Build feature sets and label
    # ------------------------------------------------------------------
    print("\n[5/7] Building feature matrices...")
    y, feature_sets = build_feature_sets(clinic_c, delta_c, a_c, b_c)
    y_vec = y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Train / test split
    # ------------------------------------------------------------------
    print("\n[6/7] Splitting data (80/20 stratified, seed=42)...")
    train_idx, test_idx = train_test_split(
        np.arange(len(y_vec)),
        test_size=0.20,
        random_state=42,
        stratify=y_vec,
    )

    y_train = y_vec.iloc[train_idx]
    y_test  = y_vec.iloc[test_idx]

    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"  y_train balance: {y_train.value_counts().to_dict()}")
    print(f"  y_test  balance: {y_test.value_counts().to_dict()}")

    X_splits = {
        name: (X.iloc[train_idx].reset_index(drop=True),
               X.iloc[test_idx].reset_index(drop=True))
        for name, X in feature_sets.items()
    }

    # ------------------------------------------------------------------
    # 7. Model fitting
    # ------------------------------------------------------------------
    print("\n[7/7] Fitting models...")

    # Reduce CV folds if minority class is too small
    min_class = int(y_train.value_counts().min())
    safe_splits = max(2, min(5, min_class))
    if safe_splits < 5:
        print(f"  Reducing inner CV folds to {safe_splits} (min class size = {min_class})")

    inner_cv = make_inner_cv(n_splits=safe_splits, random_state=42)

    # SVM
    print("\n--- SVM ---")
    svm_results, _ = fit_gridsearch_and_evaluate(
        X_splits=X_splits,
        y_train=y_train,
        y_test=y_test,
        make_pipeline_func=make_svm_pipeline,
        param_grid=default_svm_param_grid(),
        inner_cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=args.verbose,
    )

    # Random Forest
    print("\n--- Random Forest ---")
    rf_results, _ = fit_gridsearch_and_evaluate(
        X_splits=X_splits,
        y_train=y_train,
        y_test=y_test,
        make_pipeline_func=make_rf_pipeline,
        param_grid=default_rf_param_grid(),
        inner_cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    feature_set_order = ["clinical", "clin+A", "clin+B", "clin+delta"]

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

        # Save metrics to JSON
        import json

        def _to_jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        metrics_out = {}
        for model_name, res in [("svm", svm_results), ("rf", rf_results)]:
            metrics_out[model_name] = {}
            for scenario, d in res.items():
                metrics_out[model_name][scenario] = {
                    k: _to_jsonable(v)
                    for k, v in d.items()
                    if k != "report"  # report is already a string
                }
                metrics_out[model_name][scenario]["report"] = d["report"]

        metrics_path = os.path.join(args.outdir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"  Saved: {metrics_path}")

        # ------------------------------------------------------------------
        # Save plots
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plots import plot_metric_heatmap_from_results as _hmr

        def _save(name):
            path = os.path.join(args.outdir, name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")

        for metric in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
            _hmr(svm_results, rf_results, metric=metric, feature_sets=feature_set_order)
            _save(f"heatmap_{metric}.png")

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for col, (model_name, res) in enumerate([("SVM", svm_results), ("RF", rf_results)]):
            for row, sc in enumerate(["clinical", "clin+delta"]):
                ax = axes[row, col]
                cm = res[sc]["confusion_matrix"]
                ax.imshow(cm)
                ax.set_title(f"{model_name}\n{sc}")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        plt.tight_layout()
        _save("confusion_matrices.png")

    else:
        plot_metric_heatmap_from_results(
            svm_results, rf_results, metric="f1",
            feature_sets=feature_set_order,
            title="Test-set F1 comparison (SVM vs RF)",
        )
        plot_all_metric_heatmaps(svm_results, rf_results, feature_sets=feature_set_order)
        plot_confusion_matrices_strip(svm_results, scenarios=["clinical", "clin+delta"], model_name="SVM")
        plot_confusion_matrices_strip(rf_results,  scenarios=["clinical", "clin+delta"], model_name="RF")

    print("\nDone.")


if __name__ == "__main__":
    main()
