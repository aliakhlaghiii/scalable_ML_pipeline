"""
models.py

Model pipelines and training utilities for SVM and Random Forest with GridSearchCV.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_svm_pipeline(random_state: int = 42) -> Pipeline:
    """
    Create an SVM (RBF) pipeline with median imputation and standard scaling.

    Parameters
    ----------
    random_state : int
        Random seed used by SVC (only affects probability estimation internals
        when probability=True).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline: imputer -> scaler -> SVC
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_rf_pipeline(random_state: int = 42) -> Pipeline:
    """
    Create a Random Forest pipeline with median imputation and standard scaling.

    Note
    ----
    StandardScaler is not required for tree models, but keeping it here preserves
    existing behavior and keeps pipelines consistent across model types.

    Parameters
    ----------
    random_state : int
        Random seed for the forest.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline: imputer -> scaler -> RandomForestClassifier
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def default_svm_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter grid for the SVM pipeline.

    Returns
    -------
    dict
        Grid for GridSearchCV.
    """
    return {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01, 0.1],
    }


def default_rf_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter grid for the Random Forest pipeline.

    Returns
    -------
    dict
        Grid for GridSearchCV.
    """
    return {
        "rf__max_depth": [2, 3],
        "rf__min_samples_leaf": [2, 4],
        "rf__max_features": ["sqrt", 0.5],
    }


def make_inner_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Create StratifiedKFold for inner cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for shuffling.

    Returns
    -------
    sklearn.model_selection.StratifiedKFold
        StratifiedKFold instance.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _safe_roc_auc(y_true, y_score) -> float:
    """Compute ROC-AUC, returning NaN (with warning) if it cannot be computed."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        warnings.warn(f"ROC-AUC could not be computed (likely single-class test set): {e}")
        return float("nan")


def fit_gridsearch_and_evaluate(
    X_splits: Dict[str, Tuple[Any, Any]],
    y_train: Any,
    y_test: Any,
    make_pipeline_func: Callable[[], Pipeline],
    param_grid: Dict[str, Any],
    inner_cv: Any,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: bool = True,
):
    """
    Fit GridSearchCV for each scenario in X_splits and evaluate on the test split.

    Parameters
    ----------
    X_splits : dict
        Mapping: scenario_name -> (X_train, X_test)
    y_train : array-like
        Training labels aligned with X_train for each scenario.
    y_test : array-like
        Test labels aligned with X_test for each scenario.
    make_pipeline_func : callable
        Function that returns a fresh sklearn Pipeline (e.g., make_svm_pipeline).
    param_grid : dict
        GridSearchCV parameter grid (pipeline-style keys like 'svm__C').
    inner_cv : sklearn CV splitter
        Cross-validation splitter for hyperparameter tuning on the training set.
    scoring : str
        GridSearchCV scoring metric (default: 'roc_auc').
    n_jobs : int
        Parallel jobs for GridSearchCV.
    verbose : bool
        If True, prints a summary per scenario.

    Returns
    -------
    (dict, dict)
        results: scenario_name -> metrics dict
        fitted_models: scenario_name -> best_estimator (Pipeline)
    """
    results: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Pipeline] = {}

    for name, (X_train, X_test) in X_splits.items():
        if verbose:
            print("\nFitting model with GridSearchCV for scenario: {}".format(name))

        base_clf = make_pipeline_func()

        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )

        if len(X_train) != len(y_train):
            raise ValueError(
                f"[{name}] X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
            )
        if len(X_test) != len(y_test):
            raise ValueError(
                f"[{name}] X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}"
            )

        if hasattr(X_train, "index") and hasattr(y_train, "index"):
            if not X_train.index.equals(y_train.index):
                raise ValueError(f"[{name}] X_train and y_train indices are not aligned.")
        if hasattr(X_test, "index") and hasattr(y_test, "index"):
            if not X_test.index.equals(y_test.index):
                raise ValueError(f"[{name}] X_test and y_test indices are not aligned.")

        grid.fit(X_train, y_train)
        best_clf: Pipeline = grid.best_estimator_

        y_pred = best_clf.predict(X_test)

        y_score = None
        if hasattr(best_clf, "predict_proba"):
            try:
                y_score = best_clf.predict_proba(X_test)[:, 1]
            except Exception:
                y_score = None
        if y_score is None and hasattr(best_clf, "decision_function"):
            try:
                y_score = best_clf.decision_function(X_test)
            except Exception:
                y_score = None

        cm = confusion_matrix(y_test, y_pred)

        cv_std = float(grid.cv_results_["std_test_score"][grid.best_index_])

        res: Dict[str, Any] = {
            "best_params": grid.best_params_,
            "cv_best_score": float(grid.best_score_),
            "cv_best_score_std": cv_std,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": _safe_roc_auc(y_test, y_score) if y_score is not None else float("nan"),
            "confusion_matrix": cm,
            "report": classification_report(y_test, y_pred, zero_division=0),
        }

        results[name] = res
        fitted_models[name] = best_clf

        if verbose:
            print("=" * 60)
            print("Scenario: {}".format(name))
            print("Best params:", res["best_params"])
            print(
                "CV best {}: {:.3f} ± {:.3f}".format(
                    scoring, res["cv_best_score"], res["cv_best_score_std"]
                )
            )
            print(
                "Test Acc: {:.3f} | BalAcc: {:.3f} | F1: {:.3f} | Test ROC-AUC: {:.3f}".format(
                    res["accuracy"],
                    res["balanced_accuracy"],
                    res["f1"],
                    res["roc_auc"],
                )
            )
            print("Confusion matrix [[TN FP],[FN TP]]:")
            print(cm)
            print("\nClassification report:")
            print(res["report"])

    return results, fitted_models