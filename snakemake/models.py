"""
models.py

Model pipelines and training utilities for SVM and Random Forest with GridSearchCV.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def make_inner_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_svm_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        ]
    )


def make_rf_pipeline(random_state: int = 42) -> Pipeline:
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
                ),
            ),
        ]
    )


def default_svm_param_grid() -> Dict[str, Any]:
    return {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01, 0.1],
        "svm__kernel": ["rbf"],
    }


def default_rf_param_grid() -> Dict[str, Any]:
    return {
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2"],
    }


def fit_gridsearch_and_evaluate(
    X_splits: Dict[str, Tuple[Any, Any]],
    y_train,
    y_test,
    make_pipeline_func: Callable[[], Pipeline],
    param_grid: Dict[str, Any],
    inner_cv,
    scoring: str = "roc_auc",
    n_jobs: int = 1,
    verbose: bool = True,
):
    """
    Fit GridSearchCV for each scenario in X_splits and evaluate on the test split.

    Parameters
    ----------
    X_splits : dict
        scenario_name -> (X_train, X_test)
    y_train, y_test :
        labels for train/test
    make_pipeline_func :
        function returning a fresh sklearn Pipeline
    param_grid :
        GridSearchCV parameter grid
    inner_cv :
        CV splitter
    scoring :
        scoring metric for GridSearchCV
    n_jobs :
        parallelism inside GridSearchCV
    verbose :
        print progress

    Returns
    -------
    (results, fitted_models)
    """
    if n_jobs is None or int(n_jobs) < 1:
        n_jobs = 1

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
            refit=True,
            n_jobs=n_jobs,
        )

        # sanity checks to prevent silent label-feature misalignment
        if len(X_train) != len(y_train):
            raise ValueError(f"[{name}] X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
        if len(X_test) != len(y_test):
            raise ValueError(f"[{name}] X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")

        if hasattr(X_train, "index") and hasattr(y_train, "index"):
            if not X_train.index.equals(y_train.index):
                raise ValueError(f"[{name}] X_train and y_train indices are not aligned.")
        if hasattr(X_test, "index") and hasattr(y_test, "index"):
            if not X_test.index.equals(y_test.index):
                raise ValueError(f"[{name}] X_test and y_test indices are not aligned.")

        grid.fit(X_train, y_train)

        best_clf: Pipeline = grid.best_estimator_
        y_pred = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)

        res = {
            "best_params": grid.best_params_,
            "cv_best_score": float(grid.best_score_),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": cm.tolist(),
            "report": classification_report(y_test, y_pred, zero_division=0),
        }

        results[name] = res
        fitted_models[name] = best_clf

        if verbose:
            print("=" * 60)
            print("Scenario: {}".format(name))
            print("Best params:", res["best_params"])
            print("CV best {}: {:.3f}".format(scoring, res["cv_best_score"]))
            print(
                "Test Acc: {:.3f} | BalAcc: {:.3f} | F1: {:.3f} | Test ROC-AUC: {:.3f}".format(
                    res["accuracy"], res["balanced_accuracy"], res["f1"], res["roc_auc"]
                )
            )
            print("Confusion matrix [[TN FP],[FN TP]]:")
            print(cm)
            print("\nClassification report:")
            print(res["report"])

    return results, fitted_models