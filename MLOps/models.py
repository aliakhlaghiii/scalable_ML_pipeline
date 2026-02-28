"""
models.py

Model pipelines and training utilities for SVM and Random Forest with GridSearchCV.
"""

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


def make_svm_pipeline(random_state=42):
    """
    Create an SVM (RBF) pipeline with median imputation and standard scaling.

    Parameters
    ----------
    random_state : int
        Random seed used by SVC.

    Returns
    -------
    sklearn.pipeline.Pipeline
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


def make_rf_pipeline(random_state=42):
    """
    Create a Random Forest pipeline with median imputation and standard scaling.

    Parameters
    ----------
    random_state : int
        Random seed for the forest.

    Returns
    -------
    sklearn.pipeline.Pipeline
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


def default_svm_param_grid():
    """
    Default hyperparameter grid for the SVM pipeline.

    Returns
    -------
    dict
    """
    return {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01, 0.1],
    }


def default_rf_param_grid():
    """
    Default hyperparameter grid for the Random Forest pipeline.

    Returns
    -------
    dict
    """
    return {
        "rf__max_depth": [2, 3],
        "rf__min_samples_leaf": [2, 4],
        "rf__max_features": ["sqrt", 0.5],
    }


def make_inner_cv(n_splits=5, random_state=42):
    """
    Create StratifiedKFold for inner cross-validation.

    Parameters
    ----------
    n_splits : int
    random_state : int

    Returns
    -------
    sklearn.model_selection.StratifiedKFold
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def fit_gridsearch_and_evaluate(
    X_splits,
    y_train,
    y_test,
    make_pipeline_func,
    param_grid,
    inner_cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=True,
):
    """
    Fit GridSearchCV for each scenario in X_splits and evaluate on the test split.

    Parameters
    ----------
    X_splits : dict
        Mapping: scenario_name -> (X_train, X_test)
    y_train : array-like
    y_test : array-like
    make_pipeline_func : callable
        Returns a fresh sklearn Pipeline.
    param_grid : dict
    inner_cv : sklearn CV splitter
    scoring : str
    n_jobs : int
    verbose : bool

    Returns
    -------
    (dict, dict)
        results       : scenario_name -> metrics dict
        fitted_models : scenario_name -> best_estimator
    """
    results = {}
    fitted_models = {}

    for name, (X_train, X_test) in X_splits.items():
        if verbose:
            print("\nFitting model with GridSearchCV for scenario: {}".format(name))

        if len(X_train) != len(y_train):
            raise ValueError(
                f"[{name}] X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}"
            )
        if len(X_test) != len(y_test):
            raise ValueError(
                f"[{name}] X_test and y_test length mismatch: "
                f"{len(X_test)} vs {len(y_test)}"
            )

        grid = GridSearchCV(
            estimator=make_pipeline_func(),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_

        y_pred  = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1]
        cm      = confusion_matrix(y_test, y_pred)

        res = {
            "best_params":        grid.best_params_,
            "cv_best_score":      grid.best_score_,
            "accuracy":           accuracy_score(y_test, y_pred),
            "balanced_accuracy":  balanced_accuracy_score(y_test, y_pred),
            "f1":                 f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":            roc_auc_score(y_test, y_proba),
            "confusion_matrix":   cm,
            "report":             classification_report(y_test, y_pred, zero_division=0),
        }

        results[name]       = res
        fitted_models[name] = best_clf

        if verbose:
            print("=" * 60)
            print("Scenario:", name)
            print("Best params:", res["best_params"])
            print("CV best {}: {:.3f}".format(scoring, res["cv_best_score"]))
            print(
                "Test Acc: {:.3f} | BalAcc: {:.3f} | F1: {:.3f} | "
                "Test ROC-AUC: {:.3f}".format(
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
