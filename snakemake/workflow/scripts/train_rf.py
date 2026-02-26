from snakemake.script import snakemake  # type: ignore

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    default_rf_param_grid,
    fit_gridsearch_and_evaluate,
    make_inner_cv,
    make_rf_pipeline,
)

train_df = pd.read_csv(snakemake.input.train)
test_df = pd.read_csv(snakemake.input.test)

if "target" not in train_df.columns or "target" not in test_df.columns:
    raise ValueError("Input CSVs must contain a 'target' column.")

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

X_splits = {"main": (X_train, X_test)}

inner_cv = make_inner_cv(n_splits=5, random_state=42)
param_grid = default_rf_param_grid()

threads = getattr(snakemake, "threads", 1) or 1
threads = int(threads) if int(threads) > 0 else 1

results, fitted_models = fit_gridsearch_and_evaluate(
    X_splits=X_splits,
    y_train=y_train,
    y_test=y_test,
    make_pipeline_func=lambda: make_rf_pipeline(random_state=42),
    param_grid=param_grid,
    inner_cv=inner_cv,
    scoring="roc_auc",
    n_jobs=threads,   
    verbose=True,
)

best_model = fitted_models["main"]
metrics = results["main"]

Path(snakemake.output.model).parent.mkdir(parents=True, exist_ok=True)
Path(snakemake.output.metrics).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, snakemake.output.model)

with open(snakemake.output.metrics, "w") as f:
    json.dump(metrics, f, indent=2, default=str)