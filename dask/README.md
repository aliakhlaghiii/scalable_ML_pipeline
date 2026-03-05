# Dask Radiomics Pipeline

A six-task assignment comparing **Dask** and **pandas** workflows for radiomics ML from raw Excel files to a trained XGBoost classifier.

## Setup

``` bash
pip install "dask[dataframe,distributed]" pandas numpy xgboost scikit-learn pyarrow openpyxl pyyaml matplotlib
```

Copy and fill in the config:

``` bash
cp config_example.yaml config.yaml
```

``` yaml
paths:
  data_folder: "/path/to/modelling_data"
  clinical_data: "/path/to/clinical.xlsx"
```

## Usage

Run everything from the notebook:

``` bash
jupyter notebook dask_assignment_research_notebook.ipynb
```

Or run each task individually:

``` bash
# Task 1 — load benchmark
python load_and_benchmark.py --config config.yaml --which DELTA

# Task 2 — preprocess
python preprocess_delayed.py --config config.yaml --which DELTA --max-missing-frac 0.3 --out outputs/preprocessed_delta.parquet

# Task 3 — EDA
python eda_delayed.py --data outputs/preprocessed_delta.parquet --outdir outputs/eda

# Task 4 — ML dataset
python prepare_ml_dataset.py --data outputs/preprocessed_delta.parquet --outdir outputs/ml

# Task 5 — train (Dask)
python train_xgboost_dask.py --train outputs/ml/train.parquet --test outputs/ml/test.parquet --outdir outputs/xgb

# Task 5 — train (pandas baseline)
python train_xgboost_pandas.py --train outputs/ml/train.parquet --test outputs/ml/test.parquet --outdir outputs/xgb_pandas
```

## File Overview

| File | Task | Description |
|----|----|----|
| `load_and_benchmark.py` | 1 | Load radiomics; benchmark pandas vs Dask |
| `preprocess_delayed.py` | 2 | Merge, drop sparse cols, impute — Dask |
| `preprocess_pandas.py` | 2 | Same pipeline — pandas baseline |
| `eda_delayed.py` | 3 | Summary stats, distributions, group overviews |
| `prepare_ml_dataset.py` | 4 | Encode, split, scale — Dask |
| `prepare_ml_dataset_pandas.py` | 4 | Same pipeline — pandas baseline |
| `train_xgboost_dask.py` | 5 | XGBoost with `xgboost.dask` |
| `train_xgboost_pandas.py` | 5 | XGBoost — pandas baseline |
| `dask_assignment_research_notebook.ipynb` | 1–6 | Orchestration + Task 6 benchmark |

## References

-   [fenna/BFVM24PROGRAM6 — Dask examples](https://github.com/fenna/BFVM24PROGRAM6/tree/main/dask)
-   [Dask Documentation](https://docs.dask.org/en/stable/)
-   [Dask Best Practices](https://docs.dask.org/en/stable/best-practices.html)
-   [Dask Delayed Best Practices](https://docs.dask.org/en/latest/delayed-best-practices.html)
-   [Dask DataFrame Best Practices](https://docs.dask.org/en/stable/dataframe-best-practices.html)
-   [Dask Distributed — Managing Computation](https://distributed.dask.org/en/stable/manage-computation.html)
-   [XGBoost + Dask Tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/dask.html)
-   Rocklin, M. (2015). *Dask: Parallel Computation with Blocked algorithms and Task Scheduling.* SciPy 2015. [PDF](https://proceedings.scipy.org/articles/Majora-7b98e3ed-013.pdf)

This project was developed with assistance from ChatGPT for code refinement, debugging, and documentation support.
