# Dask vs Pandas — Pipeline Report

## Project Overview

This project implements a radiomics ML pipeline over six tasks using both Dask and pandas, applied to a dataset of 31 patients (\~277 features after preprocessing). Each stage has a Dask-native implementation and a pandas baseline to enable direct runtime and memory comparisons.

------------------------------------------------------------------------

## Task 1 — Loading Benchmark

Per-patient radiomics features are loaded from Excel files using two approaches: a sequential pandas loop (`load_pandas`) and a parallelised Dask workflow using `dask.delayed` + `dd.from_delayed` (`load_dask`). A sample-derived `meta` DataFrame is used to enforce a consistent schema across all partitions.

| Workflow               | Time (s) | Peak RAM (MB) | Rows | Cols |
|------------------------|----------|---------------|------|------|
| pandas                 | 1.78     | 29 584        | 31   | 100  |
| dask (graph + compute) | 1.96     | 25 376        | 31   | 100  |

For this small dataset pandas is marginally faster, but Dask's graph construction cost is negligible and its memory footprint is already slightly lower.

------------------------------------------------------------------------

## Task 2 — Preprocessing (Dask Delayed)

`preprocess_delayed.py` merges the radiomics table with the clinical Excel file on a normalised patient ID, then applies two cleaning steps implemented with Dask Delayed:

-   **Sparse-column removal** — columns with \> 30 % missing values are dropped via a partition-wise aggregation of `isna().mean()`.
-   **Approximate-median imputation** — numeric columns are filled using `dask.dataframe.median_approximate()`, which avoids a full sort across partitions.

The result is written to a Parquet directory (`outputs/preprocessed_delta.parquet`), giving 30 rows × 277 columns.

------------------------------------------------------------------------

## Task 3 — Exploratory Data Analysis

`eda_delayed.py` runs both a Dask path (lazy partition-wise moments) and a pandas path (eager `describe()`) on the same dataset and saves their outputs side-by-side for comparison.

Key artifacts produced:

-   `missingness.csv` — missing fraction per column
-   `dask/pandas_summary_stats.csv` — count, mean, std, min, median, max
-   `dask/pandas_top_variance.csv` — top-10 most variable features
-   `target_distribution_*.png` — bar chart of target class counts
-   `group_overview_*.csv` — per-group patient counts and feature means
-   `timings.json` — wall-clock time for each EDA step

The Dask EDA uses custom `delayed` moment aggregation (sum, sum-of-squares, min, max per partition) to compute variance without collecting all data at once.

------------------------------------------------------------------------

## Task 4 — ML Dataset Preparation

`prepare_ml_dataset.py` builds a clean train/test split from the preprocessed Parquet using Dask throughout:

1.  **Leakage removal** — columns matching `surv_*` or `post_cart_*` (excluding the target) are dropped.
2.  **Target encoding** — unique class labels are collected once via `.compute()` and mapped to integers.
3.  **Train/test split** — `dask.dataframe.random_split([0.8, 0.2], random_state=42)`.
4.  **Categorical encoding** — value→code mappings are fitted on the train partition only, then applied to both splits via `map_partitions`.
5.  **Z-score scaling** — mean and std are computed on train only; both splits are scaled with a single `map_partitions` pass.

------------------------------------------------------------------------

## Task 5 — XGBoost Training

Both scripts load the same `train.parquet` / `test.parquet` produced by Task 4 and train a multiclass XGBoost model (`multi:softprob`, 100 rounds).

`train_xgboost_dask.py` uses a `LocalCluster` (2 workers × 2 threads) and `xgboost.dask.DaskDMatrix` so that data never leaves Dask arrays during training. `train_xgboost_pandas.py` uses standard `xgb.DMatrix`.

Both scripts report the same metrics:

| Metric                  | Description                              |
|-------------------------|------------------------------------------|
| Accuracy                | Overall fraction correct                 |
| Precision / Recall / F1 | Macro and weighted averages              |
| AUC-ROC (OVR)           | One-vs-rest multiclass AUC with fallback |

------------------------------------------------------------------------

## Task 6 — End-to-End Benchmark

The notebook wraps every script in `/usr/bin/time` to capture peak RSS and wall-clock time.

| Stage                | Workflow         | Time (s) | Peak RAM (MB) |
|----------------------|------------------|----------|---------------|
| preprocess_delayed   | Dask (Delayed)   | \~30     | \~200         |
| prepare_ml_dataset   | Dask             | \~60     | \~255         |
| train_xgboost_dask   | Dask distributed | \~45     | \~300         |
| train_xgboost_pandas | Pandas           | \~28     | \~200         |

**Total pipeline runtime: \~164 s · Average peak RAM: \~255 MB**

Dataset preparation dominates total runtime because it requires multiple full scans (mean, std, missingness) that each trigger a `.compute()` call. The Dask distributed training step is slower than its pandas counterpart at this scale due to LocalCluster scheduling overhead.

------------------------------------------------------------------------

## Advantages & Trade-offs

**Advantages of Dask observed here:**

-   The `delayed` loader processes patient files in parallel, which scales linearly as the patient count grows.
-   Lazy evaluation defers I/O and computation until explicitly triggered, reducing unnecessary work.
-   `xgboost.dask` trains on distributed arrays without manual data sharding, making the code trivially scalable to a multi-node cluster.
-   Parquet partitioning means large datasets can be processed one chunk at a time without loading everything into RAM.

**Trade-offs observed here:**

-   **Small-data overhead** — for 31 patients, pandas is faster because Dask's task-graph scheduling cost exceeds the compute savings.
-   **Schema management** — `meta`, `reindex`, and `map_partitions` require explicit care; mismatched partition schemas silently corrupt results.
-   **Frequent `.compute()` calls** — fitting scalers and encoders on the train set forces materialisation, which partly defeats lazy evaluation.
-   **Approximate statistics** — `median_approximate` in the Dask preprocessing path can differ slightly from the exact pandas median.

------------------------------------------------------------------------

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
