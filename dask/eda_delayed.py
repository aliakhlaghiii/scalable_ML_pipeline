"""eda_delayed.py

Task 3: Exploratory Data Analysis with Dask

Covers:
- Group patients and make useful overviews
- Visualize the distribution of the target variable
- Compare processing time and outputs with pandas workflows
- Summary stats
- Top-K most variable features
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import dask.dataframe as dd

# CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the EDA run."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to parquet dataset directory (preprocessed)")
    p.add_argument("--outdir", default="outputs/eda", help="Output folder for EDA artifacts")
    p.add_argument("--topk", type=int, default=10, help="Top-K variable features")
    p.add_argument("--target-col", default="", help="Target column name (if empty: auto-detect)")
    p.add_argument(
        "--group-col",
        default="",
        help="Grouping column for overviews (if empty: uses target if available)",
    )
    return p.parse_args()


# Utils
def _timer(fn, *args, **kwargs):
    """Run a function and return (output, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def _is_numeric_dtype(dtype) -> bool:
    """Return True if a dtype should be treated as numeric for EDA."""
    # NOTE: Dask dtypes are sometimes stringified; this keeps the check simple and stable.
    s = str(dtype)
    return s not in ("object", "string", "boolean")


def _numeric_cols_ddf(ddf: dd.DataFrame, id_col: str = "id") -> List[str]:
    """List numeric feature columns in a Dask DataFrame, excluding the ID column."""
    return [c for c in ddf.columns if c != id_col and _is_numeric_dtype(ddf[c].dtype)]


def _numeric_cols_pdf(pdf: pd.DataFrame, id_col: str = "id") -> List[str]:
    """List numeric feature columns in a pandas DataFrame, excluding the ID column."""
    return [c for c in pdf.columns if c != id_col and pd.api.types.is_numeric_dtype(pdf[c])]


def pick_target_column(ddf: dd.DataFrame, user_target: str) -> str:
    """Pick a target column using user input, known candidates, or heuristics."""
    if user_target and user_target in ddf.columns:
        return user_target

    # sensible defaults for your dataset (based on columns you already saw)
    candidates = [
        "surv_bestresponse_car.factor",
        "surv_bestresponse_car",
        "surv_prog_after_car.factor",
        "surv_prog_after_car",
        "surv_time_bestresponse_car",
        "ae_summ_highestgrade_v2.factor",
        "ae_summ_highestgrade_v2",
    ]
    for c in candidates:
        if c in ddf.columns:
            return c

    # fallback: any column containing "surv_" and ".factor" first
    cols = list(ddf.columns)
    for c in cols:
        if "surv_" in c and c.endswith(".factor"):
            return c
    for c in cols:
        if "surv_" in c:
            return c

    return ""  # no target found


def plot_target_distribution(series: pd.Series, out_png: Path, title: str) -> None:
    """Save a histogram (numeric) or bar chart (categorical) for the target series."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Decide categorical vs numeric
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
        # numeric target -> histogram
        vals = pd.to_numeric(series, errors="coerce").dropna()
        plt.figure()
        plt.hist(vals.values, bins=20)
        plt.title(title)
        plt.xlabel("target value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return

    # categorical target -> bar
    counts = series.value_counts(dropna=False)
    x = counts.index.astype(str)
    y = counts.values

    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel("target class")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# EDA (Dask + Pandas)
def dask_eda(ddf: dd.DataFrame, topk: int, target_col: str) -> Dict[str, object]:
    """Compute EDA artifacts using Dask (lazy ops + compute at the end)."""
    out: Dict[str, object] = {}

    # ddf.shape[0] is lazy; compute() triggers the actual row count execution.
    out["rows"] = int(ddf.shape[0].compute())
    out["cols"] = int(len(ddf.columns))

    missing = ddf.isna().mean().compute().sort_values(ascending=False)
    out["missing"] = missing

    num_cols = _numeric_cols_ddf(ddf, id_col="id")
    out["n_numeric"] = int(len(num_cols))

    # summary stats + median_approx
    desc = ddf[num_cols].describe().compute()
    med = ddf[num_cols].median_approximate().compute()
    desc.loc["median_approx"] = med
    out["summary"] = desc

    var = ddf[num_cols].var().compute().sort_values(ascending=False).head(topk)
    out["top_variance"] = var

    if target_col and target_col in ddf.columns:
        out["target_series"] = ddf[target_col].compute()
        out["target_counts"] = ddf[target_col].value_counts(dropna=False).compute()
    else:
        out["target_series"] = None
        out["target_counts"] = None

    return out


def pandas_eda(pdf: pd.DataFrame, topk: int, target_col: str) -> Dict[str, object]:
    """Compute EDA artifacts using pandas (eager in-memory ops)."""
    out: Dict[str, object] = {}

    out["rows"] = int(pdf.shape[0])
    out["cols"] = int(pdf.shape[1])

    out["missing"] = pdf.isna().mean().sort_values(ascending=False)

    num_cols = _numeric_cols_pdf(pdf, id_col="id")
    out["n_numeric"] = int(len(num_cols))

    desc = pdf[num_cols].describe()
    desc.loc["median"] = pdf[num_cols].median(numeric_only=True)
    out["summary"] = desc

    var = pdf[num_cols].var(numeric_only=True).sort_values(ascending=False).head(topk)
    out["top_variance"] = var

    if target_col and target_col in pdf.columns:
        out["target_series"] = pdf[target_col]
        out["target_counts"] = pdf[target_col].value_counts(dropna=False)
    else:
        out["target_series"] = None
        out["target_counts"] = None

    return out


def group_overview(
    df_like,  # ddf or pdf
    group_key: str,
    rep_features: List[str],
    is_dask: bool,
) -> pd.DataFrame:
    """Aggregate per-group patient counts and representative feature means."""
    if is_dask:
        ddf: dd.DataFrame = df_like
        # NOTE: count() ignores NaNs; for IDs this is usually fine if IDs are non-null.
        n = ddf.groupby(group_key)["id"].count().rename("n_patients").compute()
        means = ddf.groupby(group_key)[rep_features].mean().compute()
        out = means.copy()
        out.insert(0, "n_patients", n)
        return out

    pdf: pd.DataFrame = df_like
    n = pdf.groupby(group_key)["id"].count().rename("n_patients")
    means = pdf.groupby(group_key)[rep_features].mean(numeric_only=True)
    out = means.copy()
    out.insert(0, "n_patients", n)
    return out


# Main
def main() -> None:
    """Run Dask vs pandas EDA, save artifacts, and print a short console report."""
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}

    ddf, t = _timer(dd.read_parquet, args.data)
    timings["dask_load"] = t

    target_col = pick_target_column(ddf, args.target_col)

    d_res, t = _timer(dask_eda, ddf, args.topk, target_col)
    timings["dask_eda_total"] = t

    # Convert the full dataset to pandas for an apples-to-apples comparison.
    pdf, t = _timer(lambda: ddf.compute())
    timings["to_pandas_compute"] = t

    p_res, t = _timer(pandas_eda, pdf, args.topk, target_col)
    timings["pandas_eda_total"] = t

    # Save common artifacts
    pd.Series(d_res["missing"]).to_csv(outdir / "missingness.csv", header=["missing_fraction"])

    pd.DataFrame(d_res["summary"]).to_csv(outdir / "dask_summary_stats.csv")
    pd.DataFrame(p_res["summary"]).to_csv(outdir / "pandas_summary_stats.csv")

    pd.Series(d_res["top_variance"]).to_csv(outdir / "dask_top_variance.csv", header=["variance"])
    pd.Series(p_res["top_variance"]).to_csv(outdir / "pandas_top_variance.csv", header=["variance"])

    # Target distribution (visualize + csv)
    if target_col:
        (outdir / "target_col.txt").write_text(target_col, encoding="utf-8")

    if d_res["target_series"] is not None:
        d_counts = pd.Series(d_res["target_counts"])
        d_counts.to_csv(outdir / "target_distribution_dask.csv", header=["count"])
        plot_target_distribution(
            pd.Series(d_res["target_series"]),
            outdir / "target_distribution_dask.png",
            title=f"Target distribution (Dask) — {target_col}",
        )

    if p_res["target_series"] is not None:
        p_counts = pd.Series(p_res["target_counts"])
        p_counts.to_csv(outdir / "target_distribution_pandas.csv", header=["count"])
        plot_target_distribution(
            pd.Series(p_res["target_series"]),
            outdir / "target_distribution_pandas.png",
            title=f"Target distribution (Pandas) — {target_col}",
        )

    # Grouping overview
    group_key = ""
    if args.group_col and args.group_col in ddf.columns:
        group_key = args.group_col
    elif target_col and target_col in ddf.columns:
        group_key = target_col

    if group_key:
        # Use the top-variance features as representative features for group means.
        rep_feats = list(pd.Series(d_res["top_variance"]).index[: min(10, args.topk)])
        d_group, t = _timer(group_overview, ddf, group_key, rep_feats, True)
        timings["dask_group_overview"] = t
        d_group.to_csv(outdir / "group_overview_dask.csv")

        p_group, t = _timer(group_overview, pdf, group_key, rep_feats, False)
        timings["pandas_group_overview"] = t
        p_group.to_csv(outdir / "group_overview_pandas.csv")

    # Timings
    (outdir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    # Console summary
    print("\nDATASET OVERVIEW")
    print(f"rows: {d_res['rows']}")
    print(f"cols: {d_res['cols']}")
    print(f"numeric_cols: {d_res['n_numeric']}")
    print(f"outdir: {outdir.resolve()}")

    print("\nTARGET")
    print(target_col if target_col else "NOT FOUND")

    print("\nTOP MISSING (fraction)")
    print(pd.Series(d_res["missing"]).head(15))

    print(f"\nTOP {args.topk} VARIANCE (Dask)")
    print(pd.Series(d_res["top_variance"]))

    if group_key:
        print(f"\nGROUP KEY USED\n{group_key}")

    print("\nTIMINGS (seconds)")
    for k in sorted(timings):
        print(f"{k}: {timings[k]:.4f}")


if __name__ == "__main__":
    main()
