# eda_delayed.py
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
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask import delayed

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
    """Measure runtime in seconds for a callable (returns (result, seconds))."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def pick_target_column(ddf: dd.DataFrame, preferred: str = "") -> str:
    """Heuristic target column picker."""
    if preferred and preferred in ddf.columns:
        return preferred

    # common candidate patterns
    candidates = [
        c
        for c in ddf.columns
        if any(k in str(c).lower() for k in ("target", "response", "surv_bestresponse"))
    ]

    return candidates[0] if candidates else ""


def _numeric_cols_pdf(pdf: pd.DataFrame, id_col: str = "id") -> List[str]:
    """Select numeric columns in pandas, excluding ID."""
    cols: List[str] = []
    for c in pdf.columns:
        if c == id_col:
            continue
        if pd.api.types.is_numeric_dtype(pdf[c]):
            cols.append(c)
    return cols


def _numeric_cols_ddf(ddf: dd.DataFrame, id_col: str = "id") -> List[str]:
    """Select numeric columns in Dask DataFrame, excluding ID."""
    cols: List[str] = []
    for c, dt in ddf.dtypes.items():
        if c == id_col:
            continue
        if pd.api.types.is_numeric_dtype(dt):
            cols.append(c)
    return cols


def plot_target_distribution(series: pd.Series, out_png: Path, title: str) -> None:
    """Plot and save the target variable distribution."""
    import matplotlib.pyplot as plt

    vc = series.value_counts(dropna=False)
    plt.figure()
    vc.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# EDA (Dask + Pandas)
def dask_eda(ddf: dd.DataFrame, topk: int, target_col: str) -> Dict[str, object]:
    """Compute EDA artifacts using Dask (lazy ops + compute at the end)."""
    out: Dict[str, object] = {}

    num_cols = _numeric_cols_ddf(ddf, id_col="id")
    out["n_numeric"] = int(len(num_cols))

    parts = ddf.to_delayed()

    def _row_and_missing(pdf: pd.DataFrame) -> Dict[str, object]:
        """Partition-level row count and missing counts."""
        n = int(len(pdf))
        miss = pdf.isna().sum().to_dict()
        return {"n": n, "miss": {k: int(v) for k, v in miss.items()}}

    stats_parts = [delayed(_row_and_missing)(p) for p in parts]

    def _agg_rows_and_missing(stats_list: List[Dict[str, object]], cols: List[str]) -> Tuple[int, pd.Series]:
        """Aggregate missing counts into missing fractions."""
        total_n = 0
        miss_totals = {c: 0 for c in cols}
        for d in stats_list:
            total_n += int(d.get("n", 0))
            miss = d.get("miss", {}) or {}
            for c in cols:
                miss_totals[c] += int(miss.get(c, 0))
        if total_n <= 0:
            return 0, pd.Series({c: 1.0 for c in cols})
        return total_n, pd.Series({c: miss_totals[c] / float(total_n) for c in cols})

    n_rows, missing_frac = delayed(_agg_rows_and_missing)(stats_parts, list(ddf.columns)).compute()
    out["rows"] = int(n_rows)
    out["cols"] = int(len(ddf.columns))
    out["missing"] = missing_frac.sort_values(ascending=False)

    if not num_cols:
        out["summary"] = pd.DataFrame()
        out["top_variance"] = pd.Series(dtype="float64")
    else:
        def _num_moments(pdf: pd.DataFrame, cols: List[str]) -> Dict[str, object]:
            """Partition-level numeric moments for summary/variance."""
            x = pdf[cols].apply(pd.to_numeric, errors="coerce")
            cnt = x.count().to_dict()
            s1 = x.sum().to_dict()
            s2 = (x * x).sum().to_dict()
            mn = x.min().to_dict()
            mx = x.max().to_dict()
            return {
                "cnt": {k: int(v) for k, v in cnt.items()},
                "s1": {k: float(v) if pd.notna(v) else 0.0 for k, v in s1.items()},
                "s2": {k: float(v) if pd.notna(v) else 0.0 for k, v in s2.items()},
                "min": {k: float(v) if pd.notna(v) else np.nan for k, v in mn.items()},
                "max": {k: float(v) if pd.notna(v) else np.nan for k, v in mx.items()},
            }

        mom_parts = [delayed(_num_moments)(p, num_cols) for p in parts]

        def _agg_moments(moms: List[Dict[str, object]], cols: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
            """Aggregate numeric moments across partitions."""
            cnt = {c: 0 for c in cols}
            s1 = {c: 0.0 for c in cols}
            s2 = {c: 0.0 for c in cols}
            mn = {c: np.nan for c in cols}
            mx = {c: np.nan for c in cols}

            for m in moms:
                mc = m.get("cnt", {}) or {}
                ms1 = m.get("s1", {}) or {}
                ms2 = m.get("s2", {}) or {}
                mmin = m.get("min", {}) or {}
                mmax = m.get("max", {}) or {}

                for c in cols:
                    cnt[c] += int(mc.get(c, 0))
                    s1[c] += float(ms1.get(c, 0.0))
                    s2[c] += float(ms2.get(c, 0.0))
                    vmin = mmin.get(c, np.nan)
                    vmax = mmax.get(c, np.nan)
                    mn[c] = np.nanmin([mn[c], vmin]) if (pd.notna(vmin) or pd.notna(mn[c])) else np.nan
                    mx[c] = np.nanmax([mx[c], vmax]) if (pd.notna(vmax) or pd.notna(mx[c])) else np.nan

            cnt_s = pd.Series(cnt, dtype="int64")
            s1_s = pd.Series(s1, dtype="float64")
            s2_s = pd.Series(s2, dtype="float64")
            mn_s = pd.Series(mn, dtype="float64")
            mx_s = pd.Series(mx, dtype="float64")
            return cnt_s, s1_s, s2_s, mn_s, mx_s

        cnt_s, s1_s, s2_s, mn_s, mx_s = delayed(_agg_moments)(mom_parts, num_cols).compute()
        mean_s = s1_s / cnt_s.replace(0, np.nan)
        var_s = (s2_s / cnt_s.replace(0, np.nan)) - (mean_s * mean_s)
        var_s = var_s.clip(lower=0.0)
        std_s = np.sqrt(var_s)

        desc = pd.DataFrame(index=["count", "mean", "std", "min", "max"], columns=num_cols, dtype="float64")
        desc.loc["count"] = cnt_s.astype("float64")
        desc.loc["mean"] = mean_s
        desc.loc["std"] = std_s
        desc.loc["min"] = mn_s
        desc.loc["max"] = mx_s

        med = ddf[num_cols].median_approximate().compute()
        desc.loc["median_approx"] = med
        out["summary"] = desc

        out["top_variance"] = var_s.sort_values(ascending=False).head(topk)

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

    print("\nTask 3: EDA report")
    print(json.dumps(timings, indent=2))
    print(f"rows={d_res['rows']}, cols={d_res['cols']}, numeric={d_res['n_numeric']}")
    print(f"Artifacts in: {outdir.resolve()}")


if __name__ == "__main__":
    main()