"""
load_and_benchmark.py

Task 1:
- Load data with pandas and with Dask DataFrame (via delayed + from_delayed)
- Compare load time + (approx) memory usage

Config format (like config_example.yaml in this repo):
paths:
  data_folder: "/path/to/modelling_data"
  clinical_data: "/path/to/clinical.xlsx"
"""

from __future__ import annotations
import argparse
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import pandas as pd
from dask import delayed
import dask.dataframe as dd


Timepoint = Literal["A", "B", "DELTA"]

# read data from yaml
def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  
    except ImportError as e:
        raise ImportError("Missing dependency: PyYAML. Install with: pip install pyyaml") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a dictionary at top level.")
    return cfg


def get_paths(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Retrieve dataset paths from the configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration loaded from a YAML file.

    Returns
    -------
    tuple[str, str]
        Paths to the radiomics data folder and clinical data file.
    """
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Config must contain a 'paths' mapping.")

    data_folder = paths.get("data_folder", "")
    clinical_data = paths.get("clinical_data", "")

    if not data_folder:
        raise ValueError("Missing config key: paths.data_folder")
    if not clinical_data:
        raise ValueError("Missing config key: paths.clinical_data")

    return str(data_folder), str(clinical_data)


def find_ab_files(patient_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate Excel files corresponding to A and B scans in a patient directory.

    Parameters
    ----------
    patient_dir : Path
        Directory containing patient radiomics files.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        Paths to the first detected A-file and B-file (.xlsx), or None if not found.
    """
    a_candidates: List[Path] = []
    b_candidates: List[Path] = []

    for p in patient_dir.iterdir():
        if p.suffix.lower() != ".xlsx":
            continue
        name = p.name.upper()
        if "_A" in name:
            a_candidates.append(p)
        elif "_B" in name:
            b_candidates.append(p)

    a_path = sorted(a_candidates)[0] if a_candidates else None
    b_path = sorted(b_candidates)[0] if b_candidates else None
    return a_path, b_path


def extract_feature_vector(
    xlsx_path: Path,
    segmentation_col: str = "Segmentation",
    segmentation_match: str = "suv2.5",
    feature_start_col: int = 23,
) -> pd.Series:
    """
    Extract a numeric radiomics feature vector from an Excel file.

    The function selects the row matching a specified segmentation label
    and returns numeric feature values starting from a given column index.

    Parameters
    ----------
    xlsx_path : Path
        Path to the radiomics Excel file.
    segmentation_col : str, default="Segmentation"
        Column used to identify the segmentation row.
    segmentation_match : str, default="suv2.5"
        Substring used to match the desired segmentation.
    feature_start_col : int, default=23
        Column index where feature values begin.

    Returns
    -------
    pd.Series
        Numeric feature vector extracted from the selected row.
    """
    df = pd.read_excel(xlsx_path)

    if segmentation_col not in df.columns:
        raise KeyError(f"Missing '{segmentation_col}' in {xlsx_path.name}")

    mask = df[segmentation_col].astype(str).str.contains(segmentation_match, na=False)
    if not mask.any():
        raise ValueError(f"No '{segmentation_match}' row in {xlsx_path.name}")

    row = df.loc[mask].iloc[0, feature_start_col:]
    numeric = pd.to_numeric(row, errors="coerce").dropna()
    numeric.index = numeric.index.astype(str)
    return numeric


def patient_to_row(
    patient_dir: Path,
    which: Timepoint,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Convert a patient's radiomics files into a single feature row.

    The function loads A and B scan features and returns either:
    A features, B features, or their difference (B - A).

    Parameters
    ----------
    patient_dir : Path
        Directory containing the patient's radiomics files.
    which : {"A", "B", "delta"}
        Timepoint to extract: baseline (A), follow-up (B), or feature change.
    id_col : str, default="id"
        Name of the patient identifier column.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing the patient ID and extracted features.
        Returns an empty DataFrame if required files are missing.
    """
    a_path, b_path = find_ab_files(patient_dir)
    if a_path is None or b_path is None:
        return pd.DataFrame(columns=[id_col])

    a_feat = extract_feature_vector(a_path)
    b_feat = extract_feature_vector(b_path)

    if which == "A":
        feat = a_feat
    elif which == "B":
        feat = b_feat
    else:
        feat = (b_feat - a_feat).dropna()

    one = feat.to_frame().T
    one.insert(0, id_col, patient_dir.name)
    return one


def build_meta_from_sample(
    data_folder: str | Path,
    which: Timepoint,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Create a Dask meta DataFrame by scanning for a first valid patient row.
    Falls back to an id-only schema if none is found.
    """
    root = Path(data_folder)
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        try:
            one = patient_to_row(p, which=which, id_col=id_col)
            if not one.empty and one.shape[1] >= 1:
                return one.iloc[0:0].copy()
        except Exception:
            continue

    return pd.DataFrame({id_col: pd.Series(dtype="object")})


# Workflows: pandas vs Dask
def load_pandas(
    data_folder: str | Path,
    which: Timepoint = "DELTA",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Load radiomics features for all patients using a pandas workflow.

    The function iterates through patient directories, extracts features
    using `patient_to_row`, and concatenates them into a single DataFrame.

    Parameters
    ----------
    data_folder : str or Path
        Root directory containing patient folders.
    which : {"A", "B", "DELTA"}, default="DELTA"
        Timepoint to extract features from.
    id_col : str, default="id"
        Name of the patient identifier column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per patient with extracted features.
        Returns an empty DataFrame if no valid rows are found.
    """
    root = Path(data_folder)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    rows: List[pd.DataFrame] = []
    for p in patient_dirs:
        try:
            one = patient_to_row(p, which=which, id_col=id_col)
            if len(one) == 1 and one.shape[1] >= 1:
                rows.append(one)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[id_col])

    return pd.concat(rows, axis=0, ignore_index=True)


def load_dask(
    data_folder: str | Path,
    which: Timepoint = "DELTA",
    id_col: str = "id",
) -> dd.DataFrame:
    """
    Load radiomics features for all patients using a Dask DataFrame workflow.

    The function builds a delayed task per patient directory, enforces a fixed
    schema using a sample-derived `meta`, and combines partitions with
    `dd.from_delayed`.

    Parameters
    ----------
    data_folder : str or Path
        Root directory containing patient folders.
    which : {"A", "B", "DELTA"}, default="DELTA"
        Timepoint to extract features from.
    id_col : str, default="id"
        Name of the patient identifier column.

    Returns
    -------
    dask.dataframe.DataFrame
        Dask DataFrame with one partition per patient (schema-aligned).
    """
    root = Path(data_folder)
    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    meta = build_meta_from_sample(data_folder, which=which, id_col=id_col)
    expected_cols = list(meta.columns)

    def patient_to_row_fixed(patient_dir: Path) -> pd.DataFrame:
        try:
            one = patient_to_row(patient_dir, which=which, id_col=id_col)
        except Exception:
            return meta.copy()

        if one.empty:
            return meta.copy()

        # Ensure exact schema match for all partitions
        return one.reindex(columns=expected_cols)

    delayed_parts = [delayed(patient_to_row_fixed)(p) for p in patient_dirs]
    return dd.from_delayed(delayed_parts, meta=meta)


# Benchmarking
@dataclass(frozen=True)
class BenchRow:
    workflow: str
    seconds: float
    peak_mb: float
    rows: int
    cols: int


def bench_call(fn, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Execute a function and measure runtime and peak memory usage.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    *args, **kwargs
        Arguments passed to the function.

    Returns
    -------
    tuple[Any, float, float]
        Function output, execution time (seconds), and peak memory usage (MB).
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    out = fn(*args, **kwargs)

    t1 = time.perf_counter()
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()

    peak_mb = sum(stat.size for stat in snap.statistics("filename")) / (1024 * 1024)
    return out, (t1 - t0), peak_mb


def run_compare(data_folder: str, which: Timepoint, compute_dask: bool) -> List[BenchRow]:
    """
    Benchmark pandas and Dask workflows for loading radiomics features.

    The function measures runtime and memory usage for both implementations
    and returns their performance statistics.

    Parameters
    ----------
    data_folder : str
        Root directory containing patient data folders.
    which : Timepoint
        Timepoint used for feature extraction.
    compute_dask : bool
        Whether to execute the Dask computation graph.

    Returns
    -------
    list[BenchRow]
        Benchmark results for pandas and Dask workflows.
    """
    pdf, t_p, m_p = bench_call(load_pandas, data_folder, which)
    out = [BenchRow("pandas", t_p, m_p, pdf.shape[0], pdf.shape[1])]

    ddf, t_d_graph, m_d_graph = bench_call(load_dask, data_folder, which)
    if compute_dask:
        computed, t_d_compute, m_d_compute = bench_call(lambda x: x.compute(), ddf)
        out.append(
            BenchRow(
                "dask(graph+compute)",
                t_d_graph + t_d_compute,
                max(m_d_graph, m_d_compute),
                computed.shape[0],
                computed.shape[1],
            )
        )
    else:
        out.append(BenchRow("dask(graph-only)", t_d_graph, m_d_graph, -1, -1))

    return out


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1: benchmark pandas vs dask load.")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--which", choices=["A", "B", "DELTA"], default="DELTA")
    p.add_argument("--no-compute-dask", action="store_true")
    p.add_argument("--out-csv", default="", help="Optional: save computed Dask dataset to CSV")
    return p.parse_args()

# main
def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    data_folder, clinical_data = get_paths(cfg)

    print(f"Config OK. data_folder={data_folder}")
    print(f"Config OK. clinical_data={clinical_data}")

    rows = run_compare(data_folder, which=args.which, compute_dask=not args.no_compute_dask)
    report = pd.DataFrame([r.__dict__ for r in rows])

    print("\nTask 1: Load benchmark")
    print(report.to_string(index=False))

    if args.out_csv:
        df = load_dask(data_folder, which=args.which).compute()
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path} (rows={df.shape[0]}, cols={df.shape[1]})")


if __name__ == "__main__":
    main()