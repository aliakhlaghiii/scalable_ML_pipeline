"""
data_io.py

Input/output helpers for reading radiomics files from disk and producing
A, B, and Delta radiomics DataFrames.
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


def get_timeA_timeB_paths(patient_folder_path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Find Time A and Time B radiomics Excel files inside a patient folder.

    Matching rule (case-insensitive):
      - Time A: filename contains "_A" and ends with ".xlsx"
      - Time B: filename contains "_B" and ends with ".xlsx"

    If multiple candidates exist, the first in sorted order is chosen (deterministic).
    """
    p = Path(patient_folder_path)
    if not p.exists() or not p.is_dir():
        return None, None

    a_candidates: list[str] = []
    b_candidates: list[str] = []

    for name in sorted(os.listdir(p)):
        full_path = str(p / name)
        if not name.lower().endswith(".xlsx"):
            continue

        upper_name = name.upper()
        if "_A" in upper_name:
            a_candidates.append(full_path)
        elif "_B" in upper_name:
            b_candidates.append(full_path)

    file_A_path = sorted(a_candidates)[0] if a_candidates else None
    file_B_path = sorted(b_candidates)[0] if b_candidates else None
    return file_A_path, file_B_path


def calculate_delta_radiomics(
    data_folder_path: str | Path,
    *,
    segmentation_value: str = "suv2.5",
    segmentation_column: str = "Segmentation",
    feature_start_col: int = 23,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read radiomics data per patient from subfolders, extract the first row whose
    segmentation label matches `segmentation_value`, and compute delta features (B - A).

    Parameters
    ----------
    data_folder_path : str | Path
        Path to the directory containing per-patient subfolders.
    segmentation_value : str
        Segmentation identifier to select (default: "suv2.5").
    segmentation_column : str
        Column name that contains segmentation labels (default: "Segmentation").
    feature_start_col : int
        Column index (0-based) where radiomics features start in the exported Excel layout.
        Assumes the first `feature_start_col` columns are metadata.

    Returns
    -------
    delta_df : pd.DataFrame
        Delta radiomics features (B - A), indexed by patient folder name.
    A_df : pd.DataFrame
        Time A radiomics features, indexed by patient folder name.
    B_df : pd.DataFrame
        Time B radiomics features, indexed by patient folder name.
    """
    base = Path(data_folder_path)

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"data_folder_path not found or not a directory: {base}")

    all_A: dict[str, dict] = {}
    all_B: dict[str, dict] = {}
    all_delta: dict[str, dict] = {}

    seg_target = str(segmentation_value).strip().lower()

    for patient_folder_name in sorted(os.listdir(base)):
        patient_dir = base / patient_folder_name
        if not patient_dir.is_dir():
            continue

        file_A_path, file_B_path = get_timeA_timeB_paths(patient_dir)

        # warn + skip if A exists but B missing (as requested)
        if file_A_path and not file_B_path:
            warnings.warn(
                f"Patient {patient_folder_name} has Time A file but missing Time B file. Skipping."
            )
            continue

        # if B exists but A missing, skip (quiet by default)
        if file_B_path and not file_A_path:
            continue

        # if neither exists, skip
        if not file_A_path and not file_B_path:
            continue

        try:
            df_A = pd.read_excel(file_A_path)
            df_B = pd.read_excel(file_B_path)
        except Exception as e:
            warnings.warn(f"Skipping {patient_folder_name}: failed to read Excel files ({e}).")
            continue

        # basic layout checks
        if segmentation_column not in df_A.columns or segmentation_column not in df_B.columns:
            warnings.warn(
                f"Skipping {patient_folder_name}: missing '{segmentation_column}' column."
            )
            continue

        if df_A.shape[1] <= feature_start_col or df_B.shape[1] <= feature_start_col:
            warnings.warn(
                f"Skipping {patient_folder_name}: unexpected column layout "
                f"(ncols A={df_A.shape[1]}, B={df_B.shape[1]}, feature_start_col={feature_start_col})."
            )
            continue

        # select segmentation rows (case-insensitive)
        seg_A = df_A[segmentation_column].astype(str).str.strip().str.lower()
        seg_B = df_B[segmentation_column].astype(str).str.strip().str.lower()

        rows_A = df_A.loc[seg_A == seg_target]
        rows_B = df_B.loc[seg_B == seg_target]

        if rows_A.empty or rows_B.empty:
            warnings.warn(
                f"Skipping {patient_folder_name}: segmentation '{segmentation_value}' not found in A and/or B."
            )
            continue

        # deterministic: take first row in original order
        row_A = rows_A.iloc[0]
        row_B = rows_B.iloc[0]

        # extract features by positional slice
        feat_A = row_A.iloc[feature_start_col:]
        feat_B = row_B.iloc[feature_start_col:]

        # coerce to numeric
        feat_A = pd.to_numeric(feat_A, errors="coerce")
        feat_B = pd.to_numeric(feat_B, errors="coerce")

        # align on feature names and compute delta
        feat_A, feat_B = feat_A.align(feat_B, join="inner")
        delta = feat_B - feat_A

        # store (drop NaN entries to avoid per-patient garbage)
        all_A[patient_folder_name] = feat_A.dropna().to_dict()
        all_B[patient_folder_name] = feat_B.dropna().to_dict()
        all_delta[patient_folder_name] = delta.dropna().to_dict()

    A_df = pd.DataFrame.from_dict(all_A, orient="index").sort_index()
    B_df = pd.DataFrame.from_dict(all_B, orient="index").sort_index()
    delta_df = pd.DataFrame.from_dict(all_delta, orient="index").sort_index()

    return delta_df, A_df, B_df