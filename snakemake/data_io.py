"""
workflow/scripts/data_io.py

File-based wrapper around the existing data_io.py logic, so Snakemake can
track inputs/outputs via files.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
import pandas as pd

# Add the project root to sys.path so this script can import modules
# (e.g., config/config.py) when executed from the workflow/scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import load_paths


def calculate_delta_radiomics(data_folder_path):
    """
    tasks:
    - For each patient folder, find A and B .xlsx
    - Read both
    - Find first row where 'Segmentation' contains 'suv2.5'
    - Take columns from index 23 onward
    - Compute delta = B - A
    - Return (delta_df, A_df, B_df)
    """
    all_delta_radiomics = {}
    A_radiomics = {}
    B_radiomics = {}

    feature_start_col = 23

    for patient_folder_name in sorted(os.listdir(data_folder_path)):
        patient_path = os.path.join(data_folder_path, patient_folder_name)

        if not os.path.isdir(patient_path):
            continue

        file_A_path, file_B_path = _find_patient_ab_files(patient_path)

        if file_A_path and not file_B_path:
            warnings.warn(
                "Patient {} has Time A file but missing Time B file. Skipping.".format(
                    patient_folder_name
                )
            )
            continue

        if file_B_path and not file_A_path:
            continue

        if not file_A_path and not file_B_path:
            continue

        try:
            df_A = pd.read_excel(file_A_path)
            df_B = pd.read_excel(file_B_path)

            if "Segmentation" not in df_A.columns or "Segmentation" not in df_B.columns:
                continue

            mask_A = df_A["Segmentation"].astype(str).str.contains("suv2.5", na=False)
            mask_B = df_B["Segmentation"].astype(str).str.contains("suv2.5", na=False)

            if not mask_A.any() or not mask_B.any():
                continue

            row_A = df_A.loc[mask_A].iloc[0, feature_start_col:]
            row_B = df_B.loc[mask_B].iloc[0, feature_start_col:]

            numeric_A = pd.to_numeric(row_A, errors="coerce")
            numeric_B = pd.to_numeric(row_B, errors="coerce")

            delta_radiomics = numeric_B - numeric_A

            all_delta_radiomics[patient_folder_name] = delta_radiomics.dropna().to_dict()
            A_radiomics[patient_folder_name] = numeric_A.dropna().to_dict()
            B_radiomics[patient_folder_name] = numeric_B.dropna().to_dict()

        except Exception:
            continue

    A_df = pd.DataFrame.from_dict(A_radiomics, orient="index")
    B_df = pd.DataFrame.from_dict(B_radiomics, orient="index")
    delta_df = pd.DataFrame.from_dict(all_delta_radiomics, orient="index")

    return delta_df, A_df, B_df


def _find_patient_ab_files(patient_path):
    a_candidates = []
    b_candidates = []

    for filename in os.listdir(patient_path):
        full_path = os.path.join(patient_path, filename)
        upper_name = filename.upper()

        if not full_path.endswith(".xlsx"):
            continue

        if "_A" in upper_name:
            a_candidates.append(full_path)
        elif "_B" in upper_name:
            b_candidates.append(full_path)

    file_A_path = sorted(a_candidates)[0] if a_candidates else None
    file_B_path = sorted(b_candidates)[0] if b_candidates else None
    return file_A_path, file_B_path


def _ensure_parent(path_str):
    '''
    an innner function to ensure the parent directory of a given path exists before writing output files.'''
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def main():
    '''
    Define required command-line arguments so this script can be executed
    by Snakemake with explicit input and output file paths.
    '''
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. config/config.yaml)")
    p.add_argument("--out-delta", required=True, help="Output CSV path for delta radiomics")
    p.add_argument("--out-a", required=True, help="Output CSV path for A radiomics")
    p.add_argument("--out-b", required=True, help="Output CSV path for B radiomics")
    args = p.parse_args()

    data_folder, _clinical_path = load_paths(args.config)

    delta_df, A_df, B_df = calculate_delta_radiomics(data_folder)

    _ensure_parent(args.out_delta)
    _ensure_parent(args.out_a)
    _ensure_parent(args.out_b)

    # Save the data frames to csv files
    delta_df.to_csv(args.out_delta, index=True)
    A_df.to_csv(args.out_a, index=True)
    B_df.to_csv(args.out_b, index=True)


if __name__ == "__main__":
    main()