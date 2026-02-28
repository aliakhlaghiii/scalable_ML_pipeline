"""
data_io.py

Input/output helpers for reading radiomics files from disk and producing
A, B, and Delta radiomics DataFrames.

Assumptions:
- Each patient has a subfolder under `data_folder_path`.
- Inside each patient folder there are Excel files for timepoints A and B,
  detected by filename containing "_A" or "_B" (case-insensitive).
- Radiomics features start at column index 23.
- We pick the FIRST row whose 'Segmentation' contains 'suv2.5'.
- Delta = B - A, computed feature-wise.
"""

import os
import warnings
import pandas as pd


def calculate_delta_radiomics(data_folder_path):
    """
    Read radiomics data per patient from subfolders, extract the first row whose
    'Segmentation' contains 'suv2.5', and compute delta features (B - A).

    Policy:
    - If patient has A but no B: warn + skip
    - If 'Segmentation' column is missing: skip patient
    - If 'suv2.5' is not found: skip patient
    - If multiple 'suv2.5' rows exist: take only the first match

    Parameters
    ----------
    data_folder_path : str
        Path to the main folder containing patient subfolders.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        delta_df : Delta radiomics (B - A), patients as index, features as columns.
        A_df     : Radiomics at time A.
        B_df     : Radiomics at time B.
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
    """
    Find the Time A and Time B radiomics Excel files for one patient folder.

    Parameters
    ----------
    patient_path : str
        Path to a single patient folder.

    Returns
    -------
    (str or None, str or None)
        file_A_path, file_B_path
    """
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
