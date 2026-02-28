"""
preprocessing.py

All clinical and radiomics preprocessing steps extracted from the notebook.

Steps performed:
1. Clean radiomics DataFrames (drop NaN columns, reset index, cast id to int)
2. Add suffixes to A/B radiomics columns to avoid name collisions
3. Load and clean clinical data (id extraction, column filtering, encoding)
4. Align cohort across all data sources (intersection of patient IDs)
5. Build the four feature matrices used for modelling
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_COL = "id"

LABEL_CANDIDATES = ["surv_status", "surv_status.factor", "surv_status_factor"]

# Columns with raw dates/times that must not be used as features
RAW_DATE_COLS = [
    "ae_summ_icans_start_v2",
    "ae_summ_icans_stop_v2",
    "surv_prog_date",
    "surv_death_date",
]

# Date-like columns to drop before modelling (cannot be imputed meaningfully)
DATE_COLS_TO_DROP = [
    "indication_ind_date",
    "tr_car_inf_adm_date",
    "tr_car_ld_start",
    "tr_car_inf_date",
    "tr_car_inf_discharge_date",
    "ae_summ_start_date_v2",
    "ae_summ_crs_start_v2",
    "ae_summ_crs_stop_v2",
    "surv_date",
    "cli_st_lab_date",
]

# Columns correlated with BMI — drop to avoid redundancy
CORRELATED_COLS = ["scr_height", "scr_weight"]

# Indicator columns already encoded in the label — drop to avoid leakage
INDICATOR_COLS = [
    "indication_ldh_uln",
    "indication_age_60",
    "indication_extran_sites",
    "indication_extran_invol",
]

# NaN-placeholder string used in the clinical export
NAN_PLACEHOLDER = "NE"

# Drop columns whose NaN count exceeds this fraction of the cohort
NAN_FRACTION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Radiomics cleaning
# ---------------------------------------------------------------------------

def clean_radiomics(delta_df, a_df, b_df):
    """
    Drop columns with any NaN, reset index, rename index column to 'id',
    cast id to int, and add _a / _b suffixes to A and B feature columns.

    Parameters
    ----------
    delta_df, a_df, b_df : pd.DataFrame
        Raw outputs from calculate_delta_radiomics().

    Returns
    -------
    (delta_df, a_df, b_df) : cleaned DataFrames with integer 'id' column.
    """
    for df in [delta_df, a_df, b_df]:
        df.dropna(axis=1, how="any", inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={"index": ID_COL}, inplace=True)
        df[ID_COL] = df[ID_COL].astype(int)

    # Add suffixes so A and B feature names don't collide
    # Preserve the id column without suffix
    a_df = a_df.rename(columns={c: c + "_a" for c in a_df.columns if c != ID_COL})
    b_df = b_df.rename(columns={c: c + "_b" for c in b_df.columns if c != ID_COL})

    return delta_df, a_df, b_df


# ---------------------------------------------------------------------------
# Clinical data cleaning
# ---------------------------------------------------------------------------

def load_and_clean_clinical(clinical_path):
    """
    Load the clinical Excel file and apply all preprocessing steps from the notebook:
    - Extract a clean 3-digit integer patient id from 'record_id'
    - Drop low-information columns (factors, comments, locations, dates,
      correlated features, cause-of-death columns)
    - One-hot encode 'indication_dis_diagnosis.factor'
    - Replace NE placeholder with NaN
    - Convert remaining object columns to numeric
    - Drop columns with > 50% NaN values
    - Drop zero-variance features
    - Invert 'surv_status' (1=alive, 0=deceased)

    Parameters
    ----------
    clinical_path : str
        Path to the clinical Excel file.

    Returns
    -------
    pd.DataFrame
        Cleaned clinical DataFrame with integer 'id' column and all numeric features.
    """
    clinic = pd.read_excel(clinical_path)

    # --- Extract clean 3-digit patient id ---
    clinic["id_cleaned"] = [str(v)[-3:] for v in clinic["record_id"].values]
    # First row is a comment row in this export; drop it and cast to int
    patient_ids = clinic["id_cleaned"].values[1:].astype(int)
    clinic["id_cleaned"] = ["ID"] + patient_ids.tolist()
    clinic = clinic[clinic["id_cleaned"] != "ID"].copy()
    clinic["id_cleaned"] = clinic["id_cleaned"].astype(int)
    clinic = clinic.rename(columns={"id_cleaned": ID_COL})

    # --- Drop all-NaN columns ---
    clinic = clinic.dropna(axis=1, how="all")

    # --- Identify column groups to drop ---
    factors = [c for c in clinic.columns if "factor" in c]
    comments = [c for c in clinic.columns if "comment" in c]
    locations = [c for c in clinic.columns if "loc" in c]
    cause_of_death = [c for c in clinic.columns if "_cause" in c]

    # One-hot encode disease diagnosis before dropping the factor column
    if "indication_dis_diagnosis.factor" in clinic.columns:
        disease_dummies = pd.get_dummies(
            clinic["indication_dis_diagnosis.factor"]
        ).astype(int)
    else:
        disease_dummies = pd.DataFrame(index=clinic.index)

    drop_cols = (
        cause_of_death
        + factors
        + comments
        + locations
        + CORRELATED_COLS
        + INDICATOR_COLS
        + ["record_id", "scr_date_tb1stmeeting", "indication_dis_diagnosis"]
    )
    drop_cols = [c for c in drop_cols if c in clinic.columns]
    clinic = clinic.drop(columns=drop_cols)

    # Attach one-hot encoded disease columns
    clinic = pd.concat([clinic, disease_dummies], axis=1)

    # --- Replace NE placeholder with NaN ---
    clinic = clinic.replace({NAN_PLACEHOLDER: np.nan})

    # --- Convert object columns to numeric (comma as decimal separator) ---
    date_cols_present = [
        c for c in clinic.columns
        if any(kw in c for kw in ("date", "start", "stop"))
    ]
    for col in clinic.columns:
        if col == ID_COL:
            continue
        if col in date_cols_present:
            clinic[col] = pd.to_datetime(clinic[col], errors="coerce")
        elif clinic[col].dtype == object:
            clinic[col] = pd.to_numeric(
                clinic[col].astype(str).str.replace(",", "."),
                errors="coerce",
            )

    # --- Drop date columns (cannot be meaningfully imputed) ---
    date_cols_to_drop = [c for c in DATE_COLS_TO_DROP if c in clinic.columns]
    clinic = clinic.drop(columns=date_cols_to_drop)

    # --- Drop columns with > 50% NaN ---
    n_patients = clinic.shape[0]
    nan_counts = clinic.isna().sum()
    high_nan_cols = nan_counts[nan_counts > NAN_FRACTION_THRESHOLD * n_patients].index
    clinic = clinic.drop(columns=high_nan_cols)

    # --- Drop zero-variance columns ---
    numeric_cols = clinic.select_dtypes(include=np.number).columns.difference([ID_COL])
    zero_var = clinic[numeric_cols].var()
    zero_var_cols = zero_var[zero_var == 0].index
    clinic = clinic.drop(columns=zero_var_cols)

    # --- Invert survival status (1 = alive / event did not occur) ---
    if "surv_status" in clinic.columns:
        clinic["surv_status"] = 1 - clinic["surv_status"]

    clinic = clinic.reset_index(drop=True)
    return clinic


# ---------------------------------------------------------------------------
# Cohort alignment
# ---------------------------------------------------------------------------

def align_cohort(clinic, delta_df, a_df, b_df):
    """
    Keep only patients present in all four DataFrames (intersection).
    Sort all DataFrames by id and validate row alignment.

    Parameters
    ----------
    clinic, delta_df, a_df, b_df : pd.DataFrame
        Each must contain an integer ID_COL column.

    Returns
    -------
    (clinic_c, delta_c, a_c, b_c) : aligned DataFrames.
    """
    # Enforce integer id in all frames
    for name, df in [("clinic", clinic), ("delta", delta_df), ("A", a_df), ("B", b_df)]:
        if ID_COL not in df.columns:
            raise ValueError(f"{name} DataFrame has no '{ID_COL}' column.")
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype(int)

    common_ids = sorted(
        set(clinic[ID_COL])
        & set(delta_df[ID_COL])
        & set(a_df[ID_COL])
        & set(b_df[ID_COL])
    )

    print(
        f"clinic N={clinic.shape[0]}, A N={a_df.shape[0]}, "
        f"B N={b_df.shape[0]}, Delta N={delta_df.shape[0]}"
    )
    print(f"Common cohort N={len(common_ids)}")

    if len(common_ids) < 10:
        print("WARNING: common cohort is very small; ML results will be unstable.")

    def _filter_sort(df):
        return (
            df[df[ID_COL].isin(common_ids)]
            .sort_values(ID_COL)
            .reset_index(drop=True)
        )

    clinic_c = _filter_sort(clinic)
    delta_c  = _filter_sort(delta_df)
    a_c      = _filter_sort(a_df)
    b_c      = _filter_sort(b_df)

    # Row-level alignment checks
    assert (clinic_c[ID_COL].values == a_c[ID_COL].values).all(), "A ids misaligned"
    assert (clinic_c[ID_COL].values == b_c[ID_COL].values).all(), "B ids misaligned"
    assert (clinic_c[ID_COL].values == delta_c[ID_COL].values).all(), "Delta ids misaligned"

    return clinic_c, delta_c, a_c, b_c


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_sets(clinic_c, delta_c, a_c, b_c):
    """
    Construct the label vector y and the four feature matrices used for modelling.

    Parameters
    ----------
    clinic_c, delta_c, a_c, b_c : aligned DataFrames from align_cohort().

    Returns
    -------
    y : pd.Series
        Integer label vector (surv_status).
    X_splits : dict
        Keys: 'clinical', 'clin+A', 'clin+B', 'clin+delta'
        Values: pd.DataFrame (feature matrix, id column excluded).
    """
    # --- Label ---
    label_col = next((c for c in LABEL_CANDIDATES if c in clinic_c.columns), None)
    if label_col is None:
        raise ValueError(
            f"Label column not found. Tried: {LABEL_CANDIDATES}. "
            f"Available columns: {list(clinic_c.columns)}"
        )

    y = clinic_c[label_col].astype(int).copy()
    print(f"Label column: '{label_col}' | class balance: {y.value_counts().to_dict()}")

    # --- Clinical features (no id, no label, no raw date cols) ---
    clin_feature_cols = [
        c for c in clinic_c.columns
        if c not in [ID_COL, label_col] + RAW_DATE_COLS
    ]
    non_numeric = (
        clinic_c[clin_feature_cols]
        .select_dtypes(exclude=[np.number])
        .columns.tolist()
    )
    if non_numeric:
        raise ValueError(f"Non-numeric clinical features: {non_numeric}")

    X_clin = clinic_c[clin_feature_cols].copy()

    # --- Radiomics feature matrices (drop id column) ---
    def _features_only(df, name):
        feat = df.drop(columns=[ID_COL]).copy()
        bad = feat.select_dtypes(exclude=[np.number]).columns.tolist()
        if bad:
            raise ValueError(f"Non-numeric columns in {name}: {bad}")
        return feat

    X_A = _features_only(a_c, "A")
    X_B = _features_only(b_c, "B")
    X_D = _features_only(delta_c, "delta")

    print(
        f"X_clin: {X_clin.shape} | X_A: {X_A.shape} | "
        f"X_B: {X_B.shape} | X_D: {X_D.shape}"
    )

    X_splits = {
        "clinical":   X_clin,
        "clin+A":     pd.concat([X_clin, X_A], axis=1),
        "clin+B":     pd.concat([X_clin, X_B], axis=1),
        "clin+delta": pd.concat([X_clin, X_D], axis=1),
    }

    return y, X_splits
