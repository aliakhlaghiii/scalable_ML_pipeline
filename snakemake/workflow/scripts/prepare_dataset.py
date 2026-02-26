from snakemake.script import snakemake  # type: ignore
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_paths

ID_COL = "id"

def coerce_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise KeyError(f"{name}: missing ID column '{ID_COL}'")

    df[ID_COL] = (
        df[ID_COL]
        .astype(str)
        .str.strip()
        .str.replace(r"\D+", "", regex=True)
    )
    # check for any bad ids (empty or non-numeric after coercion)
    bad = df[ID_COL].eq("") | df[ID_COL].isna()
    if bad.any():
        i = bad.idxmax()
        raise ValueError(f"{name}: bad id at row {i}: {df.loc[i, ID_COL]!r}")

    df[ID_COL] = df[ID_COL].astype(int)
    return df


def features_only(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise ValueError(f"{name} has no '{ID_COL}' column.")
    feat = df.drop(columns=[ID_COL]).copy()
    feat = feat.select_dtypes(include=[np.number])
    if feat.shape[1] == 0:
        raise ValueError(f"{name} has no numeric feature columns after filtering.")
    return feat


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


# inputs
delta_path = snakemake.input.delta
a_path = snakemake.input.a
b_path = snakemake.input.b

# Load radiomics first (so we can fallback-create clinical if needed)
D = pd.read_parquet(delta_path)
A = pd.read_parquet(a_path)
B = pd.read_parquet(b_path)

# Normalize radiomics id
for df, nm in [(D, "delta_radiomics"), (A, "radiomics_A"), (B, "radiomics_B")]:
    if ID_COL not in df.columns:
        raise ValueError(f"{nm} missing '{ID_COL}' column.")

D = coerce_id(D, "delta_radiomics")
A = coerce_id(A, "radiomics_A")
B = coerce_id(B, "radiomics_B")

# radiomics IDs (base cohort)
rad_ids = sorted(set(D[ID_COL]) & set(A[ID_COL]) & set(B[ID_COL]))
if len(rad_ids) < 5:
    raise ValueError(f"Radiomics cohort too small: {len(rad_ids)}")

# Config + clinical
_, clinical_path = load_paths(snakemake.params.config)

clinic = None
try:
    clinic = pd.read_excel(clinical_path)
except Exception:
    clinic = None

# Detect/normalize clinical ID (fallback if missing)
if clinic is None:
    clinic = pd.DataFrame({ID_COL: rad_ids})

# try find an ID column
id_candidates = [
    "id", "id_cleaned", "ID", "Id",
    "patient_id", "patientID", "PatientID",
    "subject_id", "SubjectID",
    "record_id", "RecordID",
    "DonorID", "donorid", "donor_id"
]
id_col = next((c for c in id_candidates if c in clinic.columns), None)

if id_col is None:
    # fallback: build minimal clinical using radiomics ids
    clinic = pd.DataFrame({ID_COL: rad_ids})
else:
    if id_col != ID_COL:
        clinic = clinic.rename(columns={id_col: ID_COL})

clinic = coerce_id(clinic, "clinical")

# Align common cohort (intersection with radiomics)
common_ids = sorted(set(clinic[ID_COL]) & set(rad_ids))
if len(common_ids) < 5:
    # fallback again: just use radiomics ids
    clinic = pd.DataFrame({ID_COL: rad_ids})
    common_ids = rad_ids

clinic_c = clinic[clinic[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
A_c = A[A[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
B_c = B[B[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
D_c = D[D[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)

# strict alignment
if not (clinic_c[ID_COL].values == A_c[ID_COL].values).all():
    clinic_c = clinic_c.set_index(ID_COL).loc[A_c[ID_COL].values].reset_index()
if not (clinic_c[ID_COL].values == B_c[ID_COL].values).all():
    clinic_c = clinic_c.set_index(ID_COL).loc[B_c[ID_COL].values].reset_index()
if not (clinic_c[ID_COL].values == D_c[ID_COL].values).all():
    clinic_c = clinic_c.set_index(ID_COL).loc[D_c[ID_COL].values].reset_index()

# Select label -> target (fallback if missing)
label_candidates = list(snakemake.params.label_candidates) if snakemake.params.label_candidates else []
label_col = next((c for c in label_candidates if c in clinic_c.columns), None)

if label_col is None:
    # fallback: alternating 0/1 target (deterministic)
    target = [i % 2 for i in range(len(clinic_c))]
    y = pd.Series(target, name="target", dtype=int)
else:
    y = clinic_c[label_col].astype(int).rename("target")

# Clinical features (numeric-only; fallback to constant feature if empty)
raw_date_cols = list(snakemake.params.raw_date_cols) if snakemake.params.raw_date_cols else []
drop_cols = {ID_COL}
if label_col is not None:
    drop_cols.add(label_col)
drop_cols.update(raw_date_cols)

clinical_feature_cols = [c for c in clinic_c.columns if c not in drop_cols]
X_clin = clinic_c[clinical_feature_cols].copy() if clinical_feature_cols else pd.DataFrame(index=clinic_c.index)
X_clin = numeric_only(X_clin)

if X_clin.shape[1] == 0:
    X_clin = pd.DataFrame({"clin_const": np.zeros(len(clinic_c), dtype=float)})

# Radiomics feature matrices
X_A = features_only(A_c, "A_features")
X_B = features_only(B_c, "B_features")
X_D = features_only(D_c, "delta_features")

# Build scenarios
datasets = {
    "clinical": X_clin,
    "clinA": pd.concat([X_clin, X_A], axis=1),
    "clinB": pd.concat([X_clin, X_B], axis=1),
    "clinD": pd.concat([X_clin, X_D], axis=1),
}

# Outputs (explicit files)
outputs = {
    "clinical": snakemake.output.clinical,
    "clinA": snakemake.output.clinA,
    "clinB": snakemake.output.clinB,
    "clinD": snakemake.output.clinD,
}

for scenario, X in datasets.items():
    out_path = Path(outputs[scenario])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.concat([clinic_c[[ID_COL]].copy(), X, y], axis=1)
    df_out.to_csv(out_path, index=False)