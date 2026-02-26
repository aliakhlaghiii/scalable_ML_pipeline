from snakemake.script import snakemake
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_paths


ID_COL = "id"


def ensure_numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    bad = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if bad:
        raise ValueError(f"Non-numeric columns in {name}: {bad}")
    return df


def features_only(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise ValueError(f"{name} has no '{ID_COL}' column.")
    feat = df.drop(columns=[ID_COL]).copy()
    return ensure_numeric(feat, name)


# inputs
delta_path = snakemake.input.delta
a_path = snakemake.input.a
b_path = snakemake.input.b

# config + clinical
data_folder_path, clinical_path = load_paths(snakemake.params.config)
clinic = pd.read_excel(clinical_path)

# normalize id column name like integromics notebook
if "id_cleaned" in clinic.columns and ID_COL not in clinic.columns:
    clinic = clinic.rename(columns={"id_cleaned": ID_COL})
if ID_COL not in clinic.columns:
    raise ValueError("Clinical file has no 'id' or 'id_cleaned' column.")

clinic[ID_COL] = pd.to_numeric(clinic[ID_COL], errors="raise").astype(int)

# radiomics
D = pd.read_parquet(delta_path)
A = pd.read_parquet(a_path)
B = pd.read_parquet(b_path)

# ensure id is int
for name, df in [("clinic", clinic), ("A", A), ("B", B), ("Delta", D)]:
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype(int)
    dup = df[ID_COL].duplicated().sum()
    if dup > 0:
        raise ValueError(f"{name}: {dup} duplicated ids found.")

# common cohort (intersection)
common_ids = sorted(set(clinic[ID_COL]) & set(A[ID_COL]) & set(B[ID_COL]) & set(D[ID_COL]))
if len(common_ids) < 5:
    raise ValueError(f"Common cohort too small: {len(common_ids)}")

clinic_c = clinic[clinic[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
A_c = A[A[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
B_c = B[B[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
D_c = D[D[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)

# alignment checks
assert (clinic_c[ID_COL].values == A_c[ID_COL].values).all()
assert (clinic_c[ID_COL].values == B_c[ID_COL].values).all()
assert (clinic_c[ID_COL].values == D_c[ID_COL].values).all()

# label selection (same idea as notebook)
label_candidates = snakemake.params.label_candidates
label_col = next((c for c in label_candidates if c in clinic_c.columns), None)
if label_col is None:
    raise ValueError(f"Label column not found. Tried: {label_candidates}")

y = clinic_c[label_col].astype(int).rename("target")

# clinical feature columns (drop raw dates too)
raw_date_cols = snakemake.params.raw_date_cols
clin_feature_cols = [
    c for c in clinic_c.columns
    if c not in [ID_COL, label_col] and c not in raw_date_cols
]
X_clin = clinic_c[clin_feature_cols].copy()
ensure_numeric(X_clin, "X_clin")

# radiomics matrices
X_A = features_only(A_c, "A_c")
X_B = features_only(B_c, "B_c")
X_D = features_only(D_c, "D_c")

# scenarios (exactly notebook’s 4 sets)
datasets = {
    "clinical": X_clin,
    "clinA": pd.concat([X_clin, X_A], axis=1),
    "clinB": pd.concat([X_clin, X_B], axis=1),
    "clinD": pd.concat([X_clin, X_D], axis=1),
}

out_dir = Path(snakemake.output.dir)
out_dir.mkdir(parents=True, exist_ok=True)

# write one CSV per scenario: includes id + features + target
for scenario, X in datasets.items():
    df_out = pd.concat([clinic_c[[ID_COL]].copy(), X, y], axis=1)
    df_out.to_csv(out_dir / f"{scenario}.csv", index=False)