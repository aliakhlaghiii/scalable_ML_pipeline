from snakemake.script import snakemake
from pathlib import Path
import sys
import pandas as pd

# project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_paths
from data_io import calculate_delta_radiomics


def clean_radiomics_df(df: pd.DataFrame) -> pd.DataFrame:
    # keep only fully-filled columns across all patients
    df = df.copy()
    df.dropna(axis=1, how="any", inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id"}, inplace=True)
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    return df


# config path is passed by snakemake (recommended) or defaults inside config module
data_folder_path, clinical_path = load_paths(snakemake.params.config)

delta_df, a_df, b_df = calculate_delta_radiomics(data_folder_path)

delta_df = clean_radiomics_df(delta_df)
a_df = clean_radiomics_df(a_df).add_suffix("_a")
b_df = clean_radiomics_df(b_df).add_suffix("_b")

# restore id column names after suffixing
if "id_a" in a_df.columns:
    a_df = a_df.rename(columns={"id_a": "id"})
if "id_b" in b_df.columns:
    b_df = b_df.rename(columns={"id_b": "id"})

Path(snakemake.output.delta).parent.mkdir(parents=True, exist_ok=True)

delta_df.to_parquet(snakemake.output.delta, index=False)
a_df.to_parquet(snakemake.output.a, index=False)
b_df.to_parquet(snakemake.output.b, index=False)