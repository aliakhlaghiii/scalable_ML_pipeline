from snakemake.script import snakemake
import sys
from pathlib import Path

# Add project root to sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import load_paths
from data_io import calculate_delta_radiomics


# Read paths from YAML (passed via Snakemake params)
data_folder, _clinical_path = load_paths(snakemake.params.config)

# Run core logic from original data_io.py
delta_df, A_df, B_df = calculate_delta_radiomics(data_folder)

# Write outputs tracked by Snakemake
delta_df.to_csv(snakemake.output.delta, index=True)
A_df.to_csv(snakemake.output.a, index=True)
B_df.to_csv(snakemake.output.b, index=True)