from snakemake.script import snakemake
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(snakemake.input.dataset)

if "target" not in df.columns:
    raise ValueError("Dataset must contain 'target' column.")

test_size = float(snakemake.params.test_size)
random_state = int(snakemake.params.random_state)

train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df["target"],
)

Path(snakemake.output.train).parent.mkdir(parents=True, exist_ok=True)
train_df.to_csv(snakemake.output.train, index=False)
test_df.to_csv(snakemake.output.test, index=False)