# Radiomics-Based Machine Learning Workflow (Snakemake)
## Overview

This repository implements an end-to-end, reproducible machine learning workflow designed to evaluate whether PET-based radiomics features and in particular delta radiomics (Time B − Time A) provide additional predictive value beyond clinical variables for short-term survival prediction in patients with diffuse large B-cell lymphoma (DLBCL) treated with CAR T-cell therapy.

The workflow is fully orchestrated using Snakemake and includes:

- Radiomics feature aggregation (Time A, Time B, Delta)
- Clinical data integration and harmonization
- ID validation and preprocessing
- Dataset preparation and cleaning
- Train/test splitting
- Model training (Random Forest and Support Vector Machine)
- Performance evaluation
- Workflow visualization via DAG

The primary objective is methodological: to refactor a monolithic machine learning analysis into a modular, parallelized, and reproducible Snakemake workflow while preserving scientific clarity and robustness.

This assignment focuses primarily on workflow engineering and reproducible pipeline design rather than model performance optimization.

------------------------------------------------------------------------

# Objectives

-   Refactor a monolithic ML pipeline into modular Snakemake rules
-   Enable parallel execution
-   Manage computational resources
-   Ensure reproducibility
-   Implement structured error handling
-   Visualize workflow dependencies

------------------------------------------------------------------------

## Project Structure

```
snakemake/
├── .gitignore
├── README.md
├── LICENSE.md
├── data_io.py
├── main.smk
├── models.py
├── images/
│   └── dag.png
├── workflow/
│   ├── scripts/
│   │   ├── build_radiomics.py
│   │   └── prepare_dataset.py
|   |   └── split_dataset.py
|   |   └── train_rf.py
|   |   └── train_svm.py
├── config/
│   └── config.yaml
├── results/
└── resources/
```

------------------------------------------------------------------------

## Data Availability

The dataset used in this study contains sensitive clinical information and is not publicly available.

To run the workflow, users must provide their own data following the expected directory structure specified in `config/config.yaml`.

The workflow itself is fully reproducible once appropriate data are supplied.

------------------------------------------------------------------------

# Running the Workflow

Dry run:

    snakemake -s main.smk -n -p

Run full pipeline:

    snakemake -s main.smk --cores 8 --rerun-incomplete -p

Force rerun a rule:

    snakemake --forcerun prepare_dataset

------------------------------------------------------------------------

## Outputs

Pipeline outputs are generated locally under:

- `results/`

This directory is **not tracked in git** (it is listed in `.gitignore`) because it contains generated artifacts.

Typical outputs include:
- `results/radiomics/*.parquet` (radiomics features)
- `results/datasets/*.csv` (prepared datasets)
- `results/models/*` (trained models and metrics)
- `logs/*` (rule logs)

------------------------------------------------------------------------

# Workflow DAG

The workflow DAG is stored at:

images/dag.png

To regenerate:

    snakemake --dag | dot -Tpng > images/dag.png

Requires graphviz.

------------------------------------------------------------------------

# Reproducibility

-   Configuration separated from code
-   Deterministic ML seeds
-   Data excluded from repository
-   Fully declarative Snakemake pipeline

------------------------------------------------------------------------

# Dependencies

-   Python 3.12
-   Snakemake
-   pandas
-   scikit-learn
-   pyarrow
-   openpyxl
-   graphviz

Install example:

    pip install pandas scikit-learn pyarrow openpyxl

------------------------------------------------------------------------

## References

- Köster, J., & Rahmann, S. (2012). *Snakemake—a scalable bioinformatics workflow engine*. Bioinformatics, 28(19), 2520–2522.

---

## Acknowledgments

Parts of the workflow structure and organization were inspired by the following public repositories:

- https://github.com/fenna/BFVM24PROGRAM6/tree/main/Snakemake  
- https://github.com/rwedema/DSLS_snakemake  

This project was developed with assistance from ChatGPT for code refinement, debugging, and documentation support.

## Contact
- Author: Ali Akhlaghi
- Email: a.akhlaghi@st.hanze.nl
