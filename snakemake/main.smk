# main.smk
# Snakemake workflow for integromics-style pipeline (assignment-ready)

configfile: "config/config.yaml"

MODELS = ["svm", "rf"]
SCENARIOS = ["clinical", "clinA", "clinB", "clinD"]


rule all:
    input:
        # radiomics outputs
        "results/radiomics/delta.parquet",
        "results/radiomics/A.parquet",
        "results/radiomics/B.parquet",

        # prepared datasets (one per scenario)
        expand("results/datasets/{scenario}.csv", scenario=SCENARIOS),
        
        # train/test splits
        expand("results/splits/{scenario}/train.csv", scenario=SCENARIOS),
        expand("results/splits/{scenario}/test.csv", scenario=SCENARIOS),

        # metrics (final deliverables for assignment)
        expand("results/metrics/svm/{scenario}.json", scenario=SCENARIOS),
        expand("results/metrics/rf/{scenario}.json", scenario=SCENARIOS),

        # models (optional but nice to keep as final artifacts)
        expand("results/models/svm/{scenario}.joblib", scenario=SCENARIOS),
        expand("results/models/rf/{scenario}.joblib", scenario=SCENARIOS)


rule build_radiomics:
    output:
        delta="results/radiomics/delta.parquet",
        a="results/radiomics/A.parquet",
        b="results/radiomics/B.parquet"
    params:
        config="config/config.yaml"
    log:
        "logs/build_radiomics.log"
    threads: 1
    resources:
        mem_mb=4000
    script:
        "workflow/scripts/build_radiomics.py"


rule prepare_dataset:
    input:
        delta="results/radiomics/delta.parquet",
        a="results/radiomics/A.parquet",
        b="results/radiomics/B.parquet"
    output:
        clinical="results/datasets/clinical.csv",
        clinA="results/datasets/clinA.csv",
        clinB="results/datasets/clinB.csv",
        clinD="results/datasets/clinD.csv"
    params:
        config="config/config.yaml",
        label_candidates=config.get("clinical", {}).get("label_candidates", []),
        raw_date_cols=config.get("clinical", {}).get("raw_date_cols", [])
    log:
        "logs/prepare_dataset.log"
    threads: 1
    resources:
        mem_mb=8000
    script:
        "workflow/scripts/prepare_dataset.py"


rule split_dataset:
    input:
        dataset="results/datasets/{scenario}.csv"
    output:
        train="results/splits/{scenario}/train.csv",
        test="results/splits/{scenario}/test.csv"
    params:
        test_size=config.get("ml", {}).get("test_size", 0.2),
        random_state=config.get("ml", {}).get("random_state", 42)
    log:
        "logs/split_{scenario}.log"
    threads: 1
    resources:
        mem_mb=2000
    script:
        "workflow/scripts/split_dataset.py"


rule train_svm:
    input:
        train="results/splits/{scenario}/train.csv",
        test="results/splits/{scenario}/test.csv"
    output:
        model="results/models/svm/{scenario}.joblib",
        metrics="results/metrics/svm/{scenario}.json"
    log:
        "logs/train_svm_{scenario}.log"
    threads: 4
    resources:
        mem_mb=8000
    script:
        "workflow/scripts/train_svm.py"


rule train_rf:
    input:
        train="results/splits/{scenario}/train.csv",
        test="results/splits/{scenario}/test.csv"
    output:
        model="results/models/rf/{scenario}.joblib",
        metrics="results/metrics/rf/{scenario}.json"
    log:
        "logs/train_rf_{scenario}.log"
    threads: 4
    resources:
        mem_mb=8000
    script:
        "workflow/scripts/train_rf.py"