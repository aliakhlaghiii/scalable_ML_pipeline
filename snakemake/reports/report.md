# Snakemake ML Workflow Report
**Radiomics-Based Classification Pipeline — Assignment Report**

---

## 1. Refactoring Process

The original code consisted of **standalone Python scripts** (`data_io.py`, `models.py`, `build_radiomics.py`) that executed preprocessing and model training **sequentially**. Refactoring into Snakemake involved three steps:

1. **Modularisation**: each logical step was isolated into a standalone script under `workflow/scripts/`: `build_radiomics.py`, `prepare_dataset.py`, `split_dataset.py`, `train_svm.py`, `train_rf.py`.
2. **File-based I/O**: every script was adapted to read inputs and write outputs via `snakemake.input` / `snakemake.output`, so Snakemake can track all file dependencies.
3. **Configuration externalisation**: paths, random seeds, test size, and label column names were moved to `config/config.yaml` and injected via `snakemake.params`, making the workflow fully reproducible without touching code.

---

## 2. Workflow Design

The pipeline follows **Workflow Option C** (comparing different ML models). It consists of five rules that form a DAG:

| Rule | Description |
|---|---|
| `build_radiomics` | Reads paired pre/post PET-CT Excel files, computes delta features, outputs 3 Parquet files |
| `prepare_dataset` | Merges radiomics with clinical data into 4 feature scenarios (clinical, clinA, clinB, clinD) |
| `split_dataset` | Stratified 80/20 train–test split per scenario — runs **4× in parallel** via `{scenario}` wildcard |
| `train_svm` | RBF-SVM with 5-fold inner GridSearchCV — runs **4× in parallel** |
| `train_rf` | Random Forest with 5-fold inner GridSearchCV — runs **4× in parallel** |

The `{scenario}` wildcard and `expand()` make adding new scenarios a one-line change. The two preprocessing rules are sequential; from `split_dataset` onward, **8 jobs run concurrently** (4 scenarios × 2 models).

**Resource allocation**: training rules use `threads: 4` and `mem_mb: 8000`. The workflow is launched with `--cores 8` to avoid monopolising the shared Assemblix cluster.

**Error handling**: every rule has a `log:` directive; scripts raise explicit exceptions on missing columns or index misalignment, and Snakemake preserves failed outputs for inspection.

![DAG](snakemake/images/dag.png)
*Figure 1. Snakemake DAG — shared preprocessing feeds 8 parallel training jobs.*

---

## 3. Results Summary

The workflow trains **8 classifiers** (2 models × 4 scenarios) and evaluates each on a held-out 20% test set. Metrics (ROC-AUC, balanced accuracy, F1) are saved as JSON files under `results/metrics/`. The four scenarios test progressively richer feature sets: clinical-only serves as baseline, while clinA/clinB/clinD add pre-, post-, and delta-radiomics features respectively.

Because Snakemake only reruns rules whose inputs have changed, comparing a new model or scenario incurs only the marginal cost of that job — the main practical advantage over a monolithic script.

---

## 4. Challenges & Limitations

- **Small cohort**: with few patients, 5-fold inner CV can be unstable; nested CV would be more rigorous.
- **Clinical data variability**: the pipeline auto-detects ID columns and falls back to dummy targets for robustness, at the cost of transparency.
- **No SLURM yet**: the workflow currently targets a shared workstation; a `--cluster` profile for HPC submission is a planned extension.

---

## References

- Köster, J., & Rahmann, S. (2012). Snakemake—a scalable bioinformatics workflow engine. Bioinformatics, 28(19), 2520–2522.
- https://github.com/fenna/BFVM24PROGRAM6/tree/main/Snakemake
- https://github.com/rwedema/DSLS_snakemake
- Snakemake documentation: https://snakemake.github.io/
