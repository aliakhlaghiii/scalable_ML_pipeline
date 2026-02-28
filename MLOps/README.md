# Radiomics + Clinical Modeling (SVM / Random Forest)

End-to-end machine learning pipeline to evaluate whether **delta radiomics** (Time B − Time A) adds predictive value beyond **clinical** variables and single-timepoint radiomics features. Compares SVM (RBF) and Random Forest models across four feature configurations using scikit-learn pipelines and GridSearchCV.

> ⚠️ **Important limitation:** the test set contains only **6 patients (3 per class)**. All test-set metrics are highly unstable and should be interpreted cautiously.

------------------------------------------------------------------------

## Project structure

```         
├── config/
│   ├── config.py            # YAML loader and path validator
│   └── config.yaml          # Paths to data (edit before running)
├── data/
│   ├── modelling_data/      # Per-patient radiomics subfolders
│   └── clinical/            # Clinical Excel file
├── data_io.py               # Reads per-patient radiomics Excel files, computes delta
├── preprocessing.py         # Radiomics cleaning, clinical cleaning,
│                            # cohort alignment, feature matrix construction
├── models.py                # SVM/RF pipelines + GridSearchCV training/evaluation
├── plots.py                 # Heatmaps and confusion matrix plots
├── run.py                   # Entry point — runs the full pipeline end-to-end
├── MODEL_CARD.md            # Model card (EU AI Act transparency documentation)
└── REFLECTION_ESSAY.md      # Reflection essay on EU AI Act transparency
```

------------------------------------------------------------------------

## Feature configurations evaluated

| Feature set  | Contents                                                 |
|--------------|----------------------------------------------------------|
| `clinical`   | Clinical variables only                                  |
| `clin+A`     | Clinical + PET radiomics at time A (baseline)            |
| `clin+B`     | Clinical + PET radiomics at time B (pre-lymphodepletion) |
| `clin+delta` | Clinical + delta radiomics (B − A)                       |

------------------------------------------------------------------------

## Setup

``` bash
pip install numpy pandas pyyaml matplotlib scikit-learn openpyxl
```

Edit `config/config.yaml` to point to your data:

``` yaml
paths:
  data_folder: "/path/to/modelling_data"
  clinical_data: "/path/to/clinical_data.xlsx"
```

------------------------------------------------------------------------

## Run

``` bash
# Print results to terminal + show plots interactively
python run.py --config config/config.yaml

# Save metrics and plots to outputs/
python run.py --config config/config.yaml --outdir outputs --verbose
```

Outputs saved to `outputs/`: - `metrics.json` — all test-set metrics and best hyperparameters per model and feature set - `heatmap_accuracy.png`, `heatmap_balanced_accuracy.png`, `heatmap_f1.png`, `heatmap_roc_auc.png` - `confusion_matrices.png`

------------------------------------------------------------------------

## Data format expected

-   `data_folder/` contains one subfolder per patient (folder name = patient ID)
-   Each patient folder contains two `.xlsx` files: one with `_A` in the filename (baseline) and one with `_B` (pre-lymphodepletion)
-   Each Excel file has metadata in columns 0–22 and numeric radiomics features from column 23 onward
-   Each file must have a `Segmentation` column; the first row containing `suv2.5` is used
-   Clinical Excel file must contain a `record_id` column and a `surv_status` column (label)

------------------------------------------------------------------------

## References

-   Mitchell et al. (2019). *Model Cards for Model Reporting.* https://doi.org/10.1145/3287560.3287596
-   Linardatos et al. (2021). *Explainable AI: A Review of Machine Learning Interpretability Methods.* Entropy 23(1). <https://doi.org/10.3390/e23010018>
-   This project was developed with assistance from ChatGPT for code refinement, debugging, and documentation support.
------------------------------------------------------------------------

## Author

Ali Akhlaghi\
[a.akhlaghi\@st.hanze.nl](mailto:a.akhlaghi@st.hanze.nl)
