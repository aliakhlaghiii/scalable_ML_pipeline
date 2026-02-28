# Model Card: Delta Radiomics + Clinical Response Prediction

**Author:** Ali Akhlaghi · [a.akhlaghi\@st.hanze.nl](mailto:a.akhlaghi@st.hanze.nl)\
**Version:** 0.1.0 · **Date:** February 2026\
**Repository:** `Programming_VI` (integromics project)

------------------------------------------------------------------------

## 1. Model Details

| Field | Details |
|------------------------------------|------------------------------------|
| **Model type** | Binary classifier (SVM-RBF, Random Forest) |
| **Framework** | scikit-learn 1.x — `Pipeline` + `GridSearchCV` |
| **Task** | Predict treatment response (responder vs. non-responder) |
| **Input** | Tabular: clinical variables and/or PET radiomics features |
| **Output** | Binary label (0 = non-responder, 1 = responder) + probability score |
| **License** | Academic / research use only |

### Algorithms

**SVM (RBF kernel):** Imputation → StandardScaler → SVC(kernel="rbf", class_weight="balanced", probability=True). Tuned hyperparameters: `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, 0.01, 0.1}`.

**Random Forest:** Imputation → StandardScaler → RandomForestClassifier(n_estimators=200, class_weight="balanced"). Tuned hyperparameters: `max_depth ∈ {2, 3}`, `min_samples_leaf ∈ {2, 4}`, `max_features ∈ {sqrt, 0.5}`.

Both pipelines use `SimpleImputer(strategy="median")` and are optimised via 5-fold stratified inner cross-validation (scoring: ROC-AUC).

------------------------------------------------------------------------

## 2. Intended Use

**Primary use:** Research tool to evaluate whether delta radiomics (change between two PET timepoints) adds predictive value over clinical variables alone for treatment response classification in a lymphoma cohort (Yescarta CAR-T, tFL study, UMCG).

**Intended users:** Clinical researchers and data scientists in an academic medical setting.

**Out-of-scope uses:** - Clinical decision-making or patient triage without prospective validation - Deployment in any real-time or production clinical system - Generalisation to cancer types, treatments, or imaging protocols not represented in this dataset

------------------------------------------------------------------------

## 3. Data

| Property | Detail |
|------------------------------------|------------------------------------|
| **Source** | UMCG — wide export, Yescarta-infused tFL study (10-16-2025) |
| **Imaging modality** | PET (SUV-based segmentation; keyword: `suv2.5`) |
| **Timepoints** | Time A (pre-treatment), Time B (post-treatment) |
| **Feature extraction** | PyRadiomics-style Excel outputs; features from column index 23 onward |
| **Clinical data** | Separate Excel file; numeric + boolean variables |
| **Label** | Binary treatment response (0/1) |
| **Total patients processed** | 30 (train: 24, test: 6) |
| **Class balance (test set)** | 3 non-responders, 3 responders (perfectly balanced) |
| **Split strategy** | Stratified train/test (80/20); inner 5-fold stratified CV on train set |

### Data preprocessing (`preprocessing.py`)

All preprocessing is implemented in `preprocessing.py` and applied before any model fitting:

-   **Radiomics:** first `suv2.5` row per patient per timepoint; columns with any NaN dropped; delta = B − A computed feature-wise; `_a` / `_b` suffixes added to avoid column name collisions
-   **Clinical:** `NE` placeholder replaced with NaN; factor/comment/location/date columns dropped; `indication_dis_diagnosis` one-hot encoded; columns with \>50% missing values dropped; zero-variance features dropped; `surv_status` inverted (1 = alive)
-   **Missing values:** median imputation inside sklearn pipeline (fitted on train set only, applied to test — no leakage)
-   **Cohort alignment:** intersection of patient IDs across all four sources; row order validated before splitting

------------------------------------------------------------------------

## 4. Feature Configurations Evaluated

Four feature sets were compared to isolate the incremental value of radiomics:

| Feature set  | Contents                           |
|--------------|------------------------------------|
| `clinical`   | Clinical variables only            |
| `clin+A`     | Clinical + PET radiomics at time A |
| `clin+B`     | Clinical + PET radiomics at time B |
| `clin+delta` | Clinical + delta radiomics (B − A) |

------------------------------------------------------------------------

## 5. Performance

> ⚠️ **Critical caveat:** The test set contains only **6 patients (3 per class)**. All metrics below are therefore highly unstable. A single misclassification changes F1 or accuracy by \~0.17. These results are exploratory and must not be used for clinical inference.

### SVM Results

| Feature set | Accuracy | Balanced Acc. | F1    | ROC-AUC | Best params    |
|-------------|----------|---------------|-------|---------|----------------|
| clinical    | 0.500    | 0.500         | 0.667 | 0.222   | C=0.1, γ=0.01  |
| clin+A      | 0.500    | 0.500         | 0.667 | 0.444   | C=0.1, γ=scale |
| clin+B      | 0.667    | 0.667         | 0.667 | 0.111   | C=1, γ=scale   |
| clin+delta  | 0.833    | 0.833         | 0.857 | 0.222   | C=10, γ=scale  |

### Random Forest Results

| Feature set | Accuracy | Balanced Acc. | F1    | ROC-AUC | Best params                |
|------------|------------|------------|------------|------------|------------|
| clinical    | 0.667    | 0.667         | 0.667 | 0.667   | depth=3, feat=sqrt, leaf=2 |
| clin+A      | 0.833    | 0.833         | 0.857 | 0.667   | depth=3, feat=0.5, leaf=2  |
| clin+B      | 0.333    | 0.333         | 0.333 | 0.444   | depth=3, feat=sqrt, leaf=2 |
| clin+delta  | 0.833    | 0.833         | 0.857 | 0.667   | depth=2, feat=0.5, leaf=2  |

### Confusion Matrices — Best Configurations

**SVM · clin+delta** `[[2, 1], [0, 3]]` → TN=2, FP=1, FN=0, TP=3\
**RF · clin+delta** `[[2, 1], [0, 3]]` → TN=2, FP=1, FN=0, TP=3\
**SVM · clinical** `[[0, 3], [0, 3]]` → predicts all as positive (degenerate)\
**RF · clinical** `[[2, 1], [1, 2]]` → balanced errors

### Key observations

The `clin+delta` feature set achieves the highest balanced accuracy and F1 for both models (0.833 / 0.857), suggesting that the change in radiomic features between timepoints carries predictive signal beyond single-timepoint or clinical variables alone. However, the SVM ROC-AUC for `clin+delta` is only 0.222, indicating poor probability calibration despite good hard-label predictions — a direct consequence of the tiny test set. The RF shows more consistent AUC (0.667) across the best-performing feature sets.

------------------------------------------------------------------------

## 6. Limitations

-   **Extremely small test set (n=6):** No result can be considered statistically reliable. Confidence intervals would span nearly the full [0, 1] range for all metrics.
-   **No external validation:** The model has been evaluated only on a held-out subset of the same dataset. Generalisation to other sites or protocols is unknown.
-   **Feature leakage risk:** Delta features encode post-treatment information (time B). In a prospective setting, time B may not be available at the point of prediction.
-   **Missing data handling:** Patients with incomplete A or B timepoint files are silently dropped, which may introduce selection bias if missingness is not random.
-   **No feature selection or dimensionality reduction:** The number of radiomics features likely exceeds the number of patients in some configurations, creating a high-dimensional regime where overfitting is expected.
-   **Single random seed:** Results are not averaged over multiple seeds or splits; variance of metrics is unknown.

------------------------------------------------------------------------

## 7. Bias and Fairness

-   The cohort is from a single centre (UMCG) and a specific treatment protocol (Yescarta CAR-T, tFL). Demographic subgroup analysis (age, sex, disease stage) was not performed due to the small sample size.
-   Class weights are set to `"balanced"` in both SVM and RF to mitigate label imbalance during training, but this does not guarantee fair performance across demographic subgroups.
-   No audit for imaging acquisition bias (scanner type, reconstruction protocol) was performed.

------------------------------------------------------------------------

## 8. Ethical Considerations and EU AI Act Relevance

This system, if deployed in a clinical support context, would fall under **Annex III, Category 5(a)** of the EU AI Act as a high-risk AI system (AI in medical devices / clinical decision support). The following obligations would apply:

-   **Transparency (Art. 13):** This model card constitutes part of the required technical documentation. Intended use, data provenance, and limitations are disclosed.
-   **Human oversight (Art. 14):** Any clinical use must remain under the supervision of a qualified clinician. The model output is not sufficient for autonomous decision-making.
-   **Accuracy, robustness, and cybersecurity (Art. 15):** The current model does not meet accuracy thresholds for clinical deployment. Prospective validation on a larger, multi-centre cohort is required.
-   **Data governance (Art. 10):** Patient data is handled under UMCG institutional data governance. The model was trained on pseudonymised data.

------------------------------------------------------------------------

## 9. Reproducibility

``` bash
# Install dependencies
pip install numpy pandas pyyaml matplotlib scikit-learn openpyxl

# Edit config.yaml to point to your data, then:
python run.py --config config/config.yaml --outdir outputs --verbose
```

**Code structure:**

| File | Role |
|------------------------------------|------------------------------------|
| `config.py` | YAML loader and path validator |
| `data_io.py` | Reads per-patient radiomics Excel files, computes delta |
| `preprocessing.py` | Radiomics cleaning, clinical cleaning, cohort alignment, feature matrix construction |
| `models.py` | SVM/RF pipelines + GridSearchCV training/evaluation |
| `plots.py` | Heatmaps and confusion matrix plots |
| `run.py` | Entry point — runs the full pipeline end-to-end |

**Random seed:** 42 (all model constructors and train/test split)\
**Train/test split:** 80/20 stratified\
**Inner CV:** 5-fold StratifiedKFold, shuffle=True, seed=42

Outputs written to `outputs/`: `metrics.json`, heatmap PNGs, `confusion_matrices.png`.

------------------------------------------------------------------------

## 10. References

-   Mitchell et al. (2019). *Model Cards for Model Reporting.* FAT\* '19. ACM. <https://doi.org/10.1145/3287560.3287596>
-   Linardatos et al. (2021). *Explainable AI: A Review of Machine Learning Interpretability Methods.* Entropy 23(1). <https://doi.org/10.3390/e23010018>
-   Martinez & Kifle (2024). *Artificial Intelligence: A Systems Approach.* MIT Press.
