# Anesthesia Outcome Prediction with Machine Learning

> An end-to-end, leakage-aware machine learning project for exploring whether preoperative patient and procedure characteristics can predict postoperative complications.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project investigates the use of machine learning to predict a binary anesthesia-related outcome from a small tabular dataset containing demographic, surgical, anesthesia, and preoperative information.

The project was designed as an end-to-end data science workflow rather than a single model-training exercise. It covers data auditing, statistical analysis, leakage detection, feature engineering, preprocessing pipelines, multiple machine learning algorithms, cross-validation, hyperparameter tuning, model evaluation, feature importance, SHAP-based explainability, and deployment with Streamlit.

A central theme of the project is **responsible evaluation**. The dataset contains variables that are closely related to, or collected after, the target outcome. These variables were investigated and excluded from the preoperative prediction pipeline to reduce target leakage and make the prediction task more realistic.

## Problem Statement

The target variable is a binary `Outcome` indicating whether complications are present:

- `0`: No complications
- `1`: Complications present

The modeling question is:

> **Can postoperative complications be predicted using information that would be available before or around the time of surgery, without using postoperative outcome information that would leak the target?**

This is an exploratory machine learning project. The resulting model is **not a clinical decision-support system** and must not be used for patient care.

## Dataset

The project uses the **Personalized Anesthesia Management Dataset** published on Kaggle:

https://www.kaggle.com/datasets/s3programmerlead/personalized-anesthesia-management-dataset

The dataset contains:

- 300 patient records
- 12 original columns
- A balanced binary target: 150 samples in each outcome class
- Demographic, procedural, anesthesia, preoperative, postoperative, and complication-related variables

### Original Features

| Feature | Description | Initial Role |
|---|---|---|
| `PatientID` | Patient identifier | Identifier, excluded from modeling |
| `Age` | Patient age | Candidate feature |
| `Gender` | Patient gender | Candidate feature |
| `BMI` | Body mass index | Candidate feature |
| `SurgeryType` | Type of surgery | Candidate feature |
| `SurgeryDuration` | Surgery duration stored as text such as `217 min` | Candidate feature, transformed |
| `AnesthesiaType` | Local or general anesthesia | Candidate feature |
| `PreoperativeNotes` | Preoperative clinical description | Candidate feature, transformed |
| `PostoperativeNotes` | Postoperative description | Excluded from preoperative prediction |
| `PainLevel` | Pain score | Excluded from preoperative prediction |
| `Complications` | Recorded complications | Excluded due to leakage |
| `Outcome` | Binary target | Target |

### Data Quality Findings

The initial audit found:

- 300 rows and 12 columns
- No duplicated rows
- No duplicated `PatientID` values
- No missing values in the candidate preoperative features after feature selection
- 73 missing values in `Complications`
- Balanced target classes: 50% `Outcome=0` and 50% `Outcome=1`
- `SurgeryDuration` required conversion from string to integer minutes
- Several variables had very limited unique values, which is an important limitation of the dataset

## Data Leakage Analysis

One of the main goals of this project was to prevent information that would not be available at prediction time from entering the model.

The following variables were excluded from the preoperative prediction pipeline:

### `Complications`

`Complications` is directly related to the target definition and therefore can act as a target proxy. Using it as an input would allow the model to recover the target from information that essentially defines the outcome.

### `PostoperativeNotes`

These notes describe postoperative status. Using them to predict postoperative complications would introduce information from after the event of interest.

### `PainLevel`

The dataset documentation describes this as postoperative pain. It was therefore excluded from the preoperative prediction task.

### `PatientID`

`PatientID` is an identifier rather than a clinically meaningful predictor and was excluded to avoid learning meaningless record-specific patterns.

This distinction is important because a model can achieve attractive metrics while solving the wrong problem. The project therefore prioritizes a **leakage-aware pipeline over artificially high performance**.

## Exploratory Data Analysis

The exploratory analysis covered:

- Dataset dimensions and schema
- Data types
- Missing-value patterns
- Duplicate detection
- Unique-value analysis
- Numerical summaries
- Target distribution
- Feature distributions by outcome
- Categorical distributions by outcome
- Surgery-duration analysis

Examples of observed distributions include:

- `Age`: 33–72 years, with only six unique age values
- `BMI`: 23–32, with only six unique BMI values
- `PainLevel`: 2–7, with six unique values; excluded from preoperative modeling because it represents postoperative information.
- `SurgeryDuration`: 60–240 minutes, with 139 unique values
- `Gender`: two categories
- `SurgeryType`: four categories
- `AnesthesiaType`: two categories

These patterns suggest that the dataset is relatively small and constrained, which must be considered when interpreting model performance.

## Statistical Analysis

Univariate statistical tests were performed before machine learning to characterize associations between candidate preoperative features and the target.

### Numerical Features

The Mann–Whitney U test was used for numerical comparisons between the two outcome groups.

| Feature | p-value | Effect Size (Rank-Biserial) |
|---|---:|---:|
| `Age` | 0.8713 | -0.0107 |
| `BMI` | 0.3147 | 0.0662 |
| `SurgeryDuration_min` | 0.8014 | 0.0168 |

No tested numerical feature showed statistically significant evidence of a difference at the 0.05 level.

### Categorical Features

Chi-square tests of independence and Cramér's V were used for categorical variables.

| Feature | Chi-square | p-value | Cramér's V |
|---|---:|---:|---:|
| `Gender` | 0.8543 | 0.3553 | 0.0534 |
| `SurgeryType` | 6.4275 | 0.0926 | 0.1464 |
| `AnesthesiaType` | 0.8534 | 0.3556 | 0.0533 |

No categorical feature reached the conventional 0.05 significance threshold. `SurgeryType` showed the largest categorical association in this analysis, but the evidence was still insufficient to characterize the association as statistically significant at the 5% level.

Statistical significance was not used as an automatic feature-selection rule. Machine learning can capture interactions and nonlinear patterns that univariate tests may not detect.

## Feature Engineering

### Surgery Duration

The original `SurgeryDuration` field was stored as text, for example:

```text
217 min
181 min
79 min
```

It was transformed into an integer feature:

```text
SurgeryDuration_min
```

### Preoperative Notes

The dataset contains two recurring preoperative note patterns. Instead of applying a complex NLP model to a very small vocabulary, the notes were converted to a simple binary feature:

```text
Has_Comorbidities
```

- `1`: hypertension or diabetes mentioned in the preoperative notes.
- `0`: those conditions were not mentioned in the notes.

This keeps the transformation interpretable and avoids overstating the NLP capabilities of a dataset that is too small and repetitive for meaningful text modeling.

## Modeling Features

The final preoperative modeling feature set is:

```text
Age
BMI
SurgeryDuration_min
Has_Comorbidities
Gender
SurgeryType
AnesthesiaType
```

The target is:

```text
Outcome
```

## Machine Learning Pipeline

The project uses scikit-learn pipelines to ensure preprocessing is fitted only on the relevant training data and is applied consistently during validation and testing.

### Numerical preprocessing

Numerical features are standardized using `StandardScaler` for models such as Logistic Regression.

### Categorical preprocessing

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This prevents unseen categories from causing failures during inference.

### ColumnTransformer

A `ColumnTransformer` applies the appropriate transformation to each feature group.

The resulting training matrix contains 12 model features:

```text
4 numerical features
+
2 gender features
+
4 surgery-type features
+
2 anesthesia-type features
=
12 encoded features
```

## Models Evaluated

Four model families were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

All baseline comparisons used the same training data, preprocessing strategy, and stratified 5-fold cross-validation procedure.

## Cross-Validation Results

The baseline 5-fold cross-validation results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4917 ± 0.0429 | 0.5044 ± 0.0594 | 0.4333 ± 0.0726 | 0.4576 ± 0.0245 | 0.5188 ± 0.0309 |
| Decision Tree | 0.4458 ± 0.0312 | 0.4542 ± 0.0256 | 0.5250 ± 0.0333 | 0.4864 ± 0.0235 | 0.4458 ± 0.0312 |
| Random Forest | 0.4708 ± 0.0312 | 0.4675 ± 0.0376 | 0.4750 ± 0.0898 | 0.4693 ± 0.0622 | 0.5068 ± 0.0114 |
| Gradient Boosting | 0.4375 ± 0.0295 | 0.4348 ± 0.0259 | 0.4083 ± 0.1000 | 0.4151 ± 0.0506 | 0.4181 ± 0.0504 |

The cross-validation results show that all models remained close to chance-level discrimination. This is consistent with the earlier statistical analysis and suggests that the available feature set contains limited predictive information for the target.

## Hyperparameter Tuning

Two promising baseline models were tuned using `GridSearchCV` with the same stratified 5-fold cross-validation framework.

### Logistic Regression

Best configuration:

```text
C = 10
class_weight = None
```

Best cross-validation ROC-AUC:

```text
0.5198
```

### Random Forest

Best configuration:

```text
n_estimators = 500
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```

Best cross-validation ROC-AUC:

```text
0.5208
```

The small improvement after tuning indicates that hyperparameter selection was not the primary limitation of the task.

## Final Test Evaluation

The tuned Random Forest was evaluated once on the held-out test set. The final Random Forest was selected based on cross-validation performance, and the held-out test was used only for final evaluation.

| Metric | Test Score |
|---|---:|
| Accuracy | **0.4500** |
| Precision | **0.4483** |
| Recall | **0.4333** |
| F1-score | **0.4407** |
| ROC-AUC | **0.5422** |
| Average Precision | **0.570** |

### Confusion Matrix

```text
[[14, 16],
 [17, 13]]
```

Interpreting the matrix as `[ [TN, FP], [FN, TP] ]`:

- True negatives: 14
- False positives: 16
- False negatives: 17
- True positives: 13

The model therefore failed to identify 17 of the 30 positive examples in the held-out test set.

## ROC and Precision–Recall Analysis

The ROC curve produced a test ROC-AUC of **0.5422**, only slightly above the 0.50 random-classifier reference.

The Precision–Recall analysis produced an Average Precision of **0.570**. Because the test set contains approximately 50% positive examples, a prevalence-based baseline is approximately 0.50.

Together, these results indicate limited discrimination rather than clinically useful prediction.

## Explainable AI

Model interpretability was explored using both Random Forest feature importance and SHAP.

### Random Forest Feature Importance

The model assigned the largest importance to:

- `SurgeryDuration_min`
- `BMI`
- `Age`
- `Has_Comorbidities`

with categorical dummy features generally contributing less.

These importance values describe how the fitted Random Forest used the available data. They should **not** be interpreted as causal effects or clinical risk factors.

### SHAP

SHAP was used to examine how individual feature values moved model predictions toward or away from `Outcome=1`.

The SHAP analysis similarly highlighted variables such as surgery duration, age, BMI, and comorbidity status as important to the fitted model.

Because the dataset is small and synthetic/restricted in its feature space, SHAP results should be treated as **model explanations for this dataset**, not as evidence of clinical causality.

## Streamlit Application

The project includes a Streamlit interface for interactive demonstration of the trained model.

The application accepts:

- Age
- BMI
- Gender
- Surgery type
- Surgery duration
- Anesthesia type
- Comorbidity status

and returns:

- Predicted outcome
- Predicted probability of `Outcome=1`

The app is intended only for research/portfolio demonstration.

### Run the App

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app/app.py
```

If needed:

```bash
python -m streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Project Structure

```text
anesthesia-ml-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── Anesthesia_Dataset.csv
│
├── models/
│   └── anesthesia_rf_model.pkl
│
├── notebooks/
│   └── 01_Data_Audit.ipynb
│
├── src/
│   └── preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Reproducibility

The modeling workflow uses fixed random seeds where appropriate, including:

```python
random_state=42
```

The train/test split is stratified to preserve the balanced outcome distribution, and model selection is performed with stratified 5-fold cross-validation.

Importantly, preprocessing is part of the scikit-learn `Pipeline` rather than being fitted on the full dataset before cross-validation. This reduces the risk of validation leakage.

## Limitations

This project has several important limitations.

### Small sample size

The dataset contains only 300 records. This limits statistical power and makes model estimates sensitive to sampling variation.

### Restricted feature diversity

Some variables have very few unique values. For example, `Age`, `BMI`, and `PainLevel` each contain only six unique values. This limits the realism and information content of the dataset.

### Dataset provenance

The dataset is published on Kaggle and appears to be a constrained/synthetic-style dataset rather than a large real-world clinical registry. Model performance should therefore not be generalized to real patient populations.

### Limited clinical variables

Important predictors that might influence postoperative complications are not represented in sufficient detail, such as richer comorbidity profiles, ASA classification, laboratory results, medications, intraoperative vital signs, blood loss, airway information, and detailed postoperative monitoring.

### Target definition and temporal structure

The outcome and several variables are closely linked to postoperative events. This makes careful definition of the prediction time point essential.

### No external validation

The model was not validated on an independent external clinical dataset.

### No clinical deployment

The model has not undergone clinical validation, prospective testing, regulatory review, calibration assessment in a real patient population, or safety evaluation.

## Key Takeaways

The main result of the project is not a high-performing predictive model. Instead, the analysis demonstrates a complete and leakage-aware machine learning workflow and shows that, with the available features and dataset size, several standard machine learning approaches remain close to chance-level performance.

This is an important modeling outcome: **a machine learning project does not become scientifically stronger by hiding weak results or introducing leakage to inflate metrics.**

The project therefore emphasizes:

- Careful data auditing
- Temporal reasoning and leakage prevention
- Statistical analysis before modeling
- Reproducible preprocessing pipelines
- Cross-validation rather than relying on a single split
- Multiple evaluation metrics
- Explainability without causal overinterpretation
- Honest reporting of limitations

## Future Work

Potential next steps include:

1. Acquire a larger, clinically richer dataset with a clearly defined prediction time point.
2. Add more informative preoperative variables such as ASA status, detailed comorbidities, medications, laboratory values, and physiological measurements.
3. Perform external validation on an independent dataset.
4. Evaluate model calibration and decision-curve analysis when clinically appropriate.
5. Build a second project around a genuinely large medical dataset suitable for deep learning, such as medical imaging or physiological time-series data.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd anesthesia-ml-project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the notebooks

Open the notebooks directory in Jupyter or VS Code and execute the notebooks in order.

### Run the Streamlit application

```bash
streamlit run app/app.py
```

## Dependencies

The project uses the following main Python libraries:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- shap
- joblib
- streamlit

See `requirements.txt` for the installable dependency list.

## Ethical and Clinical Disclaimer

This repository is an educational and portfolio project. It is **not** a medical device, diagnostic tool, treatment recommendation system, or validated clinical decision-support system.

Predictions produced by the model must not be used to make decisions about real patients.

The dataset and resulting model should not be interpreted as evidence that any individual feature causes postoperative complications.

## Author

**Ali Atri**

This project combines a background in anesthesia with practical training in data science and machine learning, with the goal of exploring responsible applications of AI to healthcare data.

## Acknowledgments

Dataset source:

**Personalized Anesthesia Management Dataset** — Kaggle

https://www.kaggle.com/datasets/s3programmerlead/personalized-anesthesia-management-dataset
