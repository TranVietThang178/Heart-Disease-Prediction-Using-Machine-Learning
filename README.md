# Heart Disease Prediction Using Machine Learning

A comparative analysis of machine learning models for heart disease prediction
using clinical data from the UCI Heart Disease dataset. This project was developed
as part of the MSc in Data Science - Machine Learning module at ATU Donegal.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Notebooks](#notebooks)
5. [Models and Results](#models-and-results)
6. [Key Findings](#key-findings)
7. [Setup and Installation](#setup-and-installation)
8. [How to Run](#how-to-run)
9. [Team](#team)

---

## Project Overview

This project builds and compares three supervised machine learning classifiers
for binary heart disease prediction. The workflow covers the full machine learning
pipeline - from raw data ingestion through to model interpretability using SHAP.

The project addresses the clinical importance of minimising false negatives
(missed disease cases) and uses both ROC-AUC and F2 score as primary evaluation
metrics. F2 score weights recall twice as heavily as precision, reflecting the
real-world cost of an undetected diagnosis.

Techniques applied:

- Data preprocessing and group-based imputation across four clinical sources
- Exploratory data analysis with outlier detection and correlation analysis
- Feature engineering with one-hot encoding and StandardScaler
- Dimensionality reduction using PCA and LDA
- Supervised classification using Random Forest, Logistic Regression, and XGBoost
- Hyperparameter tuning using RandomizedSearchCV with 5-fold cross-validation
- Model evaluation with ROC curves, confusion matrices, and threshold analysis
- Model interpretability using SHAP (SHapley Additive exPlanations)

---

## Dataset

Source: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

The dataset combines clinical records from four institutions:

| Source | Patients |
|---|---|
| Cleveland Clinic Foundation | 293 |
| Hungarian Institute of Cardiology | 294 |
| University Hospital Zurich (Switzerland) | 123 |
| VA Medical Center, Long Beach | 200 |
| **Total after cleaning** | **908** |

The raw `.data` files were used directly, not the pre-processed versions.
Each file contains 76 attributes per patient. This project uses the 13 clinically
relevant features plus the binary target variable derived from the `num` column
(0 = no disease, 1 = disease present).

| Feature | Description | Type |
|---|---|---|
| age | Age in years | Continuous |
| sex | Sex (1 = male, 0 = female) | Binary |
| cp | Chest pain type (1-4) | Categorical |
| trestbps | Resting blood pressure (mmHg) | Continuous |
| chol | Serum cholesterol (mg/dl) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise-induced angina | Binary |
| oldpeak | ST depression induced by exercise | Continuous |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels coloured | Categorical |
| thal | Thalassemia type | Categorical |
| target | Heart disease present (1) or absent (0) | Binary |

Class distribution: 496 disease (54.6%) / 412 no disease (45.4%)

---

## Project Structure

```
Heart-Disease-Prediction-Using-Machine-Learning/
│
├── data/
│   ├── raw/                        # Original UCI .data files
│   │   ├── cleveland.data
│   │   ├── hungarian.data
│   │   ├── long-beach-va.data
│   │   └── switzerland.data
│   │
│   └── processed/
│       ├── heart_disease.csv       # Cleaned and merged dataset (908 rows)
│       ├── splits/                 # Train/test splits
│       │   ├── X_train_tree.csv    # Tree model features (RF, XGBoost)
│       │   ├── X_test_tree.csv
│       │   ├── X_train_lr.csv      # LR features (encoded + scaled)
│       │   ├── X_test_lr.csv
│       │   ├── y_train.csv
│       │   └── y_test.csv
│       └── results/                # Saved model metrics (JSON)
│           ├── dr_results.json
│           ├── rf_results.json
│           ├── lr_results.json
│           └── xgb_results.json
│
├── models/                         # Saved trained models
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_Prepocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Dimensionality_Reduction.ipynb
│   ├── 05_Random_Forest.ipynb
│   ├── 06_LR.ipynb
│   ├── 07_XGBoost.ipynb
│   ├── 08_Evaluation.ipynb
│   └── 09_SHAP.ipynb
│
├── requirements.txt
└── README.md
```

---

## Notebooks

| Notebook | Description | Owner |
|---|---|---|
| 01_Preprocessing | Parse raw UCI files, clean, impute, merge | Jeevan Manoj |
| 02_EDA | Distributions, outliers, correlation analysis | Viet Thang Tran |
| 03_Feature_Engineering | Encoding, scaling, train/test splits | Viet Thang Tran |
| 04_Dimensionality_Reduction | PCA and LDA analysis | Viet Thang Tran |
| 05_Random_Forest | Baseline and tuned RF classifier | Viet Thang Tran |
| 06_LR | Baseline, pipeline, and tuned LR classifier | Jeevan Manoj |
| 07_XGBoost | Baseline and tuned XGBoost classifier | Jeevan Manoj |
| 08_Evaluation | Full model comparison and threshold analysis | Shared |
| 09_SHAP | Feature importance and individual predictions | Viet Thang Tran |

Notebooks must be run in order (01 through 09) as each depends on outputs
from the previous step.

---

## Models and Results

All models were evaluated on an identical stratified 80/20 train/test split
(random_state=42). Hyperparameter tuning used RandomizedSearchCV with
5-fold cross-validation optimising for ROC-AUC.

### Performance Summary (Test Set)

| Model | Accuracy | ROC-AUC | F2 Score | False Negatives |
|---|---|---|---|---|
| LR on PCA (10 components) | 80.8% | 0.8950 | 0.833 | - |
| LR on LDA (1 component) | 81.9% | 0.8984 | 0.843 | - |
| Base Random Forest | 78.0% | 0.8745 | 0.777 | 23 |
| Base Logistic Regression | 81.9% | 0.9134 | 0.870 | 11 |
| Base XGBoost | 77.5% | - | 0.761 | - |
| Tuned Random Forest | 83.0% | 0.8977 | 0.860 | 13 |
| Tuned Logistic Regression | 82.4% | 0.9152 | 0.871 | 11 |
| Tuned XGBoost | 80.8% | 0.8910 | 0.813 | 19 |

### Best Hyperparameters

**Random Forest**
```
n_estimators: 100, max_depth: 10, max_leaf_nodes: 20,
max_features: sqrt, min_samples_split: 2, min_samples_leaf: 2
```

**Logistic Regression**
```
solver: saga, l1_ratio: 1.0, C: 0.2336, max_iter: 2000
```

**XGBoost**
```
n_estimators: 500, max_depth: 11, learning_rate: 0.1,
subsample: 0.6, colsample_bytree: 0.6,
reg_alpha: 5.0, reg_lambda: 2.0, gamma: 0.5
```

### Threshold Analysis

Lowering the classification threshold from 0.5 increases recall
at the cost of accuracy - clinically justified for disease screening.

| Model | Threshold | Accuracy | Recall | F2 | False Negatives |
|---|---|---|---|---|---|
| Random Forest | 0.3 | 76.4% | 0.960 | 0.896 | 4 |
| Random Forest | 0.5 | 83.0% | 0.869 | 0.860 | 13 |
| Logistic Regression | 0.3 | 80.2% | 0.939 | 0.896 | 6 |
| Logistic Regression | 0.5 | 82.4% | 0.889 | 0.871 | 11 |
| XGBoost | 0.3 | 79.7% | 0.919 | 0.882 | 8 |
| XGBoost | 0.5 | 80.8% | 0.808 | 0.813 | 19 |

---

## Key Findings

**Logistic Regression is the strongest model** on this dataset despite being
the simplest, achieving the highest ROC-AUC (0.9152) and fewest false negatives
(11) at the default threshold. This suggests the relationship between clinical
features and heart disease is approximately linear - supported by LDA reducing
to a single component with clear class separation.

**XGBoost underperformed** relative to expectations. With max_depth=11 and
n_estimators=500, the model overfit to the small training set (726 rows).
The gap between its CV score (0.9018) and test ROC-AUC is the largest of the
three models, confirming overfitting.

**Random Forest at threshold 0.3** produces the fewest false negatives (4 out
of 99 disease cases) - the most clinically useful result if deployed for
screening where missing a diagnosis is the worst outcome.

**Five features consistently drive predictions** across all three models
according to SHAP analysis: cp (chest pain type), ca (vessels coloured),
oldpeak (ST depression), exang (exercise-induced angina), and thalach
(maximum heart rate). This cross-model consistency validates these as
genuine clinical risk predictors rather than model-specific artefacts.

**LDA with 1 component nearly matches full-feature models** - achieving
0.8984 ROC-AUC using a single linear projection of 13 features, demonstrating
the strong linear separability in this dataset.

---

## Setup and Installation

### Requirements

- Python 3.10 or higher
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/Viet Thang Tranr-username/Heart-Disease-Prediction-Using-Machine-Learning.git
cd Heart-Disease-Prediction-Using-Machine-Learning
```

Create and activate a virtual environment:

```bash
python -m venv heart-disease
# Windows
heart-disease\Scripts\activate
# macOS/Linux
source heart-disease/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Notebooks must be run sequentially. Each notebook saves its outputs
(cleaned data, splits, models, metrics) for the next notebook to load.

Run interactively in VS Code or JupyterLab by opening each notebook and
running all cells. For best results, restart the kernel between notebooks
to free memory:

```bash
# Run all notebooks from the terminal (recommended for full pipeline)
cd notebooks
jupyter nbconvert --to notebook --execute 01_Prepocessing.ipynb --output 01_Prepocessing.ipynb
jupyter nbconvert --to notebook --execute 02_EDA.ipynb --output 02_EDA.ipynb
jupyter nbconvert --to notebook --execute 03_Feature_Engineering.ipynb --output 03_Feature_Engineering.ipynb
jupyter nbconvert --to notebook --execute 04_Dimensionality_Reduction.ipynb --output 04_Dimensionality_Reduction.ipynb
jupyter nbconvert --to notebook --execute 05_Random_Forest.ipynb --output 05_Random_Forest.ipynb
jupyter nbconvert --to notebook --execute 06_LR.ipynb --output 06_LR.ipynb
jupyter nbconvert --to notebook --execute 07_XGBoost.ipynb --output 07_XGBoost.ipynb
jupyter nbconvert --to notebook --execute 08_Evaluation.ipynb --output 08_Evaluation.ipynb
jupyter nbconvert --to notebook --execute 09_SHAP.ipynb --output 09_SHAP.ipynb
```

Note: The raw `.data` files must be present in `data/raw/` before running
notebook 01. All other data files are generated automatically by the pipeline.

---

## Team

| Name | Contributions |
|---|---|
| Viet Thang Tran | EDA, Feature Engineering, Dimensionality Reduction, Random Forest, Evaluation, SHAP |
| Jeevan Manoj | Preprocessing, Logistic Regression, XGBoost, Presentation |

MSc in Data Science, ATU Donegal - Machine Learning Module, Semester 2 2025/2026