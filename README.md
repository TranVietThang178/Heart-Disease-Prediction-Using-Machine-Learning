# Heart Disease Prediction Using Machine Learning

A machine learning project that trains and compares **two classification models** to predict the presence of heart disease, using the Cleveland Heart Disease dataset.

---

## Overview

This project demonstrates:

- Data loading, exploration, and preprocessing
- Training two ML models:
  1. **Logistic Regression**
  2. **Random Forest Classifier**
- Evaluating and comparing both models by **accuracy** and **ROC-AUC score**
- Visualizing results (accuracy bar chart, ROC curves, confusion matrices, feature importance)

---

## Dataset

The dataset (`heart.csv`) is based on the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `age`      | Age in years                                     |
| `sex`      | Sex (1 = male; 0 = female)                       |
| `cp`       | Chest pain type (0–3)                            |
| `trestbps` | Resting blood pressure (mm Hg)                   |
| `chol`     | Serum cholesterol (mg/dl)                        |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true)       |
| `restecg`  | Resting ECG results (0–2)                        |
| `thalach`  | Maximum heart rate achieved                      |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no)        |
| `oldpeak`  | ST depression induced by exercise                |
| `slope`    | Slope of peak exercise ST segment (0–2)          |
| `ca`       | Number of major vessels colored by fluoroscopy  |
| `thal`     | Thalassemia type (0 = normal; 1 = fixed; 2 = reversible) |
| `target`   | **1 = heart disease present, 0 = no disease**    |

---

## Models

### 1. Logistic Regression
A linear model that estimates the probability of heart disease from the input features. Features are standardised with `StandardScaler` before training.

### 2. Random Forest Classifier
An ensemble of 100 decision trees. It does not require feature scaling and naturally ranks the importance of each input feature.

---

## Results

| Model                    | Accuracy | ROC-AUC |
|--------------------------|----------|---------|
| Logistic Regression      | 84.91%   | 0.8986  |
| Random Forest Classifier | 75.47%   | 0.7838  |

> Results may vary slightly depending on the random seed and train/test split.

---

## Repository Structure

```
Heart-Disease-Prediction-Using-Machine-Learning/
├── heart.csv                      # Dataset
├── heart_disease_prediction.py    # Main script (training + comparison)
├── requirements.txt               # Python dependencies
├── model_comparison.png           # Accuracy / ROC / confusion matrix chart (generated)
└── feature_importance.png         # Feature importance chart (generated)
```

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the prediction script

```bash
python heart_disease_prediction.py
```

The script will:
- Print dataset statistics and model evaluation metrics to the console
- Save two charts: `model_comparison.png` and `feature_importance.png`

---

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
