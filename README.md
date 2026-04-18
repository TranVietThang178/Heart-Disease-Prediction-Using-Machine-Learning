# Heart Disease Prediction Using Machine Learning

Comparing multiple ML models (Logistic Regression, Random Forest, XGBoost) for early heart disease detection using the Cleveland Heart Disease Dataset.

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 82% | 0.89 |
| Random Forest | 85% | 0.91 |
| XGBoost | 87% | 0.93 |

XGBoost achieved the best performance with 87% accuracy and 0.93 AUC-ROC.

## Project Structure

data/raw - Raw Cleveland Heart Disease dataset
notebooks/ - Jupyter Notebooks for EDA, modelling, evaluation
src/ - Python source code for preprocessing and models
requirements.txt - Project dependencies

## Tech Stack

- Python 3.10
- scikit-learn, XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## Pipeline

1. Exploratory Data Analysis (EDA) - feature distributions, correlations, missing values
2. Feature Engineering - handling nulls, encoding categorical variables, scaling
3. Model Training - Logistic Regression, Random Forest, XGBoost with cross-validation
4. Model Evaluation - Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix
5. Results Comparison - identifying best-performing model

## Dataset

Cleveland Heart Disease Dataset from the UCI Machine Learning Repository.
14 features including age, sex, chest pain type, resting blood pressure, cholesterol, and more.
Target: presence (1) or absence (0) of heart disease.

## Author

Viet Thang Tran
MSc Data Science, Atlantic Technological University, Ireland
GitHub: https://github.com/TranVietThang178
LinkedIn: https://www.linkedin.com/in/vietthangtran1783
