"""
Heart Disease Prediction Using Machine Learning
================================================
This script trains and compares two machine learning models to predict
the presence of heart disease:
  1. Logistic Regression
  2. Random Forest Classifier

Dataset: Cleveland Heart Disease dataset (UCI Repository)
Target: 1 = heart disease present, 0 = no heart disease
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ── 1. Load dataset ────────────────────────────────────────────────────────────

df = pd.read_csv("heart.csv")

print("=" * 60)
print("HEART DISEASE PREDICTION USING MACHINE LEARNING")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df["target"].value_counts())

# ── 2. Prepare features and target ────────────────────────────────────────────

X = df.drop("target", axis=1)
y = df["target"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Model 1 – Logistic Regression ──────────────────────────────────────────

print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"\nAccuracy : {lr_accuracy * 100:.2f}%")
print(f"ROC-AUC  : {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ── 4. Model 2 – Random Forest Classifier ─────────────────────────────────────

print("=" * 60)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)          # Random Forest doesn't need scaling

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"\nAccuracy : {rf_accuracy * 100:.2f}%")
print(f"ROC-AUC  : {rf_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ── 5. Accuracy comparison ────────────────────────────────────────────────────

print("=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10}")
print("-" * 52)
print(f"{'Logistic Regression':<30} {lr_accuracy * 100:>9.2f}% {lr_auc:>10.4f}")
print(f"{'Random Forest Classifier':<30} {rf_accuracy * 100:>9.2f}% {rf_auc:>10.4f}")
print("-" * 52)

if lr_accuracy > rf_accuracy:
    winner = "Logistic Regression"
elif rf_accuracy > lr_accuracy:
    winner = "Random Forest Classifier"
else:
    winner = "Both models (tie)"
print(f"\nBest model by accuracy: {winner}")

# ── 6. Visualizations ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Heart Disease Prediction – Model Comparison", fontsize=16, fontweight="bold")

# 6a. Accuracy bar chart
models = ["Logistic Regression", "Random Forest"]
accuracies = [lr_accuracy * 100, rf_accuracy * 100]
colors = ["steelblue", "darkorange"]
bars = axes[0, 0].bar(models, accuracies, color=colors, edgecolor="black", width=0.4)
axes[0, 0].set_ylim(0, 110)
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# 6b. ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
axes[0, 1].plot(fpr_lr, tpr_lr, color="steelblue", lw=2,
                label=f"Logistic Regression (AUC = {lr_auc:.4f})")
axes[0, 1].plot(fpr_rf, tpr_rf, color="darkorange", lw=2,
                label=f"Random Forest (AUC = {rf_auc:.4f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curve Comparison")
axes[0, 1].legend(loc="lower right")

# 6c. Confusion matrix – Logistic Regression
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
axes[1, 0].set_title("Confusion Matrix – Logistic Regression")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# 6d. Confusion matrix – Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges", ax=axes[1, 1],
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
axes[1, 1].set_title("Confusion Matrix – Random Forest")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("\nComparison chart saved to model_comparison.png")

# ── 7. Feature importance (Random Forest) ────────────────────────────────────

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
feature_importance.plot(kind="bar", color="darkorange", edgecolor="black", ax=ax2)
ax2.set_title("Feature Importance – Random Forest", fontsize=14, fontweight="bold")
ax2.set_xlabel("Feature")
ax2.set_ylabel("Importance Score")
ax2.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("Feature importance chart saved to feature_importance.png")

print("\nDone.")
