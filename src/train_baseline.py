import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, auc)
import matplotlib.pyplot as plt
import joblib

print("="*60)
print("LOGISTIC REGRESSION BASELINE MODEL")
print("="*60)

# Load features
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_target.csv').values.ravel()

print(f"\n✅ Loaded features: {X.shape}")
print(f"✅ Loaded target: {len(y)} samples")

# ===== STEP 1: Train-Test Split =====
print("\n1. TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[1]} features)")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Train churn rate: {y_train.mean():.2%}")
print(f"   Test churn rate: {y_test.mean():.2%}")

# ===== STEP 2: Train Model =====
print("\n2. TRAINING LOGISTIC REGRESSION")

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"   Model trained")
print(f"   Coefficients: {len(model.coef_[0])}")
print(f"   Intercept: {model.intercept_[0]:.4f}")

# ===== STEP 3: Predictions =====
print("\n3. MAKING PREDICTIONS")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

print(f"   Training predictions made")
print(f"   Test predictions made")

# ===== STEP 4: Evaluation =====
print("\n4. MODEL EVALUATION")

# Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\n   ACCURACY:")
print(f"   - Train: {train_acc:.4f}")
print(f"   - Test:  {test_acc:.4f}")

# ROC-AUC (better metric for imbalanced data)
train_auc = roc_auc_score(y_train, y_pred_proba_train)
test_auc = roc_auc_score(y_test, y_pred_proba_test)
print(f"\n   ROC-AUC:")
print(f"   - Train: {train_auc:.4f}")
print(f"   - Test:  {test_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n   CONFUSION MATRIX:")
print(f"   - True Negatives:  {cm[0,0]}")
print(f"   - False Positives: {cm[0,1]}")
print(f"   - False Negatives: {cm[1,0]}")
print(f"   - True Positives:  {cm[1,1]}")

# Classification Report
print(f"\n   CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Churn', 'Churn']))

# ===== STEP 5: Feature Importance =====
print("\n5. FEATURE IMPORTANCE (Coefficients)")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\n   Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['coefficient']:8.4f}")

# ===== STEP 6: Save Model =====
print("\n6. SAVING MODEL")

joblib.dump(model, 'models/logistic_regression_baseline.pkl')
feature_importance.to_csv('models/logistic_regression_importance.csv', index=False)

print(f"   ✓ Model saved: models/logistic_regression_baseline.pkl")
print(f"   ✓ Feature importance saved")

# ===== STEP 7: Visualization =====
print("\n7. CREATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve - Logistic Regression')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Confusion Matrix
im = axes[0, 1].imshow(cm, cmap='Blues')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Churn', 'Churn'])
axes[0, 1].set_yticklabels(['No Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=12)

# Plot 3: Top 10 Features
top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
axes[1, 0].barh(range(len(top_features)), top_features['coefficient'], color=colors)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 10 Feature Importance')
axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Plot 4: Prediction Distribution
axes[1, 1].hist(y_pred_proba_test[y_test == 0], bins=30, alpha=0.6, label='No Churn', color='green')
axes[1, 1].hist(y_pred_proba_test[y_test == 1], bins=30, alpha=0.6, label='Churn', color='red')
axes[1, 1].set_xlabel('Predicted Churn Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('figures/06_logistic_regression_evaluation.png', dpi=300, bbox_inches='tight')
print(f"   Visualization saved: figures/06_logistic_regression_evaluation.png")

print("\n" + "="*60)
print("✅ LOGISTIC REGRESSION BASELINE COMPLETE")
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Test ROC-AUC:  {test_auc:.4f}")
print("="*60)