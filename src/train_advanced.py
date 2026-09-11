import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import joblib

print("="*60)
print("RANDOM FOREST ADVANCED MODEL")
print("="*60)

# Load features
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_target.csv').values.ravel()

print(f"\n Loaded features: {X.shape}")
print(f" Loaded target: {len(y)} samples")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Training set: {X_train.shape[0]} samples")
print(f" Test set: {X_test.shape[0]} samples")

# ===== TRAIN RANDOM FOREST =====
print("\n1. TRAINING RANDOM FOREST")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

rf_model.fit(X_train, y_train)
print(f"    Random Forest trained ({rf_model.n_estimators} trees)")

# ===== PREDICTIONS =====
print("\n2. MAKING PREDICTIONS")

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)
y_pred_proba_train_rf = rf_model.predict_proba(X_train)[:, 1]
y_pred_proba_test_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"  Predictions made")

# ===== EVALUATION =====
print("\n3. MODEL EVALUATION")

train_acc_rf = rf_model.score(X_train, y_train)
test_acc_rf = rf_model.score(X_test, y_test)
train_auc_rf = roc_auc_score(y_train, y_pred_proba_train_rf)
test_auc_rf = roc_auc_score(y_test, y_pred_proba_test_rf)

print(f"\n   ACCURACY:")
print(f"   - Train: {train_acc_rf:.4f}")
print(f"   - Test:  {test_acc_rf:.4f}")

print(f"\n   ROC-AUC:")
print(f"   - Train: {train_auc_rf:.4f}")
print(f"   - Test:  {test_auc_rf:.4f}")

print(f"\n   CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test_rf, 
                          target_names=['No Churn', 'Churn']))

# ===== FEATURE IMPORTANCE =====
print("\n4. FEATURE IMPORTANCE (Tree-based)")

feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for idx, row in feature_importance_rf.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:8.4f}")

# ===== SAVE MODEL =====
print("\n5. SAVING MODEL")

joblib.dump(rf_model, 'models/random_forest_v1.pkl')
feature_importance_rf.to_csv('models/random_forest_importance.csv', index=False)

print(f"    Model saved: models/random_forest_v1.pkl")
print(f"    Feature importance saved")

# ===== COMPARISON =====
print("\n6. MODEL COMPARISON")
print("\n   Logistic Regression vs Random Forest:")
print(f"   {'Metric':<20} {'Logistic Reg':<15} {'Random Forest':<15}")
print(f"   {'-'*50}")
print(f"   {'Test Accuracy':<20} {0.7980:<15.4f} {test_acc_rf:<15.4f}")
print(f"   {'Test ROC-AUC':<20} {0.8406:<15.4f} {test_auc_rf:<15.4f}")

if test_auc_rf > 0.8406:
    print(f"\n    Random Forest is BETTER (AUC +{(test_auc_rf - 0.8406)*100:.2f}%)")
else:
    print(f"\n    Logistic Regression is simpler and performs well")

# ===== VISUALIZATION =====
print("\n7. CREATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: ROC Curve Comparison
from sklearn.linear_model import LogisticRegression
lr_model = joblib.load('models/logistic_regression_baseline.pkl')
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_test_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_test_rf)

axes[0, 0].plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC = {auc_lr:.3f})', linewidth=2)
axes[0, 0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_test_rf)
im = axes[0, 1].imshow(cm_rf, cmap='Blues')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_title('Confusion Matrix - Random Forest')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Churn', 'Churn'])
axes[0, 1].set_yticklabels(['No Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, str(cm_rf[i, j]), ha='center', va='center', color='white', fontsize=12)

# Plot 3: Top 10 Features
top_features_rf = feature_importance_rf.head(10)
axes[1, 0].barh(range(len(top_features_rf)), top_features_rf['importance'], color='steelblue')
axes[1, 0].set_yticks(range(len(top_features_rf)))
axes[1, 0].set_yticklabels(top_features_rf['feature'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance - Random Forest')

# Plot 4: Accuracy Comparison
models = ['Logistic Reg', 'Random Forest']
accuracies = [0.7980, test_acc_rf]
aucs = [0.8406, test_auc_rf]

x = np.arange(len(models))
width = 0.35

axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
axes[1, 1].bar(x + width/2, aucs, width, label='ROC-AUC', color='lightcoral')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].set_ylim([0.75, 0.90])

plt.tight_layout()
plt.savefig('figures/07_random_forest_evaluation.png', dpi=300, bbox_inches='tight')
print(f"    Visualization saved: figures/07_random_forest_evaluation.png")

print("\n" + "="*60)
print(" RANDOM FOREST MODEL COMPLETE")
print(f"   Test Accuracy: {test_acc_rf:.4f}")
print(f"   Test ROC-AUC:  {test_auc_rf:.4f}")
print("="*60)