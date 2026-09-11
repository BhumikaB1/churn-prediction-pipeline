import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import shap

print("="*60)
print("XGBOOST + SHAP ANALYSIS")
print("="*60)

# Load features
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_target.csv').values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Data loaded: {X.shape}")

# ===== TRAIN XGBOOST =====
print("\n1. TRAINING XGBOOST")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=3,  # Handle class imbalance
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("   ✓ XGBoost trained")

# ===== EVALUATION =====
print("\n2. EVALUATION")

y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

test_acc = xgb_model.score(X_test, y_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy: {test_acc:.4f}")
print(f"   ROC-AUC:  {test_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

# ===== COMPARE ALL MODELS =====
print("\n3. MODEL COMPARISON")

# Load RF for comparison
rf_model = joblib.load('models/random_forest_tuned.pkl')
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)
rf_acc = rf_model.score(X_test, y_test)

lr_model = joblib.load('models/logistic_regression_baseline.pkl')
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
lr_acc = lr_model.score(X_test, y_test)

print(f"\n   {'Model':<25} {'Accuracy':<12} {'ROC-AUC':<12}")
print(f"   {'-'*50}")
print(f"   {'Logistic Regression':<25} {lr_acc:<12.4f} {lr_auc:<12.4f}")
print(f"   {'Random Forest (Tuned)':<25} {rf_acc:<12.4f} {rf_auc:<12.4f}")
print(f"   {'XGBoost':<25} {test_acc:<12.4f} {test_auc:<12.4f}")

# Select best model
best_auc = max(lr_auc, rf_auc, test_auc)
if best_auc == test_auc:
    print(f"\n   🏆 XGBoost is the BEST model! (AUC: {test_auc:.4f})")
    production_model = xgb_model
    production_model_name = "XGBoost"
elif best_auc == rf_auc:
    print(f"\n   🏆 Random Forest is still BEST (AUC: {rf_auc:.4f})")
    production_model = rf_model
    production_model_name = "Random Forest"
else:
    print(f"\n   🏆 Logistic Regression is BEST (AUC: {lr_auc:.4f})")
    production_model = lr_model
    production_model_name = "Logistic Regression"

# ===== SHAP ANALYSIS =====
print("\n4. SHAP EXPLAINABILITY")

# Use XGBoost for SHAP (native support)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print("   ✓ SHAP values calculated")

# Plot 1: SHAP Summary (most important features)
print("   Creating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    show=False,
    max_display=15
)
plt.title("Feature Importance (SHAP Values)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/08_shap_feature_importance.png")

# Plot 2: SHAP Beeswarm (impact direction)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    show=False,
    max_display=15
)
plt.title("SHAP Feature Impact on Churn", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/09_shap_beeswarm.png")

# Plot 3: ROC Comparison
plt.figure(figsize=(10, 6))
for model, proba, name in [
    (lr_model, lr_pred_proba, f'Logistic Reg (AUC={lr_auc:.3f})'),
    (rf_model, rf_pred_proba, f'Random Forest (AUC={rf_auc:.3f})'),
    (xgb_model, y_pred_proba, f'XGBoost (AUC={test_auc:.3f})')
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=name)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Models Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/10_roc_all_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/10_roc_all_models.png")

# ===== SAVE MODELS =====
print("\n5. SAVING")

# Save XGBoost
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("   ✓ XGBoost saved: models/xgboost_model.pkl")

# Save best model as production model
joblib.dump(production_model, 'models/production_model.pkl')
print(f"   ✓ Production model saved: models/production_model.pkl ({production_model_name})")

# Save SHAP explainer
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("   ✓ SHAP explainer saved: models/shap_explainer.pkl")

# ===== KEY INSIGHTS =====
print("\n6. KEY SHAP INSIGHTS")

# Top features by mean SHAP value
feature_importance_shap = pd.DataFrame({
    'feature': X_test.columns,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\n   Top 10 features driving churn predictions:")
for idx, row in feature_importance_shap.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['shap_importance']:.4f}")

print("\n" + "="*60)
print(f"✅ XGBOOST + SHAP COMPLETE")
print(f"   Best Model: {production_model_name}")
print(f"   Best AUC: {best_auc:.4f}")
print("="*60)