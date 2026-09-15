import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import joblib
import json

print("="*60)
print("MODEL DRIFT DETECTION")
print("="*60)

# ===== LOAD DATA =====
# Training data (reference)
X_train_ref = pd.read_csv('data/processed/X_features.csv')
y_ref = pd.read_csv('data/processed/y_target.csv').values.ravel()

print(f"\n✅ Reference data loaded: {X_train_ref.shape}")

# ===== SIMULATE PRODUCTION DATA =====
# In real life, this would be new incoming customer data
# We simulate drift by adding noise/shifting distributions

print("\n1. SIMULATING PRODUCTION DATA WITH DRIFT")

np.random.seed(42)
X_production = X_train_ref.copy()

# Simulate drift: shift tenure (customers staying longer)
X_production['tenure'] = X_production['tenure'] * 1.2 + np.random.normal(0, 2, len(X_production))

# Simulate drift: MonthlyCharges increased (price hike)
X_production['MonthlyCharges'] = X_production['MonthlyCharges'] * 1.15 + np.random.normal(0, 3, len(X_production))

print("   ✓ Simulated production data with price hike + longer tenure")

# ===== DRIFT DETECTION: KS TEST =====
print("\n2. KOLMOGOROV-SMIRNOV DRIFT TEST")
print("   (Tests if production data distribution differs from training data)")

drift_results = []

for col in X_train_ref.columns:
    ks_stat, p_value = stats.ks_2samp(
        X_train_ref[col].values,
        X_production[col].values
    )
    
    # If p_value < 0.05, distributions are significantly different = DRIFT
    is_drift = p_value < 0.05
    
    drift_results.append({
        'feature': col,
        'ks_statistic': round(ks_stat, 4),
        'p_value': round(p_value, 4),
        'drift_detected': is_drift
    })

drift_df = pd.DataFrame(drift_results).sort_values('ks_statistic', ascending=False)

print(f"\n   Features with DRIFT detected:")
drifted = drift_df[drift_df['drift_detected'] == True]
print(f"   {len(drifted)}/{len(drift_df)} features drifted")

for _, row in drifted.head(10).iterrows():
    print(f"   ⚠️  {row['feature']:30s} KS={row['ks_statistic']:.4f} p={row['p_value']:.4f}")

print(f"\n   Features WITHOUT drift:")
not_drifted = drift_df[drift_df['drift_detected'] == False]
for _, row in not_drifted.head(5).iterrows():
    print(f"   ✅ {row['feature']:30s} KS={row['ks_statistic']:.4f} p={row['p_value']:.4f}")

# ===== MODEL PERFORMANCE DRIFT =====
print("\n3. MODEL PERFORMANCE DRIFT")

model = joblib.load('models/production_model.pkl')
scaler = joblib.load('models/scaler.pkl')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Original performance
X_train, X_test, y_train, y_test = train_test_split(
    X_train_ref, y_ref, test_size=0.2, random_state=42, stratify=y_ref
)

original_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Production performance (simulated)
# In real life: use actual labels from production
production_auc = roc_auc_score(
    y_test, 
    model.predict_proba(X_production.iloc[:len(X_test)])[:, 1]
)

performance_drop = original_auc - production_auc
print(f"\n   Original AUC:    {original_auc:.4f}")
print(f"   Production AUC:  {production_auc:.4f}")
print(f"   Performance Drop: {performance_drop:.4f}")

if performance_drop > 0.05:
    print(f"\n   🚨 SIGNIFICANT DRIFT DETECTED - Model needs retraining!")
elif performance_drop > 0.02:
    print(f"\n   ⚠️  MINOR DRIFT - Monitor closely")
else:
    print(f"\n   ✅ NO SIGNIFICANT DRIFT - Model is stable")

# ===== VISUALIZATION =====
print("\n4. CREATING DRIFT VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Drift Detection Report', fontsize=16, fontweight='bold')

# Plot 1: Top drifted features
top_drift = drift_df.head(10)
colors = ['red' if d else 'green' for d in top_drift['drift_detected']]
axes[0, 0].barh(range(len(top_drift)), top_drift['ks_statistic'], color=colors)
axes[0, 0].set_yticks(range(len(top_drift)))
axes[0, 0].set_yticklabels(top_drift['feature'])
axes[0, 0].set_xlabel('KS Statistic (higher = more drift)')
axes[0, 0].set_title('Top Features by Drift Score')
axes[0, 0].axvline(x=0.1, color='orange', linestyle='--', label='Warning threshold')
axes[0, 0].legend()

# Plot 2: Tenure distribution comparison
axes[0, 1].hist(X_train_ref['tenure'], bins=30, alpha=0.6, label='Training', color='blue')
axes[0, 1].hist(X_production['tenure'], bins=30, alpha=0.6, label='Production', color='red')
axes[0, 1].set_xlabel('Tenure (months)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Tenure Distribution: Training vs Production')
axes[0, 1].legend()

# Plot 3: Monthly Charges distribution comparison
axes[1, 0].hist(X_train_ref['MonthlyCharges'], bins=30, alpha=0.6, label='Training', color='blue')
axes[1, 0].hist(X_production['MonthlyCharges'], bins=30, alpha=0.6, label='Production', color='red')
axes[1, 0].set_xlabel('Monthly Charges ($)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Monthly Charges: Training vs Production')
axes[1, 0].legend()

# Plot 4: Performance comparison
metrics = ['Original AUC', 'Production AUC']
values = [original_auc, production_auc]
bar_colors = ['green', 'red' if performance_drop > 0.05 else 'orange' if performance_drop > 0.02 else 'green']
axes[1, 1].bar(metrics, values, color=bar_colors)
axes[1, 1].set_ylim([0.75, 0.95])
axes[1, 1].set_ylabel('ROC-AUC Score')
axes[1, 1].set_title('Model Performance: Original vs Production')
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/12_drift_detection.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/12_drift_detection.png")

# ===== SAVE DRIFT REPORT =====
print("\n5. SAVING DRIFT REPORT")

drift_report = {
    "total_features": len(drift_df),
    "drifted_features": len(drifted),
    "drift_percentage": round(len(drifted)/len(drift_df)*100, 2),
    "original_auc": round(original_auc, 4),
    "production_auc": round(production_auc, 4),
    "performance_drop": round(performance_drop, 4),
    "action_required": performance_drop > 0.05,
    "top_drifted_features": drifted.head(5)['feature'].tolist()
}

with open('models/drift_report.json', 'w') as f:
    json.dump(drift_report, f, indent=2)

print("   ✓ Drift report saved: models/drift_report.json")
print(f"\n   SUMMARY:")
print(f"   - {drift_report['drifted_features']}/{drift_report['total_features']} features drifted ({drift_report['drift_percentage']}%)")
print(f"   - Performance drop: {drift_report['performance_drop']}")
print(f"   - Action required: {drift_report['action_required']}")

print("\n" + "="*60)
print("✅ DRIFT DETECTION COMPLETE")
print("="*60)