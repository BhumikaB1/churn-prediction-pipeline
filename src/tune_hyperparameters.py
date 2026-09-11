import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

# Load features
X = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_target.csv').values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data loaded and split")

# ===== GRID SEARCH =====
print("\n1. GRID SEARCH - Testing different hyperparameters")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'class_weight': ['balanced', None]
}

print(f"   Testing {50*3*3*2*2} combinations...")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Grid search complete")

# ===== BEST PARAMETERS =====
print("\n2. BEST HYPERPARAMETERS")
print(f"\n   {grid_search.best_params_}")
print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ===== FINAL MODEL =====
print("\n3. TRAINING FINAL MODEL WITH BEST PARAMETERS")

best_model = grid_search.best_estimator_
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = best_model.predict(X_test)

test_auc = roc_auc_score(y_test, y_pred_proba_test)
test_acc = best_model.score(X_test, y_test)

print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Test ROC-AUC: {test_auc:.4f}")

print(f"\n   CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Churn', 'Churn']))

# ===== SAVE =====
print("\n4. SAVING TUNED MODEL")

joblib.dump(best_model, 'models/random_forest_tuned.pkl')
print(f"   ✓ Model saved: models/random_forest_tuned.pkl")

print("\n" + "="*60)
print("✅ HYPERPARAMETER TUNING COMPLETE")
print("="*60)