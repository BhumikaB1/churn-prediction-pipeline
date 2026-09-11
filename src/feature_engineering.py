import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("FEATURE ENGINEERING")

# Load cleaned data
df = pd.read_csv('data/processed/telco_clean.csv')
print(f"\n✅ Loaded {len(df)} rows")

# ===== STEP 1: Handle Categorical Variables =====
print("\n1. ENCODING CATEGORICAL VARIABLES")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Remove Churn (it's the target)
categorical_cols = [col for col in categorical_cols if col != 'Churn']

print(f"   Found {len(categorical_cols)} categorical columns")

# Label encode all categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"   ✓ {col}: {len(le.classes_)} unique values")

# Drop original categorical columns (keep encoded versions)
df_encoded = df.drop(columns=categorical_cols)

print(f"   Shape after encoding: {df_encoded.shape}")

# ===== STEP 2: Separate Features and Target =====
print("\n2. SEPARATING FEATURES AND TARGET")

# Convert target to numeric
y = (df['Churn'] == 'Yes').astype(int)
print(f"   Target distribution:")
print(f"   - No churn (0): {(y == 0).sum()}")
print(f"   - Churn (1): {(y == 1).sum()}")
print(f"   - Churn rate: {y.mean():.2%}")

# Features (everything except Churn and original categorical)
X = df_encoded.drop('Churn', axis=1)
print(f"   Features shape: {X.shape}")
print(f"   Feature columns: {X.columns.tolist()}")

# ===== STEP 3: Feature Scaling =====
print("\n3. SCALING NUMERIC FEATURES")

# Scale all features (important for distance-based algorithms)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"   ✓ Scaled {X_scaled.shape[1]} features")
print(f"   Mean: {X_scaled.mean().mean():.4f}")
print(f"   Std: {X_scaled.std().mean():.4f}")

# ===== STEP 4: Save for Modeling =====
print("\n4. SAVING FEATURES")

X_scaled.to_csv('data/processed/X_features.csv', index=False)
y.to_csv('data/processed/y_target.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_dict, 'models/label_encoders.pkl')

print(f"   ✓ X_features.csv ({X_scaled.shape})")
print(f"   ✓ y_target.csv ({len(y)})")
print(f"   ✓ scaler.pkl")
print(f"   ✓ label_encoders.pkl")

print("\n" + "="*60)
print("✅ FEATURE ENGINEERING COMPLETE")
print("="*60)