# Churn Prediction Pipeline — Complete Technical Guide

This guide explains the whole project from raw data to deployed API. It has three jobs:

1. **Walk through the pipeline** in order, saying what each step does and *why we did it*.
2. **Define every technical term** and say how it connects to this project (see the glossary in Part 4).
3. **Record what was reviewed and corrected**, and what is still open (Parts 5 and 6).

All numbers below come from the final retrained run on the corrected features, unless marked otherwise.

---

## Contents

1. [The problem and the data](#1-the-problem-and-the-data)
2. [The pipeline, step by step (with reasons)](#2-the-pipeline-step-by-step-with-reasons)
3. [Final results](#3-final-results)
4. [Glossary: every term, its meaning, and where it appears here](#4-glossary)
5. [Bugs found and corrected (and why they mattered)](#5-bugs-found-and-corrected)
6. [Known open issues](#6-known-open-issues)
7. [File map and how to re-run](#7-file-map-and-how-to-re-run)

---

## 1. The problem and the data

**Business problem.** A telecom company loses revenue when customers leave ("churn"). Winning a new customer costs far more than keeping one. If we can predict *who is likely to leave*, the retention team can contact those customers first (a discount, a contract upgrade, a call).

**ML framing.** This is **binary classification**: for each customer, predict `1` (churns) or `0` (stays). We use the *probability* of churn, not just the label, because a probability lets the business choose whom to contact.

**Dataset.** The Telco Customer Churn data (`data/raw_data/telco_customers.csv`): 7,043 customers, 21 columns.

- **Target:** `Churn` (Yes/No). About **26.5 %** of customers churned.
- **Numeric features:** `tenure` (months as a customer), `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` (0/1).
- **Categorical features:** gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.
- **Identifier:** `customerID` (unique per row; not a real feature).

**Why the class imbalance matters.** Only about 1 in 4 customers churns. A model that always says "stays" is already ~73.5 % accurate and completely useless. This is why we do not trust accuracy alone and use ROC-AUC, class weights, and threshold analysis.

---

## 2. The pipeline, step by step (with reasons)

```
Raw CSV -> Load to SQL (optional) -> EDA -> Cleaning -> Feature engineering
        -> Train/test split -> Models (LR, RF, tuned RF, XGBoost)
        -> Model selection -> SHAP explainability -> Save artifacts
        -> FastAPI -> Render cloud -> Drift monitoring
```

### Step 1 — Load data to a database (`scripts/load_data_to_supabase.py`)
Pushes the CSV into a Postgres database on Supabase using SQLAlchemy.
**Why:** it simulates a real setting where data lives in a database and is queried with SQL, not read from a CSV. It is not needed to train the models. (See open issue 6.1: this script contains a hardcoded password.)

### Step 2 — Exploratory Data Analysis (`notebooks/01_exploratory_analysis.ipynb`, figures 01–05, 11)
Looks at churn rate, distribution of tenure and charges, churn by contract type, and correlations.
**Why:** you must understand the data before modelling. EDA is where we saw the findings the business cares about:

- New customers (low tenure) churn far more than long-standing ones.
- Month-to-month contracts churn far more than 1- or 2-year contracts.

These were later confirmed independently by the models (Contract and tenure are the top drivers), which is a good sign the models learned real signal.

### Step 3 — Cleaning (`data/processed/telco_clean.csv`)
`TotalCharges` arrives as *text* because 11 rows (customers with `tenure = 0`, brand new) have a blank value. The cleaned file keeps it as text-like; the conversion is finished in feature engineering.
**Why it matters:** see bug 5.1. This one detail was the biggest data-handling error in the project.

### Step 4 — Feature engineering (`src/feature_engineering.py`)
What it does, in order, and why:

| Action | Why |
|---|---|
| Convert `TotalCharges` to a number; blank → 0 | It is a dollar amount, so the model must see it as ordered numbers. Blank means "no charges yet" (tenure 0), so 0 is the honest value. |
| Drop `customerID` | A unique ID has no predictive meaning. Keeping it only lets a flexible model memorise rows and adds noise. |
| Label-encode the categorical columns | Models need numbers. Each category becomes an integer (`Contract`: Month-to-month=0, One year=1, Two year=2). Saved to `models/label_encoders.pkl` so the API applies the *same* mapping. |
| Split first, then fit the `StandardScaler` on training rows only | Scaling must be learned from training data only, otherwise information from the test set leaks in. The split uses the same settings as the training scripts (`test_size=0.2`, `random_state=42`, stratified), so the rows match. |
| Scale all features to mean 0, std 1 | Logistic Regression is sensitive to feature scale (its optimiser and its coefficients are only comparable when features share a scale). Saved to `models/scaler.pkl` so the API scales identically. |
| Save `X_features.csv`, `y_target.csv` | The single input every training script reads. |

**Why label encoding instead of one-hot?** It keeps the feature count small and works fine for tree models (they split on thresholds and can isolate categories). The trade-off: for Logistic Regression it imposes a fake order (e.g. Fiber optic "greater than" DSL). One-hot encoding would likely help LR slightly; it was left unchanged because it changes the API contract. It is a reasonable future improvement.

### Step 5 — Train / test split
`train_test_split(test_size=0.2, random_state=42, stratify=y)`.

- **80 % train / 20 % test:** 5,634 train and 1,409 test rows.
- **Test set:** data the model never trains on, so its score estimates how the model behaves on new customers.
- **`random_state=42`:** makes the split reproducible. Every script must produce the same split or the comparison between models is meaningless.
- **`stratify=y`:** keeps the ~26.5 % churn rate in both halves. Without it, a random split could put noticeably different churn rates in train and test.

### Step 6 — Baseline: Logistic Regression (`src/train_baseline.py`)
A linear model: it computes a weighted sum of the features and squashes it to a probability with the sigmoid function.
**Why start here:** a baseline tells you how much value complexity adds. If a fancy model cannot beat a simple, explainable one, the fancy model is not worth its cost. Coefficients are directly interpretable: negative for `tenure` (longer tenure → less churn), positive for `MonthlyCharges`.

The script also applies a **custom threshold of 0.35** (instead of the default 0.5) for its accuracy, confusion matrix and report. **Why:** with 26.5 % churners, the default 0.5 cut-off misses many real churners. Lowering the threshold catches more churners (higher **recall**) at the cost of more false alarms (lower **precision**). For retention, missing a churner (lose a customer) usually costs more than a wasted discount, so a lower threshold is defensible. ROC-AUC is threshold-independent and is unaffected.

### Step 7 — Random Forest (`src/train_advanced.py`)
An **ensemble** of 100 decision trees, each trained on a random sample of rows and features; predictions are averaged.
Settings and reasons:

- `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=5`: limit tree size to reduce **overfitting** (memorising training data).
- `class_weight='balanced'`: makes each churner count more, counteracting the 73/27 imbalance.
- `n_jobs=-1`: use all CPU cores.

**Why RF:** it captures non-linear effects and feature interactions that Logistic Regression cannot (for example "high charges *and* month-to-month *and* short tenure").

### Step 8 — Hyperparameter tuning (`src/tune_hyperparameters.py`)
**GridSearchCV** tries every combination in a grid (3 × 3 × 2 × 2 = **36** combinations) with **5-fold cross-validation** (180 model fits), scored by ROC-AUC.
Best found: `n_estimators=150, max_depth=10, min_samples_split=10, class_weight='balanced'`, CV AUC 0.8445.
**Why:** default settings are rarely optimal. Cross-validation evaluates each setting on several train/validation splits, so we pick settings using only the *training* data and keep the test set untouched.
(The search runs serially: nesting two levels of parallelism exhausted memory on this machine.)

### Step 9 — XGBoost + SHAP (`src/train_xgboost.py`)
**XGBoost** builds trees *sequentially*, each one correcting the errors of the previous (**gradient boosting**). It is often the strongest model on tabular data. `scale_pos_weight=3` approximates the 73:27 imbalance.

**Model selection.** The three models are compared, and the one with the best **cross-validated AUC on the training set** becomes `models/production_model.pkl`.
**Why cross-validation and not the test set:** if you pick the winner by test score and then report that same test score, the number is optimistically biased. The test set should be used once, to report.

**SHAP** explains predictions (see glossary). It produces figures 08 (global importance) and 09 (beeswarm: direction of each feature's effect) and saves `models/shap_explainer.pkl`.
**Why:** a churn score is only useful if the retention team can see *why* a customer is at risk. It also validates that the model uses sensible features.

### Step 10 — Serving (`api/main.py`, FastAPI)
- `/predict`: one customer → churn probability, label, confidence, risk level (LOW < 0.4 ≤ MEDIUM ≤ 0.7 < HIGH).
- `/predict-batch`: a list of customers.
- `/health`: is the service alive and which model is loaded.
- `/debug/features`: shows what features the model expects.

Both prediction endpoints call one function, `build_features`, which repeats exactly what feature engineering did: label-encode with the saved encoders → put columns in the scaler's training order → scale.
**Why the same code path matters:** the model was trained on scaled, encoded numbers. If serving builds features differently, predictions are silently wrong. This is called **training-serving skew**, and it was a real bug here (see 5.1, 5.2, 5.7).

### Step 11 — Deployment (`render.yaml`)
Render builds the app with `pip install -r requirements.txt` and starts it with uvicorn. The API is available as a public REST service with automatic Swagger docs at `/docs`.
**Why:** a model nobody can call has no business value. An API lets any system (CRM, website, dashboard) request a score.

### Step 12 — Drift monitoring (`src/drift_detection.py`)
Compares the distribution of each feature in "production" data against the training data with the **Kolmogorov–Smirnov (KS) test**, and compares model AUC.
**Why:** models decay when the world changes (price hikes, new customer mix). Monitoring tells you when to retrain. In this project the production data is *simulated* (tenure and charges are artificially shifted), so it demonstrates the method; it is not real monitoring. See open issue 6.3.

---

## 3. Final results

Test set = 1,409 customers, never used for training or model selection.

| Model | Test ROC-AUC | Test accuracy (0.5) | CV ROC-AUC (train, 5-fold) |
|---|---|---|---|
| **Logistic Regression (selected)** | **0.8404** | 0.7984 | 0.8446 |
| Random Forest v1 (untuned) | 0.8387 | 0.7729 | not computed |
| Random Forest (tuned) | 0.8391 | 0.7693 | 0.8445 |
| XGBoost | 0.8342 | 0.7509 | 0.8356 |

**How to read this**

- The three top models are within ~0.002 AUC of each other. That is *within noise*; none is clearly better. XGBoost is slightly worse.
- When several models tie, choose the **simplest and most explainable** one. That is Logistic Regression, which is why it is now the production model.
- Random Forest and XGBoost have *lower accuracy* than Logistic Regression because `class_weight='balanced'` / `scale_pos_weight` trade accuracy for higher recall on churners. Accuracy is not comparable to LR's here in a business sense; AUC is the fairer comparison.
- **Top drivers** (consistent across models): `Contract`, `tenure`, `TotalCharges`, `MonthlyCharges`. Logistic Regression coefficients: tenure −1.23 (longer stay → less churn), MonthlyCharges +0.73, Contract −0.60.
- An AUC of about 0.84 is typical and healthy for this public dataset. Values much higher (for example 0.87+) would raise a suspicion of data leakage rather than good news.

---

## 4. Glossary

Grouped by topic. Each entry has **Meaning** and **In this project**.

### 4.1 Problem and data terms

**Churn** — Meaning: a customer ending their service. In this project: the target we predict (`Churn` Yes/No, 26.5 % Yes).

**Binary classification** — Meaning: predicting one of two classes. In this project: churn vs no churn.

**Feature** — Meaning: an input column used to predict. In this project: 19 features such as tenure, Contract, MonthlyCharges.

**Target / label (y)** — Meaning: the value the model predicts. In this project: `y_target.csv` (1 = churned).

**Class imbalance** — Meaning: one class is much rarer. In this project: 26.5 % churn vs 73.5 % stay; it makes accuracy misleading and motivates class weights and AUC.

**Tenure** — Meaning: months the customer has been with the company. In this project: one of the strongest predictors.

**Categorical vs numeric feature** — Meaning: categories (Contract type) vs measurable numbers (MonthlyCharges). In this project: categoricals must be encoded before modelling.

**Identifier column** — Meaning: a unique key with no predictive meaning. In this project: `customerID`, dropped.

### 4.2 Data preparation terms

**EDA (Exploratory Data Analysis)** — Meaning: charts and statistics to understand data before modelling. In this project: notebook 01 and figures 01–05, 11.

**Data cleaning** — Meaning: fixing types, blanks and errors. In this project: `TotalCharges` blanks (11 rows).

**Label encoding** — Meaning: replace each category with an integer. In this project: applied to 15 categorical columns; mappings saved in `label_encoders.pkl`.

**One-hot encoding** — Meaning: one 0/1 column per category, with no false ordering. In this project: not used; a possible improvement for Logistic Regression.

**Feature scaling / standardisation (`StandardScaler`)** — Meaning: subtract the mean and divide by the standard deviation so each feature has mean 0, std 1. In this project: required for Logistic Regression; saved as `scaler.pkl`.

**Data leakage** — Meaning: information from outside the training data (test set, future, or the target) sneaking into training, which makes scores unrealistically good. In this project: the scaler used to be fit on all rows before the split; fixed.

**Train / test split** — Meaning: partition data into a part to learn from and a part to evaluate on. In this project: 80/20.

**Stratified split** — Meaning: keeps class proportions identical in both parts. In this project: `stratify=y`.

**Random seed (`random_state`)** — Meaning: fixes randomness for reproducible results. In this project: 42 everywhere.

**Artifact (`.pkl` file)** — Meaning: a saved Python object serialised with `joblib`/pickle. In this project: models, scaler, encoders and SHAP explainer stored in `models/`.

### 4.3 Model terms

**Baseline model** — Meaning: a simple reference model to beat. In this project: Logistic Regression.

**Logistic Regression** — Meaning: linear model that outputs a probability through the sigmoid function. In this project: baseline and (after selection) production model; its coefficients give direction and size of each feature's effect.

**Coefficient** — Meaning: the weight a linear model assigns to a feature. In this project: saved in `logistic_regression_importance.csv`; negative → lowers churn probability.

**Decision tree** — Meaning: a flowchart of yes/no splits on features. In this project: building block of RF and XGBoost.

**Random Forest (RF)** — Meaning: many trees on random subsets, votes averaged (**bagging**). In this project: v1 and tuned versions.

**Bagging** — Meaning: train models on random resamples and average them, reducing variance. In this project: how RF works.

**Gradient boosting / XGBoost** — Meaning: trees built one after another, each fixing the previous errors. In this project: the third model.

**Ensemble** — Meaning: combining many models for a stronger prediction. In this project: RF and XGBoost.

**Overfitting** — Meaning: memorising training data so it performs worse on new data (train score ≫ test score). In this project: limited with `max_depth`, `min_samples_leaf`, and by evaluating on a held-out test set.

**Underfitting** — Meaning: model too simple to capture the pattern. In this project: the risk of a linear model on non-linear effects; not a big issue as LR matches the trees.

**Bias–variance trade-off** — Meaning: simple models err by being too rigid (bias), complex ones by being too sensitive to the data (variance). In this project: why a simple LR ties with complex models here.

**Class weight (`class_weight='balanced'`)** — Meaning: increases the penalty for mistakes on the rarer class. In this project: RF grid search.

**`scale_pos_weight`** — Meaning: XGBoost's version of class weighting (roughly negatives ÷ positives ≈ 2.8). In this project: set to 3.

**Hyperparameter** — Meaning: a setting chosen *before* training (tree depth, number of trees). Contrast with parameters, which are learned (coefficients). In this project: tuned for RF.

**`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`** — Meaning: number of trees; how deep each tree can go; minimum samples needed to split a node or to form a leaf. In this project: RF complexity controls.

**`learning_rate`** — Meaning: how much each boosting step contributes; smaller is more cautious. In this project: 0.1 for XGBoost.

**`max_iter`** — Meaning: maximum optimiser steps for Logistic Regression to converge. In this project: 1000.

**Grid search** — Meaning: try every combination of a specified set of hyperparameter values. In this project: 36 combinations.

**Cross-validation (CV, 5-fold)** — Meaning: split training data into 5 parts; train on 4, validate on 1, rotate, average. In this project: tuning and model selection without touching the test set.

### 4.4 Evaluation terms

**Accuracy** — Meaning: share of predictions that are correct. In this project: reported, but misleading under imbalance.

**Confusion matrix** — Meaning: table of true/false positives/negatives. In this project: shown in figures 06 and 07.

**True positive (TP) / false positive (FP) / true negative (TN) / false negative (FN)** — Meaning: churner caught / non-churner wrongly flagged / non-churner correctly cleared / churner missed. In this project: FN = a lost customer we failed to target; FP = a wasted retention offer.

**Precision** — Meaning: of customers flagged as churners, the share who really churn (TP ÷ (TP + FP)). In this project: cost of false alarms.

**Recall (sensitivity)** — Meaning: of real churners, the share we caught (TP ÷ (TP + FN)). In this project: the business priority.

**F1-score** — Meaning: harmonic mean of precision and recall. In this project: appears in the classification report.

**Decision threshold** — Meaning: the probability above which we predict "churn". In this project: 0.5 in the API and most scripts; 0.35 in the baseline for higher recall. Lowering it raises recall and lowers precision.

**ROC curve** — Meaning: plot of true-positive rate against false-positive rate across all thresholds. In this project: figures 06, 07, 10.

**ROC-AUC (Area Under the Curve)** — Meaning: probability that the model ranks a random churner above a random non-churner. 0.5 = coin flip, 1.0 = perfect. In this project: the primary comparison metric, because it does not depend on a threshold and is robust to imbalance.

**Model selection bias** — Meaning: an optimistic score caused by choosing among many models using the same data you report on. In this project: fixed by selecting with CV on the training set.

### 4.5 Explainability terms

**Feature importance** — Meaning: how much a feature contributes to predictions. In this project: RF's built-in importance (Contract 0.18, tenure 0.16, TotalCharges 0.13), and LR coefficients.

**SHAP (SHapley Additive exPlanations)** — Meaning: a game-theory method that assigns each feature a signed contribution to a specific prediction. In this project: `TreeExplainer` on XGBoost, figures 08 and 09.

**Beeswarm plot** — Meaning: SHAP plot where each dot is one customer; position shows push toward/away from churn and colour shows the feature value. In this project: figure 09.

**Explainability / interpretability** — Meaning: being able to justify why a model made a prediction. In this project: crucial for retention teams and stakeholders' trust.

### 4.6 Engineering and deployment terms

**Pipeline** — Meaning: the full ordered process from raw data to served prediction. In this project: the whole repo.

**REST API** — Meaning: a web service reached through HTTP requests. In this project: `/predict`, `/predict-batch`, `/health`.

**FastAPI** — Meaning: Python web framework that validates inputs and auto-generates docs. In this project: `api/main.py`.

**Pydantic model** — Meaning: a typed schema that validates request data. In this project: `CustomerData`, and response models.

**Uvicorn** — Meaning: the server that runs FastAPI. In this project: the start command in `render.yaml`.

**Swagger / `/docs`** — Meaning: interactive auto-generated API documentation page. In this project: the live link in the README.

**Render** — Meaning: a cloud host that builds and runs the service from the repo. In this project: `render.yaml` (free plan, so it may sleep when idle).

**Supabase / Postgres / SQLAlchemy** — Meaning: hosted Postgres database / relational database / Python library to talk to it. In this project: the raw table load script.

**Serialisation (pickle / joblib)** — Meaning: saving a Python object to disk. In this project: `.pkl` model files. The library versions at save time and load time must match.

**Training-serving skew** — Meaning: features computed differently at training and prediction time, so the model sees inputs it was not trained on. In this project: the old API bug.

**Risk level** — Meaning: business-friendly bucket derived from probability. In this project: LOW / MEDIUM / HIGH.

**Requirements file** — Meaning: pinned list of Python packages. In this project: `requirements.txt`.

### 4.7 Monitoring terms

**Data drift** — Meaning: input distributions change after deployment. In this project: simulated tenure and charges shift.

**Concept drift / performance drift** — Meaning: the relationship between inputs and churn changes, so accuracy decays. In this project: measured as AUC drop.

**Kolmogorov–Smirnov (KS) test** — Meaning: statistical test comparing two distributions; the KS statistic is the largest gap between their cumulative curves, and a small p-value (< 0.05) suggests they differ. In this project: run per feature.

**p-value** — Meaning: probability of seeing a difference this large if the distributions were actually the same. In this project: `< 0.05` flags drift.

**Retraining trigger** — Meaning: a rule for when to rebuild the model. In this project: an AUC drop above 0.05 flags "action required".

---

## 5. Bugs found and corrected

| # | Problem | Why it was wrong | Fix |
|---|---|---|---|
| 5.1 | `TotalCharges` label-encoded as a category | A dollar amount became arbitrary integer IDs (6,531 unique "categories"), destroying its numeric meaning; the API meanwhile sent raw dollars, so training and serving disagreed | Convert to numeric, blank → 0 |
| 5.2 | `customerID` used as a feature | Unique per row: pure noise, and a memorisation path; the API fed it a dummy 0 | Dropped |
| 5.3 | Scaler fit on the full dataset before the split | Test-set statistics leaked into training (mild leakage) | Fit on the training rows only, using the same split |
| 5.4 | Hardcoded baseline numbers (0.7980 / 0.8406) in `train_advanced.py`, and LR accuracy at the 0.35 threshold compared against RF accuracy at 0.5 | Comparison did not reflect actual results; it mixed two different thresholds | Compute from the real saved model at 0.5; baseline also prints 0.5 accuracy |
| 5.5 | XGBoost got the test set as `eval_set`; production model chosen by test AUC | Test set was influencing training/selection, so the reported score was optimistic | Removed `eval_set`; select by CV AUC on train; removed deprecated `use_label_encoder` |
| 5.6 | Grid search printed the wrong count (`50*3*3*2*2`) and nested `n_jobs=-1` | Misleading log; nested parallelism crashed the machine (paging file too small) | Correct count (36); no nested parallelism |
| 5.7 | API `/predict-batch` used different columns than training; `/health` hard-coded "Random Forest (Tuned)" | Errors or wrong predictions; misleading status | One shared `build_features` using the scaler's own column order; `/health` reports the loaded model |

**Investigated and found NOT to be a bug:** the ROC comparison plots. `lr_model`, `rf_model` and `xgb_model` load different classes (verified by printing `type(...)`), and their AUCs (0.8404 / 0.8387 / 0.8342) are genuinely close. Random Forest is not ~0.87 on the honest split.

---

## 6. Known open issues

**6.1 Hardcoded database password (security).** `scripts/load_data_to_supabase.py` contains a plaintext Supabase connection string, and it is committed in git history. Treat that password as compromised: **rotate it in Supabase**, then change the script to read the connection string from an environment variable (for example `os.environ["DATABASE_URL"]`). Removing it from the file alone does not remove it from git history.

**6.2 The README numbers are out of date and inconsistent with the code.** The README states 85.8 % accuracy, RF AUC 0.8751, and "Selected: Random Forest". The reproducible results are in Part 3 (AUC ≈ 0.84 for all models, Logistic Regression selected). The old figures likely came from an earlier run with different features; they cannot be reproduced from the current code. Update the README to the Part 3 table.

**6.3 Drift simulation is not realistic.** `drift_detection.py` shifts *already-standardised* features (adds noise with std 2 or 3 to values whose std is 1). That noise swamps the signal, so the "production AUC" collapses to ~0.50 (`models/drift_report.json`). That is an artifact of the simulation, not evidence of real decay. Also, the report was generated with the previous production model and old features: re-run `python src/drift_detection.py` after retraining.

**6.4 Deployed API still serves the old model until redeployed.** Commit the regenerated `models/*.pkl` (`scaler.pkl`, `label_encoders.pkl`, `production_model.pkl`) and the updated `api/main.py` together and let Render redeploy. The old API expected a `customerID` feature; the new artifacts do not.

**6.5 Package versions.** `requirements.txt` pins `scikit-learn==1.3.0`, which matches the local environment (1.3.0), good for unpickling. Keep them in sync when upgrading. `render.yaml` uses Python 3.9 while training ran on Python 3.11; this usually works with pickled sklearn models but is worth testing after redeploy.

**6.6 Possible improvements** (none required): one-hot encoding for LR; probability **calibration**; choose the threshold from a cost analysis (cost of a lost customer vs cost of an offer); k-fold CV for RF v1 too; add tests for `build_features`; add engineered features such as charges-per-month-of-tenure.

---

## 7. File map and how to re-run

```
data/raw_data/telco_customers.csv   original data
data/processed/telco_clean.csv      cleaned data
data/processed/X_features.csv       scaled, encoded features (model input)
data/processed/y_target.csv         target (0/1)
notebooks/01_exploratory_analysis.ipynb   EDA
src/feature_engineering.py          encode, scale, split-aware scaler
src/train_baseline.py               Logistic Regression + figure 06
src/train_advanced.py               Random Forest + figure 07
src/tune_hyperparameters.py         grid search -> random_forest_tuned.pkl
src/train_xgboost.py                XGBoost, SHAP, selection, figures 08-10
src/drift_detection.py              KS drift + figure 12 + drift_report.json
api/main.py                         FastAPI service
models/                             saved model, scaler, encoders, importances
figures/                            all plots
render.yaml, requirements.txt       deployment config
```

**Re-run order** (from the project root, with the virtual environment active):

```
python src/feature_engineering.py
python src/train_baseline.py
python src/train_advanced.py
python src/tune_hyperparameters.py
python src/train_xgboost.py
python src/drift_detection.py
```

Each script reads `X_features.csv`, so `feature_engineering.py` must run first, and `train_xgboost.py` needs the baseline and tuned-RF models saved by the earlier scripts. On Windows, set `PYTHONUTF8=1` first because the scripts print emoji.

**Run the API locally:**

```
cd api
uvicorn main:app --reload
# open http://127.0.0.1:8000/docs
```
