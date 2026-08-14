"""
02_modeling.py
===============
Module 2, Part B — predictive modeling on the cleaned Titanic data.

Reads titanic_clean.csv (produced by 01_eda.py) and:
  - trains/evaluates 3 classifiers (Logistic Regression, Decision Tree, Random Forest)
  - compares 3 ways of handling class imbalance
  - tunes the Random Forest with GridSearchCV
  - runs a linear regression side-task predicting fare
  - saves the best full pipeline with joblib

Usage:
    python 02_modeling.py
    (must be run after 01_eda.py, since it reads titanic_clean.csv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import joblib

os.makedirs("charts", exist_ok=True)

# ----------------------------------------------------------------------
# Load the cleaned data from Part A
# ----------------------------------------------------------------------
df = pd.read_csv("titanic_clean.csv")

# Features used for modeling. Deliberately EXCLUDING:
#   'alive'      - this is just `survived` written as "yes"/"no" text.
#                  Including it would leak the answer directly into the model.
#   'class'      - just `pclass` written as text ("Third" instead of 3).
#   'who', 'adult_male', 'alone' - derived from age/sex/sibsp/parch, redundant.
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "survived"

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------------------
# Task 7: stratified train/test split
# ----------------------------------------------------------------------
print("=" * 70)
print("TASK 7: Train/test split")
print("=" * 70)
class_balance = y.value_counts(normalize=True)
print(f"Class balance:\n{class_balance}")
print("Using stratify=y so both train and test sets keep this same ~62/38 "
      "split -- without it, a random split could accidentally end up with "
      "very different survival rates in train vs test, making evaluation unreliable.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ----------------------------------------------------------------------
# Task 8: preprocessing pipeline (fit on train only)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 8: Preprocessing pipeline")
print("=" * 70)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])
print("ColumnTransformer built: numeric -> impute(median)+scale, "
      "categorical (sex, embarked) -> impute(most_frequent)+one-hot encode.")
print("This preprocessor will be fit ONLY on X_train inside each model's "
      "Pipeline below -- never on X_test or the full dataset.")

# ----------------------------------------------------------------------
# Task 9 & 10: train 3 classifiers, evaluate all of them
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 9 & 10: Train and evaluate 3 classifiers")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

fitted_pipelines = {}
results = []

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
    })

    print(f"\n--- {name} ---")
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")

results_df = pd.DataFrame(results).set_index("model")
print("\n=== Classifier comparison table ===")
print(results_df.round(3))

# ROC curves, all 3 on one chart
fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — all 3 classifiers")
ax.legend()
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=100)
plt.close()

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = dt_pipe.named_steps["preprocessor"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax,
)
ax.set_title("Decision Tree (top 3 levels shown for readability)")
plt.tight_layout()
plt.savefig("charts/decision_tree.png", dpi=100)
plt.close()
print("\nSaved charts/roc_curves.png and charts/decision_tree.png")

# ----------------------------------------------------------------------
# Task 11: imbalance handling comparison (using Random Forest)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 11: Imbalance handling comparison (Random Forest)")
print("=" * 70)

# Preprocess once, reuse the transformed arrays for all 3 variants
preprocessor_imb = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])
X_train_proc = preprocessor_imb.fit_transform(X_train)
X_test_proc = preprocessor_imb.transform(X_test)

imbalance_results = []

# (a) Baseline - no handling
rf_baseline = RandomForestClassifier(n_estimators=200, random_state=42)
rf_baseline.fit(X_train_proc, y_train)
pred_a = rf_baseline.predict(X_test_proc)
imbalance_results.append({
    "strategy": "Baseline (no handling)",
    "precision": precision_score(y_test, pred_a),
    "recall": recall_score(y_test, pred_a),
    "f1": f1_score(y_test, pred_a),
})

# (b) class_weight='balanced'
rf_weighted = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf_weighted.fit(X_train_proc, y_train)
pred_b = rf_weighted.predict(X_test_proc)
imbalance_results.append({
    "strategy": "class_weight='balanced'",
    "precision": precision_score(y_test, pred_b),
    "recall": recall_score(y_test, pred_b),
    "f1": f1_score(y_test, pred_b),
})

# (c) SMOTE oversampling - applied ONLY to the training fold, never to test
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
rf_smote = RandomForestClassifier(n_estimators=200, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
pred_c = rf_smote.predict(X_test_proc)
imbalance_results.append({
    "strategy": "SMOTE (train fold only)",
    "precision": precision_score(y_test, pred_c),
    "recall": recall_score(y_test, pred_c),
    "f1": f1_score(y_test, pred_c),
})

imbalance_df = pd.DataFrame(imbalance_results).set_index("strategy")
print(f"\nOriginal train class balance: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
print(f"\n{imbalance_df.round(3)}")

best_strategy = imbalance_df["f1"].idxmax()
print(f"\nBest F1: {best_strategy}")

# ----------------------------------------------------------------------
# Task 12: GridSearchCV tuning on Random Forest
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 12: GridSearchCV tuning (Random Forest)")
print("=" * 70)

rf_grid_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
])
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}
grid_search = GridSearchCV(rf_grid_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_.named_steps["classifier"]
print(f"OOB score of best model: {best_rf.oob_score_:.3f}")
print(f"Best cross-val F1 score: {grid_search.best_score_:.3f}")

tuned_test_pred = grid_search.predict(X_test)
tuned_test_f1 = f1_score(y_test, tuned_test_pred)
print(f"Tuned model's F1 on held-out test set: {tuned_test_f1:.3f}")

# ----------------------------------------------------------------------
# Task 13: regression side-task — predict fare
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 13: Regression side-task (predict fare)")
print("=" * 70)

REG_NUMERIC = ["pclass", "age", "sibsp", "parch", "survived"]
REG_CATEGORICAL = ["sex", "embarked"]
X_reg = df[REG_NUMERIC + REG_CATEGORICAL]
y_reg = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), REG_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), REG_CATEGORICAL),
])
reg_pipe = Pipeline(steps=[("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)

yr_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)

n = len(yr_test)
p = reg_pipe.named_steps["preprocessor"].transform(Xr_test).shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.3f}")
print(f"Adjusted R2: {adj_r2:.3f}  (n={n} test samples, p={p} encoded predictors)")

residuals = yr_test - yr_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(yr_pred, residuals, alpha=0.5)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual plot — fare regression")
plt.tight_layout()
plt.savefig("charts/regression_residuals.png", dpi=100)
plt.close()

pred_median = np.median(yr_pred)
resid_std_low = residuals[yr_pred < pred_median].std()
resid_std_high = residuals[yr_pred >= pred_median].std()
print(f"\nResidual std for low predicted fares: {resid_std_low:.2f}")
print(f"Residual std for high predicted fares: {resid_std_high:.2f}")
print("Spread of residuals clearly widens at higher predicted fares -> "
      "HETEROSCEDASTICITY present (non-random, fanning-out spread of residuals)."
      if resid_std_high > resid_std_low * 1.5 else
      "Residual spread looks roughly stable across the range -> little evidence of heteroscedasticity.")

# ----------------------------------------------------------------------
# Task 14: final comparison table (printed; write-up goes in README)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 14: Final comparison")
print("=" * 70)
print("\nClassification models:")
print(results_df.round(3))
print("\nRegression model:")
print(pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2}]).round(3))

# ----------------------------------------------------------------------
# Task 15: save the best-performing full pipeline
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 15: Save best pipeline")
print("=" * 70)

best_model_name = results_df["f1"].idxmax()
best_pipeline = fitted_pipelines[best_model_name]
print(f"Best classifier by F1: {best_model_name}")

joblib.dump(best_pipeline, "best_pipeline.joblib")
print("Saved best_pipeline.joblib")

# Reload and confirm it works on raw, unprocessed input
reloaded = joblib.load("best_pipeline.joblib")
sample_raw = X_test.iloc[[0]]
original_pred = best_pipeline.predict(sample_raw)[0]
reloaded_pred = reloaded.predict(sample_raw)[0]
print(f"\nSample raw input:\n{sample_raw}")
print(f"Original pipeline prediction: {original_pred}")
print(f"Reloaded pipeline prediction: {reloaded_pred}")
print(f"Match: {original_pred == reloaded_pred}")
