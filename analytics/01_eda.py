"""
01_eda.py
=========
Module 2, Part A — profiling, cleaning, and exploratory data analysis (EDA)
on the Titanic dataset.

Run once, top to bottom. Produces:
  - titanic.csv           (offline fallback copy of the raw dataset)
  - titanic_clean.csv      (cleaned dataset used by 02_modeling.py)
  - charts/*.png           (all required plots)

Usage:
    python 01_eda.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # save charts to file instead of popping up a window
import matplotlib.pyplot as plt
import os

os.makedirs("charts", exist_ok=True)
pd.set_option("display.width", 120)

# ----------------------------------------------------------------------
# Task 1: Load once, profile, save offline fallback
# ----------------------------------------------------------------------
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)  # one-time offline fallback, per the spec

print("=" * 70)
print("TASK 1: Profiling")
print("=" * 70)
print(f"\nShape: {df.shape}")
print("\ndf.info():")
df.info()
print("\ndf.describe():")
print(df.describe())

missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)
missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nColumns with missing values (%):")
print(missing_report)

# ----------------------------------------------------------------------
# Task 2: Missing-value handling, following the threshold rule
#   < 5%   missing -> drop those rows
#   5-30%  missing -> impute
#   very high (>30%, unreliable to impute) -> drop column
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 2: Missing-value handling")
print("=" * 70)

# deck: 77.22% missing -> far above the reliable-imputation range.
# Decision: DROP the column entirely. At ~77% missing, any imputed value
# would be mostly guesswork, and deck isn't needed for the survival
# analysis this module focuses on.
print(f"\ndeck: {missing_pct['deck']}% missing -> DROP COLUMN (too high to impute reliably)")
df = df.drop(columns=["deck"])

# age: 19.87% missing -> falls in the 5-30% bucket -> IMPUTE.
# Decision: impute with the median (age is right-skewed with some
# elderly outliers, so median is more robust than mean here).
print(f"age: {missing_pct['age']}% missing -> IMPUTE with median (5-30% bucket)")
df["age"] = df["age"].fillna(df["age"].median())

# embarked / embark_town: 0.22% missing (just 2 rows) -> < 5% -> DROP ROWS.
print(f"embarked: {missing_pct['embarked']}% missing -> DROP ROWS (< 5% bucket)")
print(f"embark_town: {missing_pct['embark_town']}% missing -> DROP ROWS (< 5% bucket)")
df = df.dropna(subset=["embarked", "embark_town"])

print(f"\nShape after cleaning: {df.shape}")
print(f"Remaining missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ----------------------------------------------------------------------
# Task 3: Univariate analysis — age and fare
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 3: Univariate analysis (age, fare)")
print("=" * 70)


def iqr_outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df[col], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{col} — histogram")
    axes[1].boxplot(df[col])  # vertical by default
    axes[1].set_title(f"{col} — box plot")
    plt.tight_layout()
    plt.savefig(f"charts/univariate_{col}.png", dpi=100)
    plt.close()

    n_outliers = iqr_outlier_count(df[col])
    print(f"{col}: {n_outliers} IQR-based outliers")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
print(f"\nfare — mean: {fare_mean:.2f}, median: {fare_median:.2f}, mode: {fare_mode:.2f}")
print("mean > median > mode -> right-skewed distribution"
      if fare_mean > fare_median > fare_mode else "see values above for skew direction")

# ----------------------------------------------------------------------
# Task 4: Bivariate analysis — survival rate breakdowns + correlation heatmap
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4: Bivariate analysis")
print("=" * 70)

survival_by_sex = {
    sex: df[df["sex"] == sex]["survived"].mean()
    for sex in df["sex"].unique()
}
print(f"\nSurvival rate by sex: {survival_by_sex}")

survival_by_class = {
    pclass: df[df["pclass"] == pclass]["survived"].mean()
    for pclass in sorted(df["pclass"].unique())
}
print(f"Survival rate by pclass: {survival_by_class}")

survival_by_sex_class = {}
for sex in df["sex"].unique():
    for pclass in sorted(df["pclass"].unique()):
        mask = (df["sex"] == sex) & (df["pclass"] == pclass)
        survival_by_sex_class[(sex, pclass)] = df[mask]["survived"].mean()
print(f"Survival rate by sex+pclass: {survival_by_sex_class}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print(f"\nCorrelation matrix:\n{corr_matrix.round(2)}")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (6 numeric columns)")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=100)
plt.close()

# Find the two strongest off-diagonal correlations (by absolute value)
pairs = []
for i, c1 in enumerate(corr_cols):
    for c2 in corr_cols[i + 1:]:
        pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
print("\nTop 2 strongest correlations:")
for c1, c2, val in pairs_sorted[:2]:
    print(f"  {c1} <-> {c2}: {val:.3f}")

# ----------------------------------------------------------------------
# Task 5: Multivariate "data story" — at least 4 charts
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 5: Multivariate charts")
print("=" * 70)

# Chart 1: survival rate by class and sex (grouped bar)
fig, ax = plt.subplots(figsize=(7, 5))
pivot = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
pivot.plot(kind="bar", ax=ax)
ax.set_title("Survival rate by class and sex")
ax.set_ylabel("Survival rate")
plt.tight_layout()
plt.savefig("charts/story_1_survival_by_class_sex.png", dpi=100)
plt.close()

# Chart 2: age distribution split by survival (box plot)
fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column="age", by="survived", ax=ax)
ax.set_title("Age distribution by survival")
plt.suptitle("")
plt.tight_layout()
plt.savefig("charts/story_2_age_by_survival.png", dpi=100)
plt.close()

# Chart 3: fare vs age scatter, colored by survival
fig, ax = plt.subplots(figsize=(7, 5))
for survived_val, color in [(0, "crimson"), (1, "seagreen")]:
    subset = df[df["survived"] == survived_val]
    ax.scatter(subset["age"], subset["fare"], alpha=0.5, label=f"survived={survived_val}", color=color)
ax.set_xlabel("age")
ax.set_ylabel("fare")
ax.set_title("Fare vs age, colored by survival")
ax.legend()
plt.tight_layout()
plt.savefig("charts/story_3_fare_vs_age.png", dpi=100)
plt.close()

# Chart 4: survival count by embarkation town
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x="embark_town", hue="survived", ax=ax)
ax.set_title("Survival count by embarkation town")
plt.tight_layout()
plt.savefig("charts/story_4_embark_town_survival.png", dpi=100)
plt.close()

print("Saved 4 story charts to charts/")

# ----------------------------------------------------------------------
# Task 6: Z-score standardization sanity check (EDA-only, not modeling)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 6: Z-score standardization check (EDA sanity check only)")
print("=" * 70)

for col in ["age", "fare"]:
    before_mean, before_std = df[col].mean(), df[col].std()
    z = (df[col] - before_mean) / before_std
    print(f"{col} — before: mean={before_mean:.2f}, std={before_std:.2f} | "
          f"after z-score: mean={z.mean():.4f}, std={z.std():.4f}")

# ----------------------------------------------------------------------
# Save cleaned data for 02_modeling.py to pick up
# ----------------------------------------------------------------------
df.to_csv("titanic_clean.csv", index=False)
print("\nSaved titanic_clean.csv for the modeling stage.")
