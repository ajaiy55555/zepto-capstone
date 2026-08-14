# Module 2 — Analytics Pipeline

Profiles and cleans the Titanic dataset, tells a visual "who survived and why"
story, then builds and rigorously evaluates a full classification pipeline
(plus a regression side-task) on the same cleaned data.

## Install

```bash
pip install -r requirements.txt
```

## Run (in order)

```bash
python 01_eda.py        # loads, profiles, cleans, explores -> titanic_clean.csv + charts/
python 02_modeling.py   # trains/evaluates models, tunes, saves best_pipeline.joblib
```

`02_modeling.py` depends on `titanic_clean.csv`, so `01_eda.py` must run first.
The raw dataset is fetched once (via `sns.load_dataset`, cached locally after
the first run) and is never reloaded independently for modeling — everything
downstream works from the one cleaned CSV.

## Files

| File | Purpose |
|---|---|
| `01_eda.py` | Part A — loads, profiles, cleans, and explores the data. Outputs `titanic.csv` (raw fallback), `titanic_clean.csv`, and 7 chart PNGs. |
| `02_modeling.py` | Part B — stratified split, preprocessing pipeline, 3 classifiers, imbalance comparison, GridSearchCV tuning, regression side-task, saves the best pipeline. |
| `EDA_WRITEUP.md` | Required written interpretations for Part A (missing-value decisions, outlier counts, skew, correlations, chart-by-chart story). |
| `MODELING_WRITEUP.md` | Required written interpretations for Part B (imbalance comparison, tuning results, heteroscedasticity check, final model comparison + recommendation). |
| `titanic.csv` | One-time offline fallback of the raw dataset, loadable via `pd.read_csv` if `sns.load_dataset` can't reach the network. |
| `best_pipeline.joblib` | The saved, complete, best-performing pipeline (preprocessing + model together), reloadable and usable on raw input. |
| `charts/` | All required plots as PNG supporting artifacts. |

## Design decisions

- **Missing-value handling** follows the assignment's threshold rule exactly:
  `deck` (77.22% missing) is dropped as a column; `age` (19.87% missing) is
  median-imputed; `embarked`/`embark_town` (0.22% missing, 2 rows) have their
  rows dropped. Full reasoning in `EDA_WRITEUP.md`.
- **Modeling features** deliberately exclude `alive` (a direct text restatement
  of the `survived` target — including it would leak the answer), `class`
  (a text duplicate of `pclass`), and `who`/`adult_male`/`alone` (all derived
  from other features already in use). The feature set used is `pclass`, `sex`,
  `age`, `sibsp`, `parch`, `fare`, `embarked`.
- **Preprocessing** is a `ColumnTransformer` (median-impute + scale for numeric,
  most-frequent-impute + one-hot for categorical) wrapped in a `Pipeline` per
  model, so it's structurally fit on the training fold only — never on test
  data or the full pre-split dataset.
- **Imbalance handling** is compared using Random Forest specifically, since
  it's the model that goes on to be tuned in the next task — keeping that
  thread consistent. SMOTE is applied only inside the training fold.
- **The saved `best_pipeline.joblib`** is the best of the 3 initial classifiers
  by F1 score (Random Forest) — not the separately-tuned GridSearchCV model or
  the SMOTE variant, which are their own exploratory comparisons documented in
  `MODELING_WRITEUP.md` rather than the final deployed artifact.

See `EDA_WRITEUP.md` and `MODELING_WRITEUP.md` for every required written
interpretation, the full model comparison table, and the final recommendation.
