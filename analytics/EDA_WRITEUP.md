# Module 2, Part A — EDA write-up

## Missing-value handling (Task 2)

| Column | % missing | Decision | Why |
|---|---|---|---|
| `deck` | 77.22% | **Drop column** | Far above a reliable-imputation range — at this level almost every value would be guessed, not informed. Not needed for the survival analysis this module focuses on. |
| `age` | 19.87% | **Impute with median** | Falls in the 5–30% bucket. Median is used rather than mean because age has a long right tail (some elderly outliers), making the median more robust to skew. |
| `embarked` / `embark_town` | 0.22% (2 rows) | **Drop rows** | Under 5% — dropping 2 rows out of 891 has negligible impact on the dataset. |

## Univariate analysis (Task 3)

- **Outliers (IQR rule):** `age` has **65** outliers; `fare` has **114** outliers. Fare has far more outliers because a small number of first-class passengers paid dramatically more than the typical fare, stretching the upper range.
- **Fare skew:** mean = 32.10, median = 14.45, mode = 8.05. Since **mean > median > mode**, fare is clearly **right-skewed** — most tickets were cheap, but a handful of expensive tickets pull the average upward.

## Bivariate analysis (Task 4)

- **Survival by sex:** women survived at ~74.0%, men at ~18.9% — a massive gap, consistent with "women and children first" evacuation priority.
- **Survival by class:** 1st class ~62.6%, 2nd class ~47.3%, 3rd class ~24.2% — survival drops steadily as class drops, likely reflecting cabin location relative to lifeboats and evacuation priority.
- **Survival by sex + class combined:** the effects compound sharply — 1st class women survived at ~96.7%, while 3rd class men survived at only ~13.5%. Sex mattered more than class on its own: even 3rd class women (~50%) outsurvived 1st class men (~36.9%).
- **Two strongest correlations:** `pclass` ↔ `fare` (**-0.548**) — lower class number (better class) strongly associates with higher fare, which makes sense since class is essentially a proxy for ticket price. `sibsp` ↔ `parch` (**0.415**) — passengers traveling with siblings/spouses also tended to travel with parents/children, i.e., families tended to travel together as a unit rather than splitting these relationships.

## Multivariate "data story" (Task 5)

1. **Survival rate by class and sex** (`story_1_survival_by_class_sex.png`): Confirms sex and class interact rather than acting independently — the survival gap between men and women is present in every class, but widest in 3rd class. This chart is the clearest single visual argument for "who was more likely to survive."
2. **Age distribution by survival** (`story_2_age_by_survival.png`): Survivors and non-survivors have broadly similar age distributions, with survivors skewing very slightly younger. Age alone is a much weaker survival signal than sex or class, consistent with its near-zero correlation with `survived` (-0.07) in the heatmap.
3. **Fare vs age, colored by survival** (`story_3_fare_vs_age.png`): Survivors (green) are visibly denser at higher fare values, while non-survivors (red) dominate the low-fare region. This reinforces that fare (a proxy for class/wealth) mattered more to survival odds than age did.
4. **Survival count by embarkation town** (`story_4_embark_town_survival.png`): Southampton passengers, by far the largest embarkation group, show more non-survivors than survivors in raw counts, while Cherbourg is closer to even. This is likely a class effect in disguise — Cherbourg had a higher proportion of 1st class passengers — rather than embarkation town itself being predictive.

**Overall story:** sex was the single strongest survival factor, class was the second strongest (and correlates with fare), and age had only a weak, secondary effect. This ordering matches the correlation heatmap's magnitudes exactly (`fare` and `pclass` far outweigh `age` in their relationship with `survived`).
