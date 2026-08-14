## Imbalance handling comparison (Task 11)

Train split class balance before any handling: 439 not-survived vs 272 survived (~62/38).

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | 0.766 | 0.721 | 0.742 |
| `class_weight='balanced'` | 0.754 | 0.765 | **0.759** |
| SMOTE (train fold only) | 0.761 | 0.750 | 0.756 |

**Conclusion:** `class_weight='balanced'` gave the best F1 (0.759), driven by the
largest recall improvement of the three strategies (0.765 vs 0.721 baseline) at
a modest precision cost (0.754 vs 0.766). SMOTE also improved over baseline
(F1 0.756) via a smaller recall gain, making it a close second. For this
dataset, **`class_weight='balanced'` is the better choice** if recall
(catching more actual survivors) matters more than precision — it achieved
that goal here with less disruption to the training data than SMOTE's
synthetic oversampling required.