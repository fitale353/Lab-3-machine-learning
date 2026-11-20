# Lab 3: Ticket Escalation Binary Classification & Cost-Sensitive Thresholding

## Overview
Binary classification and threshold cost analysis on `classification_escalation.csv` (`900` tickets, `20.67%` escalated) using a `ColumnTransformer` preprocessing pipeline (`SimpleImputer` median + `StandardScaler` for numeric features; `SimpleImputer` most-frequent + `OneHotEncoder` for categorical features) and a stratified `75/25` train/test split (`random_state=42`).

## Model Comparison (Test Set @ Threshold `0.50`)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | `0.7956` | `0.0000` | `0.0000` | `0.0000` | `0.5000` |
| Logistic Regression | `0.8222` | `0.7500` | `0.1957` | `0.3103` | `0.8111` |
| Linear Discriminant Analysis | `0.8222` | `0.7500` | `0.1957` | `0.3103` | `0.8090` |
| k-NN (`k=15`) | `0.7956` | `0.5000` | `0.0870` | `0.1481` | `0.7046` |

## Threshold & Cost Evaluation (`5 * FN + 1 * FP`)
| Threshold | Precision | Recall | F1 | FP | FN | Cost (`5*FN + 1*FP`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.20` | `0.4286` | `0.7826` | `0.5538` | `48` | `10` | `98` |
| `0.30` | `0.5490` | `0.6087` | `0.5773` | `23` | `18` | `113` |
| `0.40` | `0.5714` | `0.2609` | `0.3582` | `9` | `34` | `179` |
| `0.50` | `0.7500` | `0.1957` | `0.3103` | `3` | `37` | `188` |
| `0.60` | `1.0000` | `0.0652` | `0.1224` | `0` | `43` | `215` |
| `0.70` | `0.0000` | `0.0000` | `0.0000` | `0` | `46` | `230` |

- **Lowest Test Cost Threshold:** `0.20` (Cost = `98`). Theoretical cost-optimal threshold for `C_FP = 1, C_FN = 5`: `p* = 1 / (1 + 5) ≈ 0.1667`.
- **Stratified 5-Fold CV (`k-NN` F1):** Best `k = 5` with mean CV F1 = `0.2433`.

## Contents
- `classification_escalation.csv`: Dataset (`900` rows, `6` columns).
- `Lab_3_Ticket_Escalation.ipynb`: Completed notebook with executed outputs, plots, Q&A, and 347-word deployment recommendation.
- `lr_evaluation.png`: Confusion matrix and ROC curve for Logistic Regression (`t = 0.50`).
