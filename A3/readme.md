# Predicting Fast-Growing Firms (Bisnode Panel)

Classification project that identifies high-growth firms from a firm-level panel, with a business-driven decision threshold and a comparison across industries.

## Business question

Which firms will grow fastest next year? A lender or investor wants to find them early, but wrongly flagging firms costs money and missing real growers costs more. The project builds probability models and picks a decision threshold from an explicit cost assumption.

## Data

- Bisnode firm panel: 287,829 firm-year rows, 46,412 firms, 2005 to 2016.
- Restricted to 2010 to 2015, with 2012 as the base year.
- Modelling sample: 19,446 firms with valid sales in 2012 and 2013, after removing firms that exited and extreme growth values.

## Target

A firm is **fast-growing** if its 2012 to 2013 sales growth is in the top 10% of the sample. That corresponds to growth of roughly 100% or more, so these firms at least doubled their sales. This gives 1,945 fast growers (10%) and 17,501 others.

## Features

Base-year sales, average labour, industry (NACE code), region, urban location, foreign ownership, share of female managers, number of CEOs and days in office, plus missing-value flags.

## Approach

1. Clean the panel, build the growth target and explore its distribution.
2. Preprocess inside a scikit-learn pipeline (median and mode imputation, scaling, one-hot encoding).
3. Train four models on an 80/20 stratified split: Logistic Regression, LASSO logistic regression, Random Forest and XGBoost.
4. Evaluate with AUC, Brier score and calibration curves.
5. Choose a threshold per model by minimising expected loss, assuming 50,000 per missed fast grower and 10,000 per false alarm.
6. Repeat the best model separately for manufacturing and services firms.

## Results

| Model | AUC | Brier | Best threshold | Expected loss per firm |
|---|---|---|---|---|
| Logistic Regression | 0.699 | 0.086 | 0.136 | 4,246 |
| LASSO Logistic | 0.703 | 0.086 | 0.136 | 4,236 |
| Random Forest | 0.757 | 0.082 | 0.247 | 3,850 |
| **XGBoost** | **0.785** | **0.079** | **0.127** | **3,627** |

XGBoost ranks firms best and has the lowest cost. At the default 0.5 cut-off it flags too few firms (recall 15%), so the cost-based threshold matters.

### Confusion matrix (XGBoost, threshold 0.127, test set of 3,890 firms)

| | Actual: not fast | Actual: fast |
|---|---|---|
| **Predicted: not fast** | 2,815 | 145 |
| **Predicted: fast** | 686 | 244 |

The model finds 63% of fast-growing firms (recall 0.63), with 26% of flagged firms being true growers (precision).

### Calibration and drivers

- XGBoost probabilities follow the observed rates reasonably well and slightly overpredict in the middle range. Random Forest overpredicts more, and the logistic models never predict above about 0.45.
- Sales and firm size (labour) are the strongest signals. XGBoost also leans on industry and on whether ownership data is missing, and the Random Forest uses days in office and urban location.

### Manufacturing vs accommodation and food services

| Sector | Firms | AUC | Expected loss per firm |
|---|---|---|---|
| Manufacturing | 5,889 | 0.720 | 3,853 |
| Accommodation and food services | 12,918 | 0.787 | 3,568 |

Growth is easier to predict in the services sample, although the gap is based on small test sets.

## Limitations and next steps

- The threshold was tuned on the test set, so costs are optimistic. A validation split or cross-validation would fix this.
- Cost values are assumptions, not observed figures.
- Features exclude balance-sheet variables such as assets, liabilities and profit, which would likely improve accuracy.
- Add cross-validation, hyperparameter tuning and confidence intervals before comparing models closely.

## Tech

Python, pandas, scikit-learn, XGBoost, seaborn, matplotlib.

