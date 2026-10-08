# Airbnb Price Prediction: Paris and Lyon

Predictive pricing model for a chain of Airbnb properties, built on [Inside Airbnb](https://insideairbnb.com/) listings. The project compares linear and machine-learning models on Paris listings and then checks how well the approach holds up over time (a later quarter) and across cities (Lyon).

## Business question

How should a property chain price its listings? The goal is a model that predicts nightly price from listing characteristics, and an understanding of which features drive price and how stable the model is outside the data it was built on.

## Data

- **Paris:** 95,885 listings (scraped June 2024), 74,579 with a valid price. About 89% are entire homes or apartments.
- **Validation sets:** a later Paris period (Q3) and Lyon (about 5,100 listings).
- Median price is about 164 EUR, while the mean is about 289 EUR because of a long right tail, so extreme prices were removed (below 2,000 EUR retained).

## Approach

1. **Cleaning:** price converted to numeric, missing values imputed (median for numeric fields, guests for beds), listing and room types grouped.
2. **Feature engineering:** the 100 most frequent amenities turned into binary indicators, plus size, review, host and booking-rule variables.
3. **Exploration:** price distributions by room type, property type, superhost status, reviews and capacity.
4. **Models:** OLS (on LASSO-selected features), LASSO, Random Forest, Gradient Boosting and an MLP neural network, compared on a held-out 20% test set, 5-fold cross-validation, R-squared and run time.
5. **Validity:** the same pipeline applied to a later quarter in Paris and to Lyon.

## Results

### Paris, model comparison

| Model | Test RMSE (EUR) | CV RMSE (EUR) | R² |
|---|---|---|---|
| OLS (LASSO-selected) | 163.3 | 162.7 | 0.375 |
| LASSO | 163.4 | 162.8 | 0.375 |
| Random Forest | 174.4 | 180.5 | 0.440 |
| Gradient Boosting | 175.0 | 177.5 | 0.436 |
| MLP | 180.2 | 271.5 | 0.402 |

All models explain between roughly 38% and 44% of price variation. The MLP was the least stable (its cross-validated error is far above its test error and training did not fully converge).

### Lyon

| Model | Test RMSE (EUR) | R² |
|---|---|---|
| Random Forest | 136.1 | 0.750 |
| Gradient Boosting | 200.5 | 0.459 |
| MLP | 222.9 | 0.331 |
| OLS (LASSO-selected) | 254.6 | 0.127 |
| LASSO | 256.4 | 0.115 |

Random Forest was clearly the strongest model in Lyon, while the linear models performed poorly, which suggests pricing there depends on non-linear effects.

### Paris, later quarter (Q3)

Results to be added.

### What drives price

- Listing size dominates: **bathrooms and bedrooms** are the top two features in both Random Forest and Gradient Boosting for Paris, followed by review score, minimum nights and guest capacity.
- In Lyon, bedrooms, guest capacity, elevator access and bathrooms matter most.
- Raw average price differences are largest for amenities such as an indoor fireplace (about +220 EUR), air conditioning (about +210 EUR) and free parking (about +185 EUR). These differences are partly driven by larger, more upscale listings, so they are associations and not causal effects.

## Limitations and next steps

- Add location (neighbourhood and coordinates), which is a likely major driver of price.
- Model log price to handle the skewed distribution.
- Fit preprocessing and feature selection inside a cross-validated pipeline to avoid leakage.
- Score the Paris-trained model directly on Q3 and Lyon to measure true transferability.

## Tech

Python, pandas, scikit-learn, seaborn, matplotlib.

