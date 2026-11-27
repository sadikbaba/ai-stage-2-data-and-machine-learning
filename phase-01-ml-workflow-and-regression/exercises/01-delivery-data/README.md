# Delivery Time Prediction - Regression Project

## 1. Project Overview

This project uses machine learning regression models to predict
delivery time in minutes based on delivery distance and weather.

## 2. Dataset

File: `data/delivery.csv`

- Samples: 10
- Features:
  - `distance_km`: delivery distance
  - `weather`: weather condition
- Target:
  - `delivery_minutes`: delivery duration

The dataset is small and is intended for educational purposes.

## 3. Data Preparation

The dataset is split into:

- 60% training
- 20% validation
- 20% testing

Random state: 42.

Preprocessing is fitted using training data only.

### Numeric feature

- Missing values: median imputation
- Scaling: standardization

### Categorical feature

- Missing values: most-frequent imputation
- Encoding: one-hot encoding with the first category dropped

## 4. Models

Five approaches were compared:

1. Mean baseline regression
2. Manual Linear Regression
3. Scikit-learn Linear Regression
4. Manual KNN Regression
5. Scikit-learn KNN Regression

The manual implementations were compared against their
scikit-learn equivalents.

## 5. Validation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 5.50 | 7.43 | -0.83 |
| Manual Linear Regression | 1.60 | 1.80 | 0.89 |
| Scikit-learn Linear Regression | 1.60 | 1.80 | 0.89 |
| Manual KNN | 6.00 | 8.14 | -1.19 |
| Scikit-learn KNN | 6.00 | 8.14 | -1.19 |

Linear Regression performed best on the validation split.

## 6. Final Test Results

Selected model: Manual Linear Regression

- MAE: 8.87 minutes
- MSE: 126.50
- RMSE: 11.25 minutes
- R²: -0.265

Manual Linear Regression and Scikit-learn Linear Regression
produced matching test predictions.

## 7. Important Findings

- Manual Linear Regression matched Scikit-learn.
- Manual KNN matched Scikit-learn.
- Linear Regression outperformed the baseline and KNN on validation.
- Test performance was substantially worse than validation.
- The dataset contains only 10 observations, so results cannot
  establish reliable real-world generalization.

## 8. Running the Project

Activate the project virtual environment, then run:

```bash
python -m training.manual_main
```

Run the manual pipeline test:

```bash
python -m training.test_manual_pipeline
```

## 9. Project Limitations

- Very small dataset.
- Only two input features.
- Validation and test sets contain only two observations each.
- Model performance may change substantially with a different split.


This project is an educational demonstration of a complete
regression workflow, not a production delivery-time prediction system.