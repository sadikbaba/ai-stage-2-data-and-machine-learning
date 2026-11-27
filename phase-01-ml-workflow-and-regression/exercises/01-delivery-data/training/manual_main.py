import numpy as np
import pandas as pd
from lib.manual_knn_regression import ManualKNNRegressor
from lib.manual_linear_regression import ManualLinearRegression
from lib.manual_onehot import ManualOneHotEncoder
from lib.manual_scaler import ManualStandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from lib.manual_imputer import ManualSimpleImputer
from lib.manual_column_transformer import ManualColumnTransformer
from lib.manual_pipeline import ManualPipeline
from lib.regression_metrics import RegressionMetrics

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

delivery_data = pd.read_csv("data/delivery.csv")

x = delivery_data[["distance_km", "weather"]]
y = delivery_data["delivery_minutes"]
# Split: 60% training, 20% validation, 20% testing
X_train, X_temp, Y_train, Y_temp = train_test_split(
    x, y, test_size=0.4, random_state=42
)

X_validation, X_test, Y_validation, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)


# Manual numeric preprocessing
numeric_steps = [
    ManualSimpleImputer(strategy="median"),
    ManualStandardScaler(),
]

# Manual categorical preprocessing
categorical_steps = [
    ManualSimpleImputer(strategy="most_frequent"),
    ManualOneHotEncoder(drop_first=True),
]

transformers = [
    ("numeric", numeric_steps, ["distance_km"]),
    ("categorical", categorical_steps, ["weather"]),
]

manual_preprocessor = ManualColumnTransformer(transformers=transformers)


delivery_data = pd.read_csv("data/delivery.csv")

X = delivery_data[["distance_km", "weather"]]
y = delivery_data["delivery_minutes"]

# 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Manual preprocessing
numeric_steps = [
    ManualSimpleImputer(strategy="median"),
    ManualStandardScaler(),
]

categorical_steps = [
    ManualSimpleImputer(strategy="most_frequent"),
    ManualOneHotEncoder(drop_first=True),
]

manual_preprocessor = ManualColumnTransformer(
    [
        ("numeric", numeric_steps, ["distance_km"]),
        ("categorical", categorical_steps, ["weather"]),
    ]
)

# Manual models
manual_linear = ManualPipeline(
    preprocessor=manual_preprocessor,
    estimator=ManualLinearRegression(),
)

manual_knn = ManualPipeline(
    preprocessor=ManualColumnTransformer(
        [
            ("numeric", numeric_steps, ["distance_km"]),
            ("categorical", categorical_steps, ["weather"]),
        ]
    ),
    estimator=ManualKNNRegressor(k=2),
)

# Scikit-learn preprocessing
sklearn_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            ["distance_km"],
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            ["weather"],
        ),
    ]
)

sklearn_linear = Pipeline(
    [
        ("preprocessor", sklearn_preprocessor),
        ("model", LinearRegression()),
    ]
)

sklearn_knn = Pipeline(
    [
        ("preprocessor", sklearn_preprocessor),
        ("model", KNeighborsRegressor(n_neighbors=2)),
    ]
)

# Fit models using training data only
manual_linear.fit(X_train, y_train)
manual_knn.fit(X_train, y_train)
sklearn_linear.fit(X_train, y_train)
sklearn_knn.fit(X_train, y_train)

# Baseline: predict the training target mean
baseline_prediction = y_train.mean()
baseline_predictions = [baseline_prediction] * len(y_validation)

# Validation predictions
predictions = {
    "Baseline": baseline_predictions,
    "Manual Linear Regression": manual_linear.predict(X_validation),
    "Sklearn Linear Regression": sklearn_linear.predict(X_validation),
    "Manual KNN": manual_knn.predict(X_validation),
    "Sklearn KNN": sklearn_knn.predict(X_validation),
}

# Compare regression metrics
print("=== VALIDATION MODEL COMPARISON ===")

for model_name, model_predictions in predictions.items():
    print(f"\n{model_name}")
    print("MAE:", RegressionMetrics.mae(y_validation.tolist(), list(model_predictions)))
    print("MSE:", RegressionMetrics.mse(y_validation.tolist(), list(model_predictions)))
    print(
        "RMSE:", RegressionMetrics.rmse(y_validation.tolist(), list(model_predictions))
    )
    print(
        "R²:",
        RegressionMetrics.r_square(y_validation.tolist(), list(model_predictions)),
    )

print("\nActual validation values:", y_validation.tolist())

# Final evaluation on the untouched test set
test_predictions = manual_linear.predict(X_test)

print("\n=== FINAL TEST-SET EVALUATION ===")
print("Actual values:", y_test.tolist())
print("Predictions:", test_predictions)

print("MAE:", RegressionMetrics.mae(y_test.tolist(), test_predictions))
print("MSE:", RegressionMetrics.mse(y_test.tolist(), test_predictions))
print("RMSE:", RegressionMetrics.rmse(y_test.tolist(), test_predictions))
print("R²:", RegressionMetrics.r_square(y_test.tolist(), test_predictions))

# Confirm manual and sklearn linear regression agree on test data
sklearn_test_predictions = sklearn_linear.predict(X_test)

np.testing.assert_allclose(
    test_predictions,
    sklearn_test_predictions,
    rtol=1e-6,
    atol=1e-6,
)

print("Manual and sklearn test predictions match.")
