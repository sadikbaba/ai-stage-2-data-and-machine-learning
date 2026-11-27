import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from lib.manual_imputer import ManualSimpleImputer
from lib.manual_scaler import ManualStandardScaler
from lib.manual_onehot import ManualOneHotEncoder
from lib.manual_column_transformer import ManualColumnTransformer
from lib.manual_linear_regression import ManualLinearRegression
from lib.manual_pipeline import ManualPipeline


from sklearn.compose import ColumnTransformer as SklearnColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LinearRegression

# Load dataset
delivery_data = pd.read_csv("data/delivery.csv")

X = delivery_data[["distance_km", "weather"]]
y = delivery_data["delivery_minutes"]


# Split data: 60% training, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# Numeric preprocessing branch
numeric_steps = [
    ManualSimpleImputer(strategy="median"),
    ManualStandardScaler(),
]

# Categorical preprocessing branch
categorical_steps = [
    ManualSimpleImputer(strategy="most_frequent"),
    ManualOneHotEncoder(drop_first=True),
]


# Combine preprocessing branches
transformers = [
    ("numeric", numeric_steps, ["distance_km"]),
    ("categorical", categorical_steps, ["weather"]),
]

preprocessor = ManualColumnTransformer(transformers=transformers)


# Create the complete pipeline
pipeline = ManualPipeline(
    preprocessor=preprocessor,
    estimator=ManualLinearRegression(),
)


# Train
pipeline.fit(X_train, y_train)
transformed_train = preprocessor.transform(X_train)

print("Transformed training data:")
print(transformed_train)

print("Matrix shape:", transformed_train.shape)
print("Matrix rank:", np.linalg.matrix_rank(transformed_train))


# Predict on validation data
predictions = pipeline.predict(X_validation)

print("Validation predictions:", predictions)
print("Actual values:", y_validation.tolist())
print("Prediction count:", len(predictions))
print("Validation row count:", len(X_validation))


assert len(predictions) == len(y_validation)
assert np.isfinite(predictions).all()
assert transformed_train.shape[0] == len(X_train)

print("All ManualPipeline checks passed.")


# Equivalent scikit-learn preprocessing
sklearn_numeric = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

sklearn_categorical = SklearnPipeline(
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
)

sklearn_preprocessor = SklearnColumnTransformer(
    [
        ("numeric", sklearn_numeric, ["distance_km"]),
        ("categorical", sklearn_categorical, ["weather"]),
    ]
)

sklearn_pipeline = SklearnPipeline(
    [
        ("preprocessor", sklearn_preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Train and predict
sklearn_pipeline.fit(X_train, y_train)
sklearn_predictions = sklearn_pipeline.predict(X_validation)

print("\nScikit-learn predictions:", sklearn_predictions)
print("Manual predictions:", predictions)

# Verify equivalent predictions
np.testing.assert_allclose(
    predictions,
    sklearn_predictions,
    rtol=1e-6,
    atol=1e-6,
)

print("Manual and scikit-learn predictions match.")
