import numpy as np
import pandas as pd
from lib.manual_knn_regression import ManualKNNRegressor
from lib.manual_linear_regression import ManualLinearRegression
from lib.manual_onehot import ManualOneHotEncoder
from lib.manual_scaler import ManualStandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

delivery_data = pd.read_csv("data/delivery.csv")

x = delivery_data[["distance_km", "weather"]]
y = delivery_data["delivery_minutes"]

# Split the data into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Split the training data again to create a validation set.
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.25, random_state=42
)  # .25 here means 25% of train data, which is overral data 60% train,
# 20% validation, 20% test, all == 100


# Calculate the median distance using training data.
median_distance = X_train["distance_km"].median()

# Fill missing distances in the test set using the training median.
# The missing distance falls in the test set for this split.
X_test["distance_km"] = X_test["distance_km"].fillna(median_distance)


# # feature engineering: create a new feature "rainy_distance" that is the distance if the weather is rainy, otherwise 0
# X_train["rainy_distance"] = np.where(
#     X_train["weather"] == "rain", X_train["distance_km"], 0
# )

# X_validation["rainy_distance"] = np.where(
#     X_validation["weather"] == "rain", X_validation["distance_km"], 0
# )

# X_test["rainy_distance"] = np.where(
#     X_test["weather"] == "rain", X_test["distance_km"], 0
# )


# let scaled data for machine learning to understand easier
scaler = ManualStandardScaler()
train_scaled_distance = scaler.fit_transform(X_train["distance_km"])
validation_scaled_distance = scaler.transform(X_validation["distance_km"])
test_scaled_distance = scaler.transform(X_test["distance_km"])

# let encode data for machine learning to understand easier
encode = ManualOneHotEncoder()
train_encoded_weather = encode.fit_transform(X_train["weather"])
validation_encoded_weather = encode.transform(X_validation["weather"])
test_encoded_weather = encode.transform(X_test["weather"])

# # scale the rainy distance feature
# rainy_distance_scaler = ManualStandardScaler()
# train_scaled_rainy_distance = rainy_distance_scaler.fit_transform(
#     X_train["rainy_distance"]
# )
# validation_scaled_rainy_distance = rainy_distance_scaler.transform(
#     X_validation["rainy_distance"]
# )
# test_scaled_rainy_distance = rainy_distance_scaler.transform(X_test["rainy_distance"])


# matrix for machine learning
X_train_matrix = np.column_stack((train_scaled_distance, train_encoded_weather))
X_validation_matrix = np.column_stack(
    (
        validation_scaled_distance,
        validation_encoded_weather,
    )
)
X_test_matrix = np.column_stack((test_scaled_distance, test_encoded_weather))

X_train_model = X_train_matrix[:, :2]
X_validation_model = X_validation_matrix[:, :2]
X_test_model = X_test_matrix[:, :2]


if __name__ == "__main__":
    # Manual model
    model = ManualLinearRegression()
    model.fit(X_train_model, Y_train)
    manual_pred = model.predict(X_validation_model)

    print("=== Manual Linear model ===")
    print("weights:", model.weights)
    print("bias:", model.bias)
    print("validation predictions:", manual_pred)

    # Scikit-learn model
    sk_model = LinearRegression()
    sk_model.fit(X_train_model, Y_train)
    sk_pred = sk_model.predict(X_validation_model)

    print("\n=== Scikit-learn Linear model ===")
    print("coef_:", sk_model.coef_)
    print("intercept_:", sk_model.intercept_)
    print("validation predictions:", sk_pred)

    knn_model = ManualKNNRegressor(k=2)
    knn_model.fit(X_train_model, Y_train)
    knn_pred = knn_model.predict(X_validation_model)

    print("\n=== Manual Knn model ===")
    print("validation predictions:", knn_pred)

    print("\n=== KNN neighbor investigation ===")
    for query in X_validation_model:
        distances = np.sqrt(np.sum((X_train_model - query) ** 2, axis=1))

        nearest_indices = np.argsort(distances)[: knn_model.k]

        print("\nQuery:", query)
        print("Nearest distances:", distances[nearest_indices])
        print("Neighbor targets:", Y_train.iloc[nearest_indices].to_list())

    print("\n===Neighbor indices===")
    print("Neighbor indices:", nearest_indices)
    print("Neighbor feature rows:", X_train.iloc[nearest_indices].to_dict("records"))

    sk_kkn_model = KNeighborsRegressor(n_neighbors=2)
    sk_kkn_model.fit(X_train_model, Y_train)
    sk_knn_pred = sk_kkn_model.predict(X_validation_model)

    print("\n=== Manual sk Knn model ===")
    print("validation predictions:", sk_knn_pred)

    print("\n=== Actual validation targets ===")
    print(Y_validation.to_list())
