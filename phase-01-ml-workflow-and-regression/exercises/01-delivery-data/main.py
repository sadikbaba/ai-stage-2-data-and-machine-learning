import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

delivery_data = pd.read_csv("delivery.csv")

x = delivery_data[["distance_km", "weather"]]
y = delivery_data["delivery_minutes"]

# First split: reserve 2 rows for testing
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Second split: reserve 2 of the remaining 8 for validation
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.25, random_state=42
)

median_distance = X_train["distance_km"].median()

# fill the missing data in distance_km colum of X_test
X_test["distance_km"] = X_test["distance_km"].fillna(median_distance)

# encoder
encoder = OneHotEncoder(handle_unknown="ignore")
# encode training
encoder.fit(X_train[["weather"]])

# encoded weather
train_encoded_weather = encoder.transform(X_train[["weather"]])
validation_encoded_weather = encoder.transform(X_validation[["weather"]])
test_encoded_weather = encoder.transform(X_test[["weather"]])

# scaler
scaler = StandardScaler()
# scaler training
scaler.fit(X_train[["distance_km"]])

# uses those learned numbers to convert distances into standardized values.
train_scaler_distance = scaler.transform(X_train[["distance_km"]])
validation_scaler_distance = scaler.transform(X_validation[["distance_km"]])
test_scaler_distance = scaler.transform(X_test[["distance_km"]])
