import numpy as np
import pandas as pd
from lib.manual_onehot import ManualOneHotEncoder
from lib.manual_scaler import ManualStandardScaler
from sklearn.model_selection import train_test_split

delivery_data = pd.read_csv("delivery.csv")

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


X_train_matrix = np.column_stack((train_scaled_distance, train_encoded_weather))
X_validation_matrix = np.column_stack(
    (validation_scaled_distance, validation_encoded_weather)
)
X_test_matrix = np.column_stack((test_scaled_distance, test_encoded_weather))
