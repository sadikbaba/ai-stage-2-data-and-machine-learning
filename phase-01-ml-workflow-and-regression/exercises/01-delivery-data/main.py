
import pandas as pd
from sklearn.model_selection import train_test_split

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

