# Baseline model: predict the mean of the training set for all validation samples.

from manual_main import Y_train, Y_validation

# Training
average_delivery_time = Y_train.mean()

# Prediction — plain Python list, one prediction per validation row
Y_pred = []
for _ in range(len(Y_validation)):
    Y_pred.append(average_delivery_time)

actuals = list(Y_validation)
errors = []

# Evaluation — Mean Absolute Error (pure Python)
# formula: MAE = (1/n) * sum(|actual - predicted|)
for i in range(len(Y_validation)):
    actual = actuals[i]
    predicted = Y_pred[i]
    error = abs(actual - predicted)
    errors.append(error)

mae = sum(errors) / len(errors)  # mean absolute error

if __name__ == "__main__":
    print("\nAverage delivery time (learned from train):")
    print(average_delivery_time)

    print("\nY_validation (actual values):")
    print(list(Y_validation))

    print("\nY_pred (baseline predictions):")
    print(Y_pred)

    print("\nEvaluation (MAE):")
    print(f"Baseline MAE: {mae:.2f} minutes")
