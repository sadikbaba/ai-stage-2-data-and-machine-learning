from manual_main import (  # noqa: I001
    test_scaled_distance,
    train_scaled_distance,
    validation_scaled_distance,
    # encoded weather
    encode,
    train_encoded_weather,
    validation_encoded_weather,
    test_encoded_weather,
    # matrix
    X_train_matrix,
    X_validation_matrix,
    X_test_matrix,
)

if __name__ == "__main__":
    print("\nManual scaled distance in km:")
    print("Train Scaled distance:", train_scaled_distance)
    print("\nValidation Scaled distance", validation_scaled_distance)
    print("\nTest Scaled distance:", test_scaled_distance)

    print("\nManual Encoded weather:")
    print("Learned weather categories:", encode.categories)
    print("\nTrain encoded wether:", train_encoded_weather)
    print("\nValidation encoded wether:", validation_encoded_weather)
    print("\nTest encoded wether:", test_encoded_weather)

    print("\nMatrix shape X")
    print(X_train_matrix.shape)
    print(X_validation_matrix.shape)
    print(X_test_matrix.shape)

    print("\nTraining matrix:\n", X_train_matrix)
    print("\nValidation matrix:\n", X_validation_matrix)
    print("\nTest matrix:\n", X_test_matrix)
