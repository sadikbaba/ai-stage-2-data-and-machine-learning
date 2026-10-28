from main import (  # noqa: I001
    X_test,
    X_train,
    X_validation,
    Y_test,
    Y_train,
    Y_validation,
    delivery_data,
    encoder,
    median_distance,
    scaler,
    test_encoded_weather,
    train_encoded_weather,
    validation_encoded_weather,
    x,
    y,
    train_scaler_distance,
    validation_scaler_distance,
    test_scaler_distance,
)

if __name__ == "__main__":
    print("\nData:")
    print(delivery_data.head())

    print("\nShape of the data:")
    print(delivery_data.shape)

    print("\nFeatures X:")
    print(x.head(2))

    print("\nTarget y:")
    print(y.head(2))

    # Split the data into training and test sets

    print("\nTrain-test split:")
    print(X_train.shape)
    print(X_test.shape)

    print("\nTrain-test split:")
    print(Y_train.shape)
    print(Y_test.shape)

    print("\nTest features:")
    print(X_test)

    print("\nTest targets:")
    print(Y_test)

    print("\nValidation features:")
    print("Training:", X_train.shape, Y_train.shape)
    print("Validation:", X_validation.shape, Y_validation.shape)
    print("Testing:", X_test.shape, Y_test.shape)

    print("\nMissing values in training:")
    print(X_train.isna().sum())

    print("\nMissing values in validation:")
    print(X_validation.isna().sum())

    print("\nMissing values in testing:")
    print(X_test.isna().sum())

    print("\nMedian Distance")
    print(median_distance)

    print("\nFilled the missed value in testing data")
    print(X_test)
    print(X_test.isna().sum())

    print("\nencoder category")
    print(encoder.categories_)

    print("\nencoded weather")
    print("\nTrain encoded:\n", train_encoded_weather.toarray())
    print("\nValidation encoded:\n", validation_encoded_weather.toarray())
    print("\nTest encoded:\n", test_encoded_weather.toarray())

    # scaler
    print("\nscaler")
    print(scaler.mean_)
    print(scaler.scale_)

    #  scaler transform

    print("\nTrain scaler:\n", train_scaler_distance.toarray())
    print("\nValidation scaler:\n", validation_scaler_distance.toarray())
    print("\nTest scaler:\n", test_scaler_distance.toarray())
