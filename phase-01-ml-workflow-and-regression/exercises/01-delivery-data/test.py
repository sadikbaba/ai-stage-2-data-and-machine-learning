from main import *

if __name__ == "__main__":
    print("\nData:")
    print(delivery_data.head())

    print("\nShape of the data:")
    print(delivery_data.shape)

    print("\nFeatures X:")
    print(x.head(2))

    print("\nTarget y:")
    print(y.head(2))


    " Split the data into training and test sets "

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
