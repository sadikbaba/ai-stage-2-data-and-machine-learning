class ManualLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, Y, epochs=1000, learning_rate=0.01):
        number_of_rows = len(X)
        number_of_columns = len(X[0])

        self.weights = [0.0] * number_of_columns
        self.bias = 0.0

        # gradient descent for every column
        for epoch in range(epochs):
            for row_index in range(number_of_rows):
                row = X[row_index]
                actual = Y[row_index]

                # prediction
                pred_y = self.bias
                for col in range(number_of_columns):
                    pred_y += self.weights[col] * row[col]

                # error
                error = pred_y - actual

                # update every weight
                for col in range(number_of_columns):
                    gradient_weight = error * row[col]
                    self.weights[col] = self.weights[col] - learning_rate * gradient_weight

                # update bias
                self.bias = self.bias - learning_rate * error

    def predict(self, X):
        predictions = []
        for row in X:
            y_pred = self.bias
            for col in range(len(row)):
                y_pred += self.weights[col] * row[col]
            predictions.append(y_pred)
        return predictions