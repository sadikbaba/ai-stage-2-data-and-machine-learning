import numpy as np


class ManualKNNRegressor:
    def __init__(self, k=2):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.X_train = None
        self.y_train = None
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X).tolist()
        y = np.asarray(y).tolist()

        if not X or not y:
            raise ValueError("Training data must not be empty")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before predict")

        X = np.asarray(X).tolist()

        if not X:
            return []

        # formula = d(X,A) = sqrt(((X1 - A1)**2) + ((X2- A2)**2))
        # take the of Num k sum them and divide by k

        predictions = []
        for query_row in X:
            row_distances = []
            for train_row in self.X_train:
                squared_distance = 0
                for f in range(len(query_row)):
                    squared_distance += (query_row[f] - train_row[f]) ** 2
                distance = (squared_distance) ** 0.5
                row_distances.append(distance)

            sorted_indices = np.argsort(row_distances).tolist()
            num_neighbors = min(self.k, len(sorted_indices))

            neighbor_indices = []
            for i in range(num_neighbors):
                neighbor_indices.append(sorted_indices[i])

            neighbor_targets = []
            for idx in neighbor_indices:
                neighbor_targets.append(self.y_train[int(idx)])

            prediction = sum(neighbor_targets) / len(neighbor_targets)
            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    X = [[2, 0], [4, 1]]
    y = [1, 2]

    model = ManualKNNRegressor(k=1)
    model.fit(X, y)

    queries = [[2, 0], [6, 0]]

    print(model.predict(queries))

    for value in [0, -1, 1.5, "2", True]:
        try:
            ManualKNNRegressor(k=value)
        except (TypeError, ValueError) as error:
            print(value, type(error).__name__)

    for value in [1, 2]:
        model = ManualKNNRegressor(k=value)
        print(value, "accepted")


"""

list calculate the distance by hand
formula = d(X,A) = sqrt(((X1 - A1)**2) + ((X2- A2)**2))

A = [[2, 0],
    [4, 1]]

X = [[6, 0]]


    (6 - 2 )^2 + (0 - 0)^2 
   = (4)^2 +  0
   = sqrt(16)
   = 4

   


    (6 - 4)^2 + (0 - 1)^2
   = (2)^2 + (1)^2  
   = 5
   = sqrt(5)
   = 2.236



"""
