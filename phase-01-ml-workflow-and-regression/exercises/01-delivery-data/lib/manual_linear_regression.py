# transpose(X)
# matrix_multiplication(Xᵀ, X)
# matrix_vector_multiplication(Xᵀ, y)
# minor_matrix(A, row_to_remove, col_to_remove)
# determinant()
# cofactor_matrix()
# adjugate()
# inverse()
# fit()
# predict()

import numpy as np


class ManualLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def transpose(self, X):

        if not X or not X[0]:
            return []

        result = []
        for j in range(len(X[0])):
            row = []
            for i in range(len(X)):
                row.append(X[i][j])
            result.append(row)

        return result

    def matrix_multiplication(self, Xt, X):
        if not Xt or not Xt[0] or not X or not X[0]:
            return []

        rows_A = len(Xt)
        cols_A = len(Xt[0])

        rows_B = len(X)
        cols_B = len(X[0])

        if cols_A != rows_B:
            raise ValueError("Incompatible matrix dimensions for multiplication.")

        result = []
        for i in range(rows_A):
            new_row = []
            for j in range(cols_B):
                total = 0
                for k in range(cols_A):
                    total += Xt[i][k] * X[k][j]
                new_row.append(total)
            result.append(new_row)

        return result

    def matrix_vector_multiplication(self, Xt, y):

        if len(Xt[0]) != len(y):
            raise ValueError("Incompatible dimensions.")

        result = []
        for i in range(len(Xt)):
            total = 0
            for j in range(len(y)):
                total += Xt[i][j] * y[j]
            result.append(total)
        return result

    def minor_matrix(self, A, row_to_remove, col_to_remove):

        if not A:
            return []
        result = []
        for i in range(len(A)):
            if i == row_to_remove:
                continue
            new_row = []
            for j in range(len(A[i])):
                if j == col_to_remove:
                    continue
                new_row.append(A[i][j])
            result.append(new_row)
        return result

    def determinant(self, A):

        if not A:
            return 0

        n = len(A)
        if any(len(row) != n for row in A):
            raise ValueError("Matrix must be square")
        if n == 1:
            return A[0][0]

        if n == 2:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

        if n == 3:
            total = 0
            for i in range(len(A[0])):
                sign = -1 if i % 2 else 1
                minor = self.minor_matrix(A, 0, i)
                total += sign * A[0][i] * self.determinant(minor)
            return total

    def cofactor_matrix(self, A):

        result = []

        for i in range(len(A)):
            new_row = []
            for j in range(len(A[i])):
                minor = self.minor_matrix(A, i, j)
                ans = self.determinant(minor)
                sign = 0
                if (i + j) % 2 == 0:
                    sign = 1
                else:
                    sign = -1

                new_row.append(sign * ans)
            result.append(new_row)

        return result

    def adjugate(self, A):
        cofactor = self.cofactor_matrix(A)
        return self.transpose(cofactor)

    def inverse(self, A):
        det = self.determinant(A)
        if det == 0:
            raise ValueError("No inverse")

        adj = self.adjugate(A)
        result = []
        for i in range(len(adj)):
            new_row = []
            for j in range(len(adj[i])):
                new_row.append(adj[i][j] / det)
            result.append(new_row)
        return result

    def fit(self, X, Y):

        if not X or not X[0]:
            raise ValueError("X must contain samples and features")

        if len(X) != len(Y):
            raise ValueError("Number of samples in X must match number of targets in Y")
        for i in range(len(X)):
            sample = X[i]
            if len(sample) != len(X[0]):
                raise ValueError("All samples must have the same number of features")

        x_transpose = self.transpose(X)
        xtx = self.matrix_multiplication(x_transpose, X)
        xty = self.matrix_vector_multiplication(x_transpose, Y)
        xtx_inv = self.inverse(xtx)

        weights = self.matrix_vector_multiplication(xtx_inv, xty)

        mean_y = sum(Y) / len(Y)

        num_samples = len(X)
        num_features = len(X[0])
        mean_x = [0] * num_features

        for j in range(num_samples):
            for k in range(num_features):
                mean_x[k] += X[j][k]

        for k in range(num_features):
            mean_x[k] /= num_samples

        weighted_mean = 0
        for i in range(len(weights)):
            weighted_mean += weights[i] * mean_x[i]

        bias = mean_y - weighted_mean
        self.weights = weights
        self.bias = bias
        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        if not X:
            return []

        predictions = []
        for sample in X:
            if len(sample) != len(self.weights):
                raise ValueError(
                    "Each sample must have the same number of features as weights."
                )

            pred = self.bias
            for i in range(len(self.weights)):
                pred += self.weights[i] * sample[i]
            predictions.append(pred)

        return predictions


# test
if __name__ == "__main__":
    model = ManualLinearRegression()
    tests = [[[2, 0], [4, 1], [6, 0], [8, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], []]
    for x in tests:
        transpose = model.transpose(x)
        print(transpose)

    a = [[2, 4], [6, 8]]
    b = model.transpose(a)

    matrix_mul = model.matrix_multiplication(b, a)
    print(matrix_mul)

    a = np.array(a)
    b = np.array(b)
    print(b @ a)

    matrix = [[1, 2, 3], [4, 5, 6]]
    vector = [10, 20, 30]

    mvm = model.matrix_vector_multiplication(matrix, vector)
    print(mvm)
    # out put [140, 320]

    print("\n")
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    minor = model.minor_matrix(A, 0, 1)
    print(minor)

    # output [
    #     [4, 6],
    #     [7, 9]
    # ]

    A = [[2, 3], [4, 5]]
    print("\n")
    print(model.determinant(A))

    A = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
    print("\n")
    print(model.determinant(A))
    # output 22

    A = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
    print("\n")

    print(model.cofactor_matrix(A))

    # [
    # [24, 5, -4],
    # [-12, 3, 2],
    # [-2, -5, 4]
    # ]

    print("\n")
    print(model.adjugate(A))

    A = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]

    print("\n")
    print(model.inverse(A))

    X = [[2, 0], [4, 1], [6, 0], [8, 1]]
    Y = [12, 32, 35, 55]

    model.fit(X, Y)

    print("\n")
    print("Weights:", model.weights)
    print("Bias:", model.bias)

    print("predicted:", model.predict(X))


"""

Xt = [[2, 4, 6, 8],
      [0, 1, 0, 1]]
r
X  = [ [2, 0],
    [4, 1],
    [6, 0],
    [8, 1]
    ]

[2, 4, 6, 8] @ [2, 4, 6, 8] =  2 * 2 + 4 * 4 + 6 * 6 + 8 * 8 =  120
[2, 4, 6, 8] @ [0, 1, 0, 1] = 2 * 0 + 4 * 1 + 6 * 0 + 8 * 1 = 12
[0, 1, 0, 1] @  [2, 4, 6, 8] = 0 * 2 + 1 * 4 + 0 * 6 + 1 * 8 = 12
[0, 1, 0, 1] @ [0, 1, 0, 1] = 0 * 0 + 1 * 1 + 0 * 0 + 1 * 1 =2

[ [120, 12 ] ,
   [12, 2]
    ]

    

"""


"""
Xt.y

Xt = [[2, 4, 6, 8],
      [0, 1, 0, 1]]


y = [12,
    32, 
    35,
   55] 

[2, 4, 6, 8] @ [12,32, 35, 55] =  2 * 12 + 4 * 32 + 6 * 35 + 8 * 55 =  802
[0, 1, 0, 1] @ [ 12, 32, 35, 55] =  0 * 12 + 1 * 32 + 0 * 35 + 1 * 55 = 


[[802], 
[87]]


"""


"""

A = [ [120, 12 ] ,
      [12, 2]
    ]

det(A) = ad - bc
det(X^TX) = (120 x 2) - (12 x 12)

= 240 - 144
= 96

"""


"""
1. Swap the two diagonal values.
2. Change the signs of the other two values.
3. Divide every value by the determinant.


inverse = A^-1 = 1 / ad - bc [[d, -b], 
                                [-c, a]]

inverse A [[2, -12], 
            [-12 120]]

= 1/ (120* 2) - (-12 * 12)
= 240 - 144
= 96


3. 

2/96
-12/96
-12/96
120/96


[   [0.0208, -0.125 ] ,
    [-0.125, 1.25]
]

[   [0.0208333, -0.125 ] ,
    [-0.125, 1.25]
] @ [802, 87]

w1 = [0.0208, -0.125] @ [802, 87] = 0.0208 * 802 + (-0.125 * 87) = 5.8333
w2 = [-0.125, 1.25] @ [802, 87] = -0.125 * 802 + 1.25 * 87 =  8.5
"""

"""
bias

b = mean_y - (w1 mean_x1 + w2 mean_x2)


given

mean_x1 = 5
mean_x2 = 0.5
mean_y = 33.5
w1 = 5.8333
w2 = 8.5


b = 33.5 - ((5.8333*5) + (8.5 * 0.5))

= 33.5 -(29.1665 + 4.2) 
= 33.5 - 33.3665
= -0.1335

"""
