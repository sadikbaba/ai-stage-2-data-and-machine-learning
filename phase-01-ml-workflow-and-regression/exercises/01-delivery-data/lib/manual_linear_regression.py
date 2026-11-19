# transpose(X)

# Matrix multiplication for XᵀX

# Matrix-vector multiplication for Xᵀy

# minor_matrix(A, row_to_remove, col_to_remove)

# determinant()

# Cofactor matrix

# Adjugate and inverse()

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

    def fit(self, X, Y):
        pass


# test
if __name__ == "__main__":
    model = ManualLinearRegression()
    tests = [[[2, 0], [4, 1], [6, 0], [8, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], []]
    for x in tests:
        transpose = model.transpose(x)
        # print(transpose)

    a = [[2, 4], [6, 8]]
    b = model.transpose(a)
  

    matrix_mul = model.matrix_multiplication(b, a)
    print(matrix_mul)

    a = np.array(a)
    b = np.array(b)
    print(b@a)


"""

Xt = [[2, 4, 6, 8],
      [0, 1, 0, 1]]

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
