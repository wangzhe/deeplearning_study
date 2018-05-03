import numpy as np


def _init():
    X = None
    x = None
    a = None
    b = None
    return


a = np.random.randn(5)
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))

print("-------unsuitable example above-----------------")
print("-------the right way to do below----------------")

a = np.random.randn(5, 1)
print(a.shape)
print(a)
print(a.T)

print("----user assert to throw out exceptions---------")
assert (a.shape == (5, 1))
print("----the result---------")
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a * b
print(c)

print("----normalizing rows---------")
print("----Although I have no idea about what is norm right now---------")
x = np.array([[0, 3, 4], [2, 6, 4]])
x_norm = np.linalg.norm(x, axis=1, keepdims=True)
print(x_norm)

print("---new understanding of sum--------")
_init()
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0, 0]])
# Apply exp() element-wise to x. Use np.exp(...).
x_exp = np.exp(x)

# Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
# axis = 1 means do sum for rows
x_sum = np.sum(np.exp(x), axis=1, keepdims=True)
print(x_sum)

# Dot Product to generate real number from w1x1 + w2x2 + .... + wnxn
# Outer Product to generate matrix (m*n) from (1, m) by (1, n)
# Element-wise Product to generate matrix (1*n) n= len(x) [w1x1, w2x2, w3x3, .... , wnxx]
# General Dot (gdot) Product to generate real number from a product of [ [w11, w12, ..., w1n], by [x1, x2, ..., xn]
#                                                                        [w21, w22, ..., w2n]
#                                                                        [w31, w32, ..., w3n]]
#                                                           result is  [ [w11x1 + w12x2 + ... + w1nxn]
#                                                                        [w21x1 + w22x2 + ... + w2nxn]
#                                                                        [w31x1 + w32x2 + ... + w3nxn]]
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
res = np.sum(np.abs(yhat - y))
print(res)

print("----Learning about reshape---------")
# to get the number of training examples
_init()
X = np.array([[[[1, 2], [2, 3], [3, 4]],
               [[4, 5], [5, 6], [6, 7]],
               [[7, 8], [8, 9], [9, 10]],
               [[10, 11], [11, 12], [12, 13]]
               ],
              [[[1, 2], [2, 3], [3, 4]],
               [[4, 5], [5, 6], [6, 7]],
               [[7, 8], [8, 9], [9, 10]],
               [[10, 11], [11, 12], [12, 13]]
               ],
              [[[1, 2], [2, 3], [3, 4]],
               [[4, 5], [5, 6], [6, 7]],
               [[7, 8], [8, 9], [9, 10]],
               [[10, 11], [11, 12], [12, 13]]
               ],
              [[[1, 2], [2, 3], [3, 4]],
               [[4, 5], [5, 6], [6, 7]],
               [[7, 8], [8, 9], [9, 10]],
               [[10, 11], [11, 12], [12, 13]]
               ],
              [[[1, 2], [2, 3], [3, 4]],
               [[4, 5], [5, 6], [6, 7]],
               [[7, 8], [8, 9], [9, 10]],
               [[10, 11], [11, 12], [12, 13]]
               ]
              ])
print(X.shape)
# X_flatten = X.reshape(X.shape[0], -1).T
# print(X_flatten.shape)
x1 = X[0]
x1_flat = x1.reshape(1, -1)  # So -1 means all the others, looks easy
print(x1_flat.shape)
print(x1_flat)

print("----standardize dataset---------")
_init()
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5]])
y = np.mean(x)
print(y)

X = np.array([[1., 2., -1.], [3., 4., -3.2]])
print(X.shape)
m = X.shape[1]
print(m)
