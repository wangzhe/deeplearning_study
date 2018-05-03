import numpy as np

n = 30
u = np.zeros((n, 1))

v = np.random.rand(10)

u = np.exp(v)

a = np.array([[1, 2, 3], [1, 2, 3]])
b = np.array([[7], [8]])
print(a.T.shape)
print(b.shape)
z = np.dot(a.T, b)
print(z)
