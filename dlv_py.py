import numpy as np
import time

a = np.array([1, 2, 3, 4])
print(a)

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Vectorized version:" + str(1000 * (toc - tic)) + "ms")
print(a)
print(c)

c = 0
tic = time.time()
for i in range(a.size):
    c += a[i] * b[i]
toc = time.time()

print("For-loop version:" + str(1000 * (toc - tic)) + "ms")
print(c)


a = np.random.rand(1000000)
b = np.random.rand(1000000)
a1 = a.reshape(1, -1)
b1 = b.reshape(-1, 1)
c = np.dot(a1, b1)
print(a1.shape)
print(b1.shape)
