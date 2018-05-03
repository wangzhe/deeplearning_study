import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
print(A)

cal = A.sum(axis=0)  # axis=0 means vertical summary
# print(cal)
# print(A.reshape(12, 1))
print("----------------------------------")
print(A[1])

percentage = 100 * A / cal.reshape(1, 4)
print(percentage)

print("-------------matrix by real number--------------")
B = np.array([1, 2, 3])
res = B + 100
print("1 by 3 vector：", res)
res = B.reshape(3, 1) + 100
print("3 by 1 vector：", res)
res = A + 100
print("m by n matrix:", res)
