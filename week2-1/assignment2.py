import numpy as np

# # GRADED FUNCTION: compute_cost_with_regularization
#
# def compute_cost_with_regularization(A3, Y, parameters, lambd):
#     """
#     Implement the cost function with L2 regularization. See formula (2) above.
#
#     Arguments:
#     A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
#     Y -- "true" labels vector, of shape (output size, number of examples)
#     parameters -- python dictionary containing parameters of the model
#
#     Returns:
#     cost - value of the regularized loss function (formula (2))
#     """
#     m = Y.shape[1]
#     W1 = parameters["W1"]
#     W2 = parameters["W2"]
#     W3 = parameters["W3"]
#
#     cross_entropy_cost = compute_cost(A3, Y)  # This gives you the cross-entropy part of the cost
#
#     ### START CODE HERE ### (approx. 1 line)
#     L2_regularization_cost = None
#     ### END CODER HERE ###
#
#     cost = cross_entropy_cost + L2_regularization_cost
#
#     return cost

A1 = np.array([[1, 2, 3], [2, 3, 4]])
print(A1.shape)
D1 = np.random.rand(A1.shape[0], A1.shape[1])
D1 = D1 < 0.5

print(D1)
