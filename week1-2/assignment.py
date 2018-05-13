import numpy as np
import matplotlib.pyplot as plt
# import h5py
import scipy
from scipy import ndimage
from lr_utils import load_dataset


def load_data(with_test=False):
    # Loading the data (cat/non-cat)
    train_x_orig, train_y, test_x_orig, test_y, the_classes = load_dataset()
    if with_test:
        # Example of a picture
        index = 23
        plt.imshow(train_x_orig[index])
        print("y = " + str(train_y[:, index]) + ", it's a '" + the_classes[np.squeeze(train_y[:, index])].decode(
            "utf-8") + "' picture.")
        m_train = train_x_orig.shape[0]
        m_test = test_x_orig.shape[0]
        num_px = train_x_orig.shape[1]

        print("Number of training examples: m_train = " + str(m_train))
        print("Number of testing examples: m_test = " + str(m_test))
        print("Height/Width of each images: num_px = " + str(num_px))
        print("Each images is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        plt.show()
    return train_x_orig, train_y, test_x_orig, test_y, the_classes


def standardize(train_set_x_orig_, test_set_x_orig_):
    train_set_x_flatten = train_set_x_orig_.reshape(train_set_x_orig_.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig_.reshape(test_set_x_orig_.shape[0], -1).T
    return train_set_x_flatten / 255., test_set_x_flatten / 255.


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(X.T, w) + b).reshape(1, -1)  # compute activation
    cost = - (1 / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis=1, keepdims=True)  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(X.T, w) + b).reshape(1, -1)
    ### END CODE HERE ###

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0][i] < 0.50:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
        ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def plot_learning_curve(d_array, plot_curve_=False):
    if plot_curve_:
        costs = np.squeeze(d_array['costs'])
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(d_array["learning_rate"]))
        plt.show()


def different_learning_rate(show_different_learning_rate_curve_=False):
    if show_different_learning_rate_curve_:
        learning_rates = [0.01, 0.001, 0.0001]
        models = {}
        for i in learning_rates:
            print("learning rate is: " + str(i))
            models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500,
                                   learning_rate=i, print_cost=False)
            print('\n' + "-------------------------------------------------------" + '\n')

        for i in learning_rates:
            plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

        plt.ylabel('cost')
        plt.xlabel('iterations (hundreds)')

        legend = plt.legend(loc='upper center', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        plt.show()


def test_img(test_my_data=False):
    if test_my_data:
        my_image = "la_defense.jpg"  # change this to the name of your image file

        # We preprocess the image to fit your algorithm.
        fname = "images/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        num_px = train_set_x_orig.shape[1]
        my_image_mid = scipy.misc.imresize(image, size=(num_px, num_px))
        my_image = my_image_mid.reshape((1, num_px * num_px * 3)).T
        my_predicted_image = predict(d["w"], d["b"], my_image)

        plt.imshow(image)
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
            int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
        plt.show()


if __name__ == '__main__':
    visualizing_data = False
    test_my_data = True
    show_learning_curve = False
    show_different_learning_rate_curve = False

    print("Ready...go...")
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data(visualizing_data)
    train_set_x, test_set_x = standardize(train_set_x_orig, test_set_x_orig)
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    # Test my data
    test_img(test_my_data)

    # Further analysis
    plot_learning_curve(d, show_learning_curve)
    different_learning_rate(show_different_learning_rate_curve)
