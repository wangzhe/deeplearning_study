import numpy as np
from .emojyfier.emo_utils import *
import emoji
import matplotlib.pyplot as plt


def init_test(X_train, Y_train, Y_oh_train):
    index = 1
    print(X_train[index], label_to_emoji(Y_train[index]))
    index = 50
    print(Y_train[index], "is converted into one hot", Y_oh_train[index])


def init_word_test(word_to_index, index_to_word):
    word = "cucumber"
    index = 289846
    print("the index of", word, "in the vocabulary is", word_to_index[word])
    print("the", str(index) + "th word in the vocabulary is", index_to_word[index])


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """

    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(word_to_vec_map[words[0]].shape)

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)

    ### END CODE HERE ###

    return avg


def sentence_to_avg_test(word_to_vec_map):
    avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
    print("avg = ", avg)


def some_middle_test(X_train, Y_train):
    print(X_train.shape)
    print(Y_train.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(X_train[0])
    print(type(X_train))
    Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
    print(Y.shape)

    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
                    'Lets go party and drinks', 'Congrats on the new job', 'Congratulations',
                    'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
                    'You totally deserve this prize', 'Let us go play football',
                    'Are you down for football this afternoon', 'Work hard play harder',
                    'It is suprising how people can be dumb sometimes',
                    'I am very disappointed', 'It is the best day in my life',
                    'I think I will end up alone', 'My life is so boring', 'Good job',
                    'Great so awesome'])

    print(X.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(type(X_train))


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 50  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples

            ### START CODE HERE ### (≈ 4 lines of code)
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = -np.sum(Y_oh[i] * np.log(a))
            ### END CODE HERE ###

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


def using_pred(X_train, Y_train, X_test, Y_test, W, b, word_to_vec_map):
    print("Training set:")
    pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    print('Test set:')
    pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)
    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

    pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    print_predictions(X_my_sentences, pred)


def emojify_practice():
    print("emojify practice works")
    X_train, Y_train = read_csv('course5w2/emojyfier/data/train_emoji.csv')
    X_test, Y_test = read_csv('course5w2/emojyfier/data/tesss.csv')
    maxLen = len(max(X_train, key=len).split())
    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)
    # init_test(X_train, Y_train, Y_oh_train)
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('course5w2/data/glove.6B.50d.txt')
    # init_word_test(word_to_index, index_to_word)
    # sentence_to_avg_test(word_to_vec_map)
    # some_middle_test(X_train, Y_train)
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    # print(pred)
    using_pred(X_train, Y_train, X_test, Y_test, W, b, word_to_vec_map)
