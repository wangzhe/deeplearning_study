from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from .mt_attention.nmt_utils import *
import matplotlib.pyplot as plt


# human_vocab: a python dictionary mapping all characters used in the human readable dates to an integer-valued index
# machine_vocab: a python dictionary mapping all characters used in machine readable dates to an integer-valued index.
#                These indices are not necessarily consistent with human_vocab.
# inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to characters.
def init(m):
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
    print(human_vocab)
    print(machine_vocab)
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Xoh.shape:", Xoh.shape)
    print("Yoh.shape:", Yoh.shape)
    return dataset, X, Y, Xoh, Yoh, human_vocab, machine_vocab, inv_machine_vocab


def look_at_examples(dataset, X, Y, Xoh, Yoh, index):
    print("Source date:", dataset[index][0])
    print("Target date:", dataset[index][1])
    print()
    print("Source after preprocessing (indices):", X[index])
    print("Target after preprocessing (indices):", Y[index])
    print()
    print("Source after preprocessing (one-hot):", Xoh[index])
    print("Target after preprocessing (one-hot):", Yoh[index])


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###

    return context


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True, name='bidirectional_1'), merge_mode='concat')(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model


def categorical_mod(x, human_vocab):
    categorical = [[]]
    categorical[0] = to_categorical(x, num_classes=len(human_vocab))
    return categorical


def translation_with_attention_practice():
    global Tx, Ty, repeator, concatenator, densor1, densor2, activator, dotor
    global post_activation_LSTM_cell, output_layer

    print("translation with attention")

    # Part 1
    Tx = 30
    Ty = 10
    m = 10000

    dataset, X, Y, Xoh, Yoh, human_vocab, machine_vocab, inv_machine_vocab = init(m)
    look_at_examples(dataset, X, Y, Xoh, Yoh, index=0)

    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax,
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)
    # Part 2
    n_a = 32
    n_s = 64
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(len(machine_vocab), activation=softmax)

    my_model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    # Training
    # opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1)
    # my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    # outputs = list(Yoh.swapaxes(0, 1))
    # my_model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

    my_model.load_weights('course5w3/mt_attention/models/model.h5')
    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source_prev = list(map(lambda x: categorical_mod(x, human_vocab), source))
        # print(source_prev[0])
        source = np.array(source_prev).swapaxes(0, 1)
        prediction = my_model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output))
