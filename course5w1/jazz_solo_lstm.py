from __future__ import print_function

import time

import IPython
import sys
from music21 import *
import numpy as np
from .lstm_network.grammar import *
from .lstm_network.qa import *
from .lstm_network.preprocess import *
from .lstm_network.music_utils import *
from .lstm_network.data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import pygame


def play_mp3_music(filename='course5w1/lstm_network/data/30s_seq.mp3'):
    # this is only for wav not mp3
    # sounda = pygame.mixer.Sound("course5w1/lstm_network/data/30s_seq.mp3")
    # sounda.play()
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play(loops=1)
    time.sleep(2)
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def init_lstm(n_values, n_a):
    reshapor = Reshape((1, 78))  # Used in Step 2.B of djmodel(), below
    LSTM_cell = LSTM(n_a, return_state=True)  # Used in Step 2.C
    densor = Dense(n_values, activation='softmax')  # Used in Step 2.D
    return reshapor, LSTM_cell, densor


def load_music():
    X, Y, n_values, indices_values = load_music_utils()
    print('shape of X:', X.shape)
    print('number of training examples:', X.shape[0])
    print('Tx (length of sequence):', X.shape[1])
    print('total # of unique values:', n_values)
    print('Shape of Y:', Y.shape)
    return X, Y, n_values, indices_values


def djmodel(reshapor, LSTM_cell, densor, Tx, n_a, n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """

    # Define the input of your model with a shape
    X = Input(shape=(Tx, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    ### START CODE HERE ###
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []

    # Step 2: Loop
    for t in range(Tx):
        # Step 2.A: select the "t"th time step vector from X.
        x = Lambda(lambda x: X[:, t, :])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)

    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model


def jazz_solo_lstm_practice():
    # play_mp3_music('course5w1/lstm_network/data/30s_seq.mp3')
    X, Y, n_values, indices_values = load_music()
    n_a = 64
    reshapor, LSTM_cell, densor = init_lstm(n_values, n_a)
    model = djmodel(reshapor, LSTM_cell, densor, Tx=30, n_a=64, n_values=78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit([X, a0, c0], list(Y), epochs=100)
    print("jazz")
