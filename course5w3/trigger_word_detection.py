import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from .word_detection.td_utils import *
import pygame
import time


def play_mp3_music(filename='course5w3/word_detection/train.wav'):
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


def show_graph(filename='course5w3/word_detection/train.wav'):
    x = graph_spectrogram(filename)
    print("Time steps in input after spectrogram", x.shape)


def init(filename):
    _, data = wavfile.read(filename)
    print("Time steps in audio recording before spectrogram", data[:, 0].shape)


def trigger_word_detection_practice():
    print("welcome to word detection")
    filename = 'course5w3/word_detection/train.wav'
    # play_mp3_music()
    # show_graph()
    init(filename)

    Tx = 5511  # The number of time steps input to the model from the spectrogram
    n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
    Ty = 1375  # The number of time steps in the output of our model

    # Load audio segments using pydub
    activates, negatives, backgrounds = load_raw_audio()

    print("background len: " + str(len(backgrounds[0])))  # Should be 10,000, since it is a 10 sec clip
    print("activate[0] len: " + str(len(
        activates[0])))  # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
    print("activate[1] len: " + str(len(activates[1])))  # Different "activate" clips can have different lengths
