import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    print(box_scores.shape)
    ### END CODE HERE ###

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.arg
    box_class_scores = None
    ### END CODE HERE ###


def yolo_filter_boxes_test():
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.shape))
        print("boxes.shape = " + str(boxes.shape))
        print("classes.shape = " + str(classes.shape))


def iou(box1, box2):
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###

    return iou


def iou_test():
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    print("iou = " + str(iou(box1, box2)))


def yolo_non_max_suppression_test():
    with tf.Session() as test_b:
        scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        print(scores.shape)
        boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
        classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)


# yolo_filter_boxes_test()
# iou_test()
yolo_non_max_suppression_test()

