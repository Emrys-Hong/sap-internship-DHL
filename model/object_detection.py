import os
import cv2
import numpy as np
import sys
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
from pytesseract import image_to_string
import pandas as pd
from os import listdir
from os.path import isfile, join
import cv2
import math
import string
from collections import defaultdict
import re
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.cluster import KMeans
from skimage.filters import threshold_local
import tensorflow as tf
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
from .utils import label_map_util
from .utils import visualization_utils as vis_util
import imutils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input, decode_predictions
import fire

def get_address_tag_coordinates(image, boxes, classes, scores):
    im_width, im_height = image.shape[1], image.shape[0]
    # create list to store coordinates
    coordinate_list = []
    # we assume that there cannot be over 5 objects in the boxes that have a score over 80%
    for i in range(0,5):
        if classes[0][i] == 2 and scores[0][i] >= 0.8:
            coordinate_list.append(boxes[0][i])
    # create the list that output the final answer
    final_list = []
    for coordinates in coordinate_list:
        ymin, xmin, ymax, xmax = coordinates
        (left, top, right, bottom) = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
        final_list.append([left, top, right, bottom])
    return final_list

def get_bar_code_coordinates(image, boxes, classes, scores):
    im_width, im_height = image.shape[1], image.shape[0]
    # create list to store coordinates
    coordinate_list = []
    # we assume that there cannot be over 5 objects in the boxes that have a score over 80%
    for i in range(0,5):
        if classes[0][i] == 1 and scores[0][i] >= 0.8:
            coordinate_list.append(boxes[0][i])
    # create the list that output the final answer
    final_list = []
    for coordinates in coordinate_list:
        ymin, xmin, ymax, xmax = coordinates
        (left, top, right, bottom) = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
        final_list.append([left, top, right, bottom])
    return final_list

def object_detection(PATH_TO_IMAGE):
    
    """change the test image path in PATH_TO_IMAGE"""
    print(PATH_TO_IMAGE)
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join('results','frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join('results/','label_map.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 2

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph, config=tfconfig)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # contains coordinate of text
    address_tag = get_address_tag_coordinates(image, boxes, classes, scores)
    bar_code = get_bar_code_coordinates(image, boxes, classes, scores)
    # All the results have been drawn on image. Now display the image.
    return address_tag, bar_code

if __name__ == '__main__':
    fire.Fire(object_detection)
