import os
import cv2
import numpy as np
import sys
sys.path.append("..")
from matplotlib import pyplot as plt
from PIL import Image
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

PATH = "/fs-object-detection/paperspace/data/pvsh/"
sz=224
batch_size=128
train_data_dir = f'{PATH}train'
validation_data_dir = f'{PATH}valid'
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2, zoom_range=1, horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model1 = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers: layer.trainable = False
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model1.fit_generator(train_generator, train_generator.n // batch_size, epochs=3, workers=4,
        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)
split_at = 200
for layer in model1.layers[:split_at]: layer.trainable = False
for layer in model1.layers[split_at:]: layer.trainable = True
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])   
model1.fit_generator(train_generator, train_generator.n // batch_size, epochs=5, workers=3,
        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)


def classify(address):
    img = image.load_img(address, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model1.predict(x)
    
    return preds[0][0] > 0.5


if __name__ == '__main__':
    fire.Fire(classify)
