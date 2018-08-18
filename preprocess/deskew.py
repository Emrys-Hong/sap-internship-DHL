import os
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import pandas as pd
from os import listdir
from os.path import isfile, join
import cv2
import math
import string
from collections import defaultdict
import re
from collections import Counter
from sklearn.cluster import KMeans
from skimage.filters import threshold_local
import numpy as np
import fire
import imutils


def compute_skew(image, index):
    
    ### two ways:
    # 1. high threshold one value for canny
    # 2. adaptive filter plus canny
    
    height, width = image.shape
    
    ## first par: whether connect the dots together. second par: should consider it as dots? 3rd: apture parameters
    edges = cv2.Canny(image, 800, 200, 3, 5)
    
    
    #########
    # 1. angle clusters
    #########
    """PARAMETERS:
    2nd rho 3rd theta parameter for accuracy, rho in pixel and theta in rad
    4th voting threshold, pixel.
    5th min length for the line to be considered a line
    6th max width for line to vary"""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength= max(width,height) / 2, maxLineGap=30)   ##### sort the length
    
    ## if did not get any line, it will return a None, and we lower the threshold
    try:    
        if lines == None:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength= max(width,height) / 2, maxLineGap=30)
        if len(lines) <= 10:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength= max(width,height) / 2, maxLineGap=30)
    except Exception:
        pass
    
    ## if number does not satisfy certain value we also change threshold for 150
       
    if len(lines) <= 10:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength= min(width,height) / 2, maxLineGap=30)

    ## prepare X for clustering
    X = []
    for coordinates in lines:
        for x1, y1, x2, y2 in coordinates:
            angle = math.atan2((y2-y1),(x2-x1))/math.pi*180
            X.append(angle)
    
    ## finding the cluster using kmeans
    angle_list = np.array(X).reshape(-1,1)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(angle_list)
    
    ## find the average of the biggest cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    count_dict = dict(zip(unique, counts))
    max_ocur_key = [key for key, value in count_dict.items() if value == max(list(count_dict.values()))][0]
    mylist = []
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == max_ocur_key:
            mylist.append(angle_list[i])
    
    rotate_angle = sum(mylist)/len(mylist)
    
    return rotate_angle

def isFlipped(image):
    height, width = image.shape
    
    ## first par: whether connect the dots together. second par: should consider it as dots? 3rd: apture parameters
    edges = cv2.Canny(image, 800, 200, 3, 5)
    left_part = edges[:,:int(width/2)]
    upper_part = edges[:int(height/2),:]
    right_part = edges[:, -int(width/2):]
    lower_part = edges[int(height/2):,:]
    up_left_pixel = np.count_nonzero(left_part) + np.count_nonzero(upper_part)*0.5
    down_right_pixel = np.count_nonzero(right_part) + np.count_nonzero(lower_part)*0.5
    return down_right_pixel > up_left_pixel


def process(image):
    img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2) 
    
    """ grab the rotation matrix (applying the negative of the
     angle to rotate clockwise), then grab the sine and cosine
     (i.e., the rotation components of the matrix) """
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),borderMode=0, borderValue=(255,255,255))


def rotateImg(image):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # sometimes cannot detect anyline so need to try
    try:
        angle = compute_skew(img, 1)
        # commented process for generating training data
        img = process(img)
        img = rotate_bound(img, -angle)
        if isFlipped(img):
            img = rotate_bound(img, 180)
    except Exception as e:
        # print index if cannot detect anyline
        print('rotate wrong: ', index)
        img = process(image)
    return img

def rotateBarcode(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # sometimes cannot detect anyline so need to try
    try:
        angle = compute_skew(img, 1)
        # commented process for generating training data
        img = process(img)
        img = rotate_bound(img, -angle)
    except Exception as e:
        # print index if cannot detect anyline
        print('rotate wrong: ', index)
        img = process(image)
    return img

def rotateImg_color(image, index, address):
    
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # sometimes cannot detect anyline so need to try
    try:
        angle = compute_skew(img1, index)
        # commented process for generating training data
        img1 = process(img1)
        img1 = rotate_bound(img1, -angle)
        image = rotate_bound(image, -angle)
        if isFlipped(img1):
            image = rotate_bound(image, 180)
        cv2.imwrite(address, image)
        return address
    except Exception as e:
        # print index if cannot detect anyline
        raise e
        print('rotate wrong: ', index)


if __name__ == '__main__':
    fire.Fire(rotateImg)
