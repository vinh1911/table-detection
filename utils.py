import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np
from scipy import ndimage
import math

graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        model = load_model('mnist_keras_995.h5')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,10))

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def process_digit(img):
    gray = img
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    return gray

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def predict_number(img):
    
    img_invert = 255 - img
    im_th = cv2.dilate(img_invert, kernel, iterations=1)

    # Find contours in the image
    contours,hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    contours, boundingBoxes = sort_contours(contours)
    digits = ''
 
    for rect in contours:
        # Draw the rectangles
        x,y,w,h = cv2.boundingRect(rect)
        if 2 < w < 40:
            digits += '.'
        elif w >= 40 :
            roi = img_invert[y:y+h,x:x+w]
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            roi = process_digit(roi)
            roi = roi.reshape((1, 28, 28, 1)).astype('float32')
            with graph.as_default():
                with session.as_default():
                    nbr = model.predict_classes(roi)
            digits += str(nbr[0])
        # cv2.imshow('hihi',img)
        # cv2.waitKey(0)
    return digits