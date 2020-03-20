import cv2
from keras.models import load_model
import tensorflow as tf
import numpy as np

# model = load_model('mnist_keras.h5')
# model._make_predict_function()

graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        model = load_model('mnist_keras_cnn_model.h5')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

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
    
    im_th = 255 - img
    im_th = cv2.dilate(im_th, kernel, iterations=3)
    # Find contours in the image
    contours,hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    contours, boundingBoxes = sort_contours(contours)
    digits = ''
    for rect in contours:

        # Draw the rectangles
        x,y,w,h = cv2.boundingRect(rect)

        #this could vary depending on the image you are trying to predict
        #trying to extract ONLY the rectangles with the digit in it
        #ex: there could be a bounding box inside every 6,9,8 because of the circle in the number's shape - you don't want that.
        #read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html

        if (25 <= w <= 200) and (35 <= h <= 200):
            roi = im_th[y:y+h,x:x+w]
            roi = cv2.resize(roi,(28, 28))
            if roi.size > 0:
                roi = roi.reshape((1, 28, 28, 1)).astype('float32')
                with graph.as_default():
                    with session.as_default():
                        nbr = model.predict_classes(roi)
                digits += str(nbr[0])
    return digits