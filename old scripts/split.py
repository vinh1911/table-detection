from cv2 import cv2
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

#read your file
file=r'input/example3.jpg'
img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
img.shape

#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find Canny edges 
edged = cv2.Canny(img_bin, 30, 200) 
cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(img, contours, 2, (0, 255, 0), 3) 
  
cv2.imshow('Contours', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 