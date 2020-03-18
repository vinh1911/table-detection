from cv2 import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#read your file
file=r'input/1.jpg'
img = cv2.imread(file)
# Convert to grayscale and apply Gaussian filtering
# im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_grey, (5, 5), 0)
# ret, im_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plotting = plt.imshow(img,cmap='gray')
plt.show()

out = pytesseract.image_to_string(img,lang='eng',config=" -c tessedit_char_whitelist=.0123456789")
print(out)