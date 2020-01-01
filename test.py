# Import stuff
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import pandas as pd
import cv2
import numpy as np

# Config and variables
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
img_path = 'table.png'
# custom_oem_psm_config = r'-c preserve_interword_spaces=1x1 --psm 1 --oem 1'
# columns_name = ['Date','Before Breakfast','After Breakfast','Before Lunch','After Lunch','Before Dinner','After Dinner']

# # Simple image to string
# result = pytesseract.image_to_string(Image.open('Image_bin.png'),config=custom_oem_psm_config)
# print(result)

# text_file = open('output.txt', 'w')
# n = text_file.write(result)
# text_file.close()

# df = pd.read_table('output.txt',sep = '[ \t|]{2,}',engine = 'python')
# # df = pd.read_table('output.txt',sep = '[ \t|]{2,}',engine = 'python', skiprows=[0])
# # df.columns = columns_name
# df.to_csv('output.csv',index = False)
# df.to_excel('output.xlsx',index = False)

